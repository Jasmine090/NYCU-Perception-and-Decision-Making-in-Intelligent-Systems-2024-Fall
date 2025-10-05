import numpy as np
import pandas as pd
import open3d as o3d
import argparse
from tqdm import tqdm
import os
import cv2
import copy
import itertools
from sklearn.neighbors import NearestNeighbors
import time

ta = 0.0
tb = 0.0

def ex_mat(alpha, beta, gamma, tx, ty, tz):
    r11 = np.cos(alpha)*np.cos(beta)
    r12 = np.cos(alpha)*np.sin(beta)*np.sin(gamma) - np.sin(alpha)*np.cos(gamma)
    r13 = np.cos(alpha)*np.sin(beta)*np.cos(gamma) + np.sin(alpha)*np.sin(gamma)
    r21 = np.sin(alpha)*np.cos(beta)
    r22 = np.sin(alpha)*np.sin(beta)*np.sin(gamma) + np.cos(alpha)*np.cos(gamma)
    r23 = np.sin(alpha)*np.sin(beta)*np.cos(gamma) - np.cos(alpha)*np.sin(gamma)
    r31 = -np.sin(beta)
    r32 = np.cos(beta)*np.sin(gamma)
    r33 = np.cos(beta)*np.cos(gamma)
    ex = np.array([[r11, r12, r13, tx],
    			  [r21, r22, r23, ty],
    			  [r31, r32, r33, tz],
    			  [0, 0, 0, 1]])
    return ex
    

def make_pcd(rgb, depth):

    pcd = o3d.geometry.PointCloud()
    hight, weight, focal = 512, 512, 256
    v, u = np.mgrid[0:hight, 0:weight]
    

    z = depth[:, :, 0].astype(np.float32) / 255 * (-10)  #convert depth map to meters
    x = -(u - weight*.5) * z / focal
    y = (v - hight*.5) * z / focal

    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    colors = (rgb[:, :, [2, 1, 0]].astype(np.float32) / 255).reshape(-1, 3)

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def depth_image_to_point_cloud(width, path):
    # TODO: Get point cloud from rgb and depth image 
    focal = width / (2 * np.tan(np.pi / 4))
    focal = float(focal)

    in_matrix = o3d.camera.PinholeCameraIntrinsic(width, width, focal, focal, width / 2.0, width / 2.0)
    n_img = len(os.listdir(path + "rgb/"))
    prog_bar = tqdm(desc='making point cloud...', total=n_img)
    for i in range(n_img):
        depth = cv2.imread(f"{path}/depth/{i}.png")
        rgb = cv2.imread(f"{path}/rgb/{i}.png")
        pcd = make_pcd(rgb, depth)
        o3d.io.write_point_cloud(path + 'pcd/{}.xyzrgb'.format(i), pcd, True)
        prog_bar.update(1)
    return pcd




def preprocess_point_cloud(pcd, voxel_size):
    # TODO: Do voxelization to reduce the number of points for less memory usage and speedup
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down, 
                                                               o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=30))

    # raise NotImplementedError
    return pcd_down, pcd_fpfh


def prepare_dataset(voxel_size, path, i):
    pcd = o3d.io.read_point_cloud(path + '{}.xyzrgb'.format(i))
    pcd_down, pcd_fpfh = preprocess_point_cloud(pcd, voxel_size)
    return pcd, pcd_down, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 2.0
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.95),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(500000, 0.99))
    return result


def local_icp_algorithm(source, target, source_fpfh, target_fpfh, voxel_size, transformation):
    # TODO: Use Open3D ICP function to implement
    distance_threshold = voxel_size * 0.4
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))
    return result



def my_local_icp_algorithm(points, reference_points, voxel_size, mtx_init, max_iterations=10, tolerance=1e-3, point_pairs_threshold=10, verbose=False):
    # TODO: Write your own ICP function
    points = np.array(points.points)
    reference_points = np.array(reference_points.points)
    n = points.shape[0]  # total number of points
    m = points.shape[1]  # matrix size
    src = np.ones((m + 1, n))
    trg = np.ones((m + 1, reference_points.shape[0]))

    src[:m, :] = np.copy(points.T)  # source
    trg[:m, :] = np.copy(reference_points.T)  # target

    transformation_matrix = mtx_init

    # ICP loop
    # 5. Align point cloud at ti+1 to point cloud at ti, and apply the same process to the whole trajectory
    prev_error = float('inf')
    nbrs = NearestNeighbors(n_neighbors=1).fit(trg[:m, :].T)
    for iteration in range(max_iterations):
        src_transformed = transformation_matrix @ src
        
        distances, indices = nbrs.kneighbors(src_transformed[:m, :].T)

        # Filter out invalid pairs (those with large distances)
        valid_pairs = distances < voxel_size
        valid_source_points = src_transformed[:m, valid_pairs.flatten()]
        valid_target_points = trg[:m, indices.flatten()[valid_pairs.flatten()]]

        if valid_source_points.shape[1] < point_pairs_threshold:
            break

        centroid_source = np.mean(valid_source_points, axis=1)
        centroid_target = np.mean(valid_target_points, axis=1)

        source_centered = valid_source_points - centroid_source[:, np.newaxis]
        target_centered = valid_target_points - centroid_target[:, np.newaxis]

        H = source_centered @ target_centered.T

        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = centroid_target - R @ centroid_source

        new_transform = np.identity(4)
        new_transform[:3, :3] = R
        new_transform[:3, 3] = t

        transformation_matrix = new_transform @ transformation_matrix

        mean_error = np.mean(distances)

        # Check for convergence
        if np.abs(prev_error - mean_error) < tolerance:
            break

        prev_error = mean_error

    # Transform points using the final transformation matrix
    aligned_points = (transformation_matrix @ src)[:3, :].T

    return transformation_matrix, aligned_points

    
def add_points(vis, pcd, ceiling_threshold):
        
    tmp = copy.deepcopy(pcd)
    points = np.array(tmp.points)
    colors = np.array(tmp.colors)
    # remove ceiling
    mask = points[:, 1] < ceiling_threshold
    tmp.points = o3d.utility.Vector3dVector(points[mask])
    tmp.colors = o3d.utility.Vector3dVector(colors[mask])
    vis.create_window()
    vis.add_geometry(tmp)
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()


def reconstruct(args):
    # TODO: Return results

    
    os.makedirs(args.data_root + 'pcd/', exist_ok=True)
    # 1. Unproject depth images ti and ti+1 to reconstruct two point clouds.
    depth_image_to_point_cloud(512, args.data_root)
    
    n = len(os.listdir(args.data_root + 'rgb/'))


    # 2. Do voxelization to your point cloud for reducing the number of points for less memory usage and speedup.
    #o3d_img_to_3d(start=0, total_img=n)
    target, target_down, target_fpfh = prepare_dataset(args.voxel_size, args.data_root + 'pcd/', 1)
    
    vis = o3d.visualization.Visualizer()
    add_points(vis, target, 0.00005)
    
    if args.version == 'open3d':
        print('Use Open3D version.')
    elif args.version == 'my_icp':
        print('Use my ICP version.')
    else:
        raise Exception(f"Invalid version. Expected 'open3d' or 'my_icp'.")

    trans = []
    lines = []

    # Dealing with trajectory
    ex_matrix = ex_mat(0, 0, np.pi, 0, 0, 0)[:3, :3]
    track_point = np.zeros((1, 3))
    track_arr = np.zeros((1, 3))
    track_gt = pd.read_csv(args.data_root+'record.txt', header=None, sep=' ')
    #print(track_gt)
    track_gt_arr = np.array(track_gt.drop(track_gt.columns[3:], axis=1))
    #print(track_gt_arr[0])
    #print(track_gt_arr)
    track_gt_arr = np.subtract(track_gt_arr, track_gt_arr[0])
    track_gt_arr[:,2] *= -1.0
    #print(track_gt_arr)
    #track_gt_arr[1:] += 1.0
    track_point_pc = o3d.geometry.PointCloud()
    track_point = np.zeros((1, 3))
    track_point_pc.points = o3d.utility.Vector3dVector(track_point)
    
    prog = tqdm(total=n)
    for i in range(1, n+1):
        source, source_down, source_fpfh = prepare_dataset(args.voxel_size, args.data_root + 'pcd/', i-1)
        if args.wo_RANSAC:
            result_ransac = np.eye(4)
        else:
            result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, args.voxel_size)
            result_ransac = result_ransac.transformation
        
        
        if args.version == 'open3d':
        	# 3. Apply global registration first which is used as the initialization of the local methods.
            result = local_icp_algorithm(
                source_down,
                target_down,
                source_fpfh,
                target_fpfh,
                args.voxel_size,
                result_ransac
            )
            trans.append(result.transformation)
        else:
        	# 4. Apply local registration, using ICP to obtain the transformation matrix.
        	transformation_matrix, aligned_points = my_local_icp_algorithm(
        		source_down, 
        		target_down,  
        		args.voxel_size, 
        		result_ransac,
        		max_iterations=60, 
        		tolerance=1e-7, 
        		point_pairs_threshold=10, 
        		verbose=False
        	)
        	trans.append(transformation_matrix)

        source_temp = copy.deepcopy(source)
        track_pcd = o3d.geometry.PointCloud()
        track_pcd.points = o3d.utility.Vector3dVector(track_point)
        #print('track_gt_arr: ', track_gt_arr.shape)
        track_gt_arr[i-1] = np.dot(ex_matrix, track_gt_arr[i-1].T).T
        tmp = []
        for j in range(0, len(trans)):

            source_temp.transform(trans[-j])
            track_pcd.transform(trans[-j])
        add_points(vis, source_temp, 0.000005)
        track_point_T = np.array(track_pcd.points)
        track_arr = np.concatenate((track_arr, track_point_T))
        lines.append([i - 1, i])
        target, target_down, target_fpfh = copy.deepcopy(source), copy.deepcopy(source_down), copy.deepcopy(source_fpfh)
        prog.update(1)
    
    # Estinated trajectory
    #print('track_arr: ', track_arr)
    track = o3d.geometry.PointCloud()
    track.points = o3d.utility.Vector3dVector(track_arr[1:])
    track.paint_uniform_color([1, 0, 0])# red
    add_points(vis, track, 0.00005)
    track_line = o3d.geometry.LineSet()
    track_line.points = o3d.utility.Vector3dVector(track.points)
    track_line.lines = o3d.utility.Vector2iVector(lines[:n-1])
    track_line.paint_uniform_color([1, 0, 0])
    vis.add_geometry(track_line)
    
    # GT trajectory
    track_gt = o3d.geometry.PointCloud()
    track_gt.points = o3d.utility.Vector3dVector(track_gt_arr)
    track_gt.paint_uniform_color([0, 0, 0])# red
    add_points(vis, track_gt, 0.00005)
    track_gt_line = o3d.geometry.LineSet()
    track_gt_line.points = o3d.utility.Vector3dVector(track_gt.points)
    track_gt_line.lines = o3d.utility.Vector2iVector(lines[:n-1])
    track_gt_line.paint_uniform_color([0, 0, 0])
    vis.add_geometry(track_gt_line)
    error = track_gt_arr[:track_arr.shape[0]]- track_arr[1:]
    print("Trajectory Error: " , np.linalg.norm(error))
    tb = time.time()
    print('Time: ', tb-ta)
    vis.run()
        



if __name__ == '__main__':
    ta = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--floor', type=int, default=1)
    parser.add_argument('-v', '--version', type=str, default='my_icp', help='open3d or my_icp')
    parser.add_argument('--data_root', type=str, default='data_collection/first_floor/')
    parser.add_argument('--voxel_size', type=float, default=0.00001)
    parser.add_argument('-wor', '--wo_RANSAC', action='store_true')
    
    args = parser.parse_args()
    

    if args.floor == 1:
        args.data_root = "data_collection/first_floor/"
    elif args.floor == 2:
        args.data_root = "data_collection/second_floor/"

    # TODO: Output result point cloud and estimated camera pose
    '''
    Hint: Follow the steps on the spec
    '''
    #result_pcd, pred_cam_pos = reconstruct(args)
    reconstruct(args)
    

    # TODO: Calculate and print L2 distance
    '''
    Hint: Mean L2 distance = mean(norm(ground truth - estimated camera trajectory))
    '''
    #print("Mean L2 distance: ")

    # TODO: Visualize result
    '''
    Hint: Should visualize
    1. Reconstructed point cloud
    2. Red line: estimated camera pose
    3. Black line: ground truth camera pose
    '''
    #o3d.visualization.draw_geometries()

