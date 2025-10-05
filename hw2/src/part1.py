import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
import cv2

def plot_points(point, pcd, pcd_color):
    combined_coordinates = np.vstack((pcd, point))
    combined_colors = np.vstack((pcd_color, np.array([0, 0, 0], dtype=np.float32)))
    plt.figure()
    plt.scatter(combined_coordinates[:, 2], combined_coordinates[:, 0], c=combined_colors, s=0.25)
    plt.axis('off')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('map.png', dpi=300, bbox_inches='tight')
    
    image = cv2.imread('map.png')
    black_pixels = np.all(image == [0, 0, 0], axis=-1)
    pixel_indices = np.where(black_pixels)
    pixel_indices = np.transpose(pixel_indices)
    average_position = np.mean(pixel_indices, axis=0)

    return average_position

def get_trans_matrix():
    point_cloud = np.load('./semantic_3d_pointcloud/point.npy')
    colors = np.load('./semantic_3d_pointcloud/color01.npy')

    mask_roof = point_cloud[:, 1] < -0.03
    mask_floor = point_cloud[:, 1] > -0.001
    mask_others = ~(mask_roof | mask_floor)

    point_cloud = point_cloud[mask_others]
    colors = colors[mask_others]

    reference_points = np.float32([[0.15, 0, 0], [0.05, 0, 0], [-0.05, 0, 0.1], [0, 0, -0.1]])
    transformed_points = [plot_points(pt, point_cloud, colors) for pt in reference_points]

    src_points = np.float32([[pt[0], pt[1]] for pt in transformed_points])
    dst_points = np.float32([[pt[0], pt[2]] for pt in reference_points])

    transformation_matrix, _ = cv2.findHomography(src_points, dst_points)


    scaled_transformation_matrix = transformation_matrix * 10000 / 255

    return scaled_transformation_matrix

if __name__ == '__main__':
    transformation_matrix = get_trans_matrix()
    print(transformation_matrix)
