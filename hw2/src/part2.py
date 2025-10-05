from scipy.spatial import KDTree
import numpy as np
import random
import pandas as pd
import cv2


class Node:
    def __init__(self, x, y, parent = None):
        self.x = x
        self.y = y
        self.parent = parent

class RRT:
    def __init__(self, start_p, target, map, step=4, max_iter=80):
        self.start_p = start_p
        self.start_Node = Node(x=start_p[0], y=start_p[1], parent=None)
        self.target = target
        self.map = map
        self.step = step
        self.max_iter = max_iter

        self.tree = [self.start_Node]

    def get_randpoint(self):
        rand_x = round(random.uniform(0, self.map.shape[0]), 2)
        rand_y = round(random.uniform(0, self.map.shape[1]), 2)
        return [rand_x, rand_y]
    
    def find_nearest(self, p_rand):
        closest_node = None
        closest_dist = np.inf
        for node in self.tree:
            dist = np.linalg.norm(np.array([node.x, node.y]) - np.array(p_rand))
            if dist < closest_dist:
                closest_dist = dist
                closest_node = node
        return closest_node, [closest_node.x, closest_node.y]
    
    def steer(self, p_near, p_rand):
        diff = np.array(p_rand) - np.array(p_near)
        dist = np.linalg.norm(diff)
        
        if dist <= self.step:
            return np.array(p_rand).astype(int)
        extend = (diff / dist) * self.step
        return (np.array(p_near) + extend).astype(int)

    def collision_check(self, p_near, p_new, sample_n=20):
        near = np.array(p_near, dtype=int)
        new = np.array(p_new, dtype=int)
        
        points = np.linspace(near, new, sample_n, endpoint=True, dtype=int)
        return np.any(self.map[points[:, 0], points[:, 1]] != [255, 255, 255])

    def update_all(self, p_new, p_nearNode):
        newNode = Node(x=p_new[0], y=p_new[1], parent=p_nearNode)
        self.tree.append(newNode)
        return newNode

    def target_check(self, p_new, target_coor, window_size=50):
        distance = np.linalg.norm(np.array(p_new) - np.array(target_coor))
        return distance < window_size

    def search(self, target_coor):
        print('---Start RRT Algorithm---')
        print('Start point: ', self.start_p)
        print('Target color: ', self.target)

        consecutive_no_progress = 0
        for i in range(self.max_iter):
            p_rand = self.get_randpoint()
            p_nearNode, p_near = self.find_nearest(p_rand)
            p_new = self.steer(p_near, p_rand)

            if self.collision_check(p_near, p_new):
                consecutive_no_progress += 1
                if consecutive_no_progress > 10:
                    print("No progress made, terminating early.")
                    return None, self.tree
                continue

            consecutive_no_progress = 0
            tmpNode = self.update_all(p_new, p_nearNode)

            if self.target_check(p_new, target_coor):
                print('Target is found.')
                self.update_all(target_coor, tmpNode)
                path = []
                tmp = self.tree[-1]
                while tmp is not None:
                    path.append([tmp.x, tmp.y])
                    tmp = tmp.parent
                return path, self.tree

        print('No path was found, repeat RRT again.')
        return None, self.tree

def click_event(event, x, y, flags, params):
    global start_p, img
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        start_p = [x, y]
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('img', img)

def plot(path, tree, image, avg_target_point, target_object):
    # Define colors and thicknesses
    line_color = (183, 183, 183)  # Gray for the tree
    path_color = (0, 0, 255)      # Red for the path
    node_color = (100, 0, 0)      # Blue for nodes
    start_color = (0, 255, 0)     # Green for start point
    target_color = (0, 0, 255)    # Red for target point
    line_thickness_tree = 2
    line_thickness_path = 5
    circle_thickness_filled = -1

    # Plot the tree
    for node in tree:
        if node.parent is not None:
            cv2.line(image, (node.y, node.x), (node.parent.y, node.parent.x), line_color, line_thickness_tree)
        cv2.circle(image, (node.y, node.x), 6, node_color, thickness=circle_thickness_filled)

    # Plot the path
    for (current_node, next_node) in zip(path[:-1], path[1:]):
        cv2.line(image, (current_node[1], current_node[0]), (next_node[1], next_node[0]), path_color, line_thickness_path)
        cv2.circle(image, (current_node[1], current_node[0]), 6, path_color, thickness=circle_thickness_filled)

    cv2.circle(image, (path[-1][1], path[-1][0]), 6, path_color, thickness=circle_thickness_filled)
    cv2.circle(image, (avg_target_point[1], avg_target_point[0]), 12, target_color, thickness=circle_thickness_filled)
    cv2.circle(image, (path[-1][1], path[-1][0]), 6, node_color, thickness=circle_thickness_filled)
    cv2.circle(image, (path[0][1], path[0][0]), 6, start_color, thickness=circle_thickness_filled)

    # Save the image
    cv2.imwrite(f'{target_object}.png', image)


def get_target_data(target, _dict, _dict2, _dict3):
    target_color = _dict[target]
    target_coor = _dict2[target]
    target_coor1 = _dict3[target]

    target_color = target_color.strip('()')
    r, g, b = map(int, target_color.split(','))
    target_color = np.array([b, g, r])  # RGB->BGR
    return target_color, target_coor, target_coor1


def handle_target_coordinates(img, target_color, target_coor):
    if pd.isna(target_coor) == False:
        x, y = map(int, target_coor.split(' '))
        return np.array([x, y])
    else:
        indices = np.where(np.all(img == target_color, axis=-1))
        pixel_coor = np.column_stack((indices[0], indices[1]))
        return np.mean(pixel_coor, axis=0).astype(int)


def find_path_to_target(target_coor, target_color, original_img):
    start_points = [start_p[1], start_p[0]]
    path = None
    while path is None:
        _RRT = RRT(start_points, target_color, original_img, step=100)
        path, tree = _RRT.search(target_coor)
    return path, tree


def check_secondary_target_coor(target_coor1):
    if pd.isna(target_coor1) == False:
        x, y = map(int, target_coor1.split(' '))
        return np.array([x, y])
    return None


def find_path():
    global start_p, img
    data = pd.read_excel('color_coding_semantic_segmentation_classes.xlsx')
    items = data['Name'].tolist()
    colors = data['Color_Code (R,G,B)'].tolist()
    coors = data['Coordinates'].tolist()
    coors1 = data['Coordinates1'].tolist()

    _dict = dict(zip(items, colors))
    _dict2 = dict(zip(items, coors))
    _dict3 = dict(zip(items, coors1))
    
    img = cv2.imread('map.png')
    original_img = img.copy()
    
    target = input('Input Target: ')
    while _dict.get(target) is None:
        target = input('Target not found, please try again: ')

    target_color, target_coor, target_coor1 = get_target_data(target, _dict, _dict2, _dict3)
    
    
    start_p = []
    img = cv2.imread('map.png')
    original_img = img.copy()
    cv2.namedWindow('Image')
    cv2.imshow('Image', img)
    cv2.setMouseCallback('Image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    target_coor = handle_target_coordinates(img, target_color, target_coor)
    path, tree = find_path_to_target(target_coor, target_color, original_img)
    target_coor1 = check_secondary_target_coor(target_coor1)

    if path is not None and tree is not None:
        plot(path, tree, original_img, target_coor, target)
        return np.array(path), target_coor1, target, target_color
    
    return None, None, target, target_color


if __name__ == '__main__':
    find_path()

