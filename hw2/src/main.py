from part1 import get_trans_matrix
from part2 import find_path
from part3 import navigation
from PIL import Image
import os
import natsort
import numpy as np
import pandas as pd

def find_obj_id(obj_name):
    data = pd.read_excel('color_coding_semantic_segmentation_classes.xlsx')
    row = data[data['Name']==obj_name]
    return row.index[0]+1

def save_to_gif(object=None):
    root_dir = f"{object}_path"
    img_list = [Image.open(os.path.join(root_dir, i)) for i in natsort.natsorted(os.listdir(root_dir)) 
                if i.endswith(".jpg") and i.startswith("RGB")]

    if not img_list:
        print("No images found.")
        return

        # Save the images as a GIF
    img_list[0].save(os.path.join(root_dir, f"{object}.gif"), save_all=True, 
                         append_images=img_list[1:], duration=100, loop=0)

    print(f"GIF saved as 'video.gif' in {root_dir}")


if __name__ == '__main__':
    # Part 1: 2D semantic map construction
    trans_m = get_trans_matrix()
    # Part 2: RRT Algorithm
    # get the path bt rrt
    path, obj_points, obj, _ = find_path()
    # obtain pbject id
    target_id = find_obj_id(obj)
    # Transform path and points to 3d
    path_3d = np.c_[path, np.ones(path.shape[0])] @ trans_m.T
    obj_points_3d = np.append(obj_points, 1) @ trans_m.T
    # Part 3: Robot navigation
    target_semantic_id = find_obj_id(obj)
    navigation(path_3d, obj_points_3d, obj, target_semantic_id)
    # Save as gif
    save_to_gif(obj)