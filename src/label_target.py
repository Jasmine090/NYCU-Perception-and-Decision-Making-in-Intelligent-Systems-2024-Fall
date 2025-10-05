from scipy.spatial import KDTree
import numpy as np
import random
import pandas as pd
import cv2

def click_event(event, x, y, flags, params):
    global target_p, img
    if event == cv2.EVENT_LBUTTONDOWN:
        target_p = [x, y]
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('image', img)

    if event == cv2.EVENT_RBUTTONDOWN:
        target_p = [x, y]
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('image', img)


if __name__ == '__main__':
    excel_file = 'color_coding_semantic_segmentation_classes.xlsx'
    data = pd.read_excel(excel_file)

    if 'Coordinates' not in data.columns:
        data['Coordinates'] = None  

    img = cv2.imread('map.png')
    original_img = img.copy()

    items = data['Name'].tolist()
    colors = data['Color_Code (R,G,B)'].tolist()
    _dict = dict(zip(items, colors))

    target = input('Input Target: ')
    while _dict.get(target) is None:
        target = input('Target not found, please try again: ')
    
    target_color = _dict[target]
    target_color = target_color.strip('()')
    r, g, b = map(int, target_color.split(','))
    target_color = np.array([b, g, r])  # RGB -> BGR

    while target_color not in img:
        target = input('Target not found, please try again: ')

    target_p = []


    cv2.namedWindow('Image')
    cv2.imshow('Image', img)
    cv2.setMouseCallback('Image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



    target_index = data[data['Name'] == target].index[0]

    
    
    data.at[target_index, 'Coordinates'] = str(target_p[1])+ ' ' + str(target_p[0])

    cv2.namedWindow('Image')
    cv2.imshow('Image', img)
    cv2.setMouseCallback('Image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    
    target_index = data[data['Name'] == target].index[0]

    
    data.at[target_index, 'Coordinates1'] = str(target_p[1])+ ' ' + str(target_p[0])

    
    with pd.ExcelWriter(excel_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        data.to_excel(writer, index=False)

