import plotly.express as px
import argparse

import os
import os.path as osp
import json

import numpy as np
from PIL import Image

import tqdm
from utils import rotate_point, rotate_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-to-train', type=str, default="./data/train_dataset_train/train")
    parser.add_argument('--save-dir', type=str, default="./data/")

    args = parser.parse_args()
    
    train_path = args.path_to_train
    save_dir = args.save_dir

    images = sorted(os.listdir(osp.join(train_path, "img")), key=lambda x: int(x.split('.')[0]))
    labels = sorted(os.listdir(osp.join(train_path, "json")), key=lambda x: int(x.split('.')[0]))

    test_image = np.zeros((10496, 10496, 3))
    crop_size = 1024
    assert crop_size % 2 == 0
    
    for idx, (img_name, label_name) in enumerate(tqdm.tqdm(zip(images, labels), total=len(images))):
        img = np.array(Image.open(osp.join(train_path, 'img', img_name)))
        label = json.load(open(osp.join(train_path, 'json', label_name)))
        
        left_top = label["left_top"]
        right_bottom = label["right_bottom"]
        angle = label["angle"]
        center_point = [int((left_top[0] + right_bottom[0]) / 2), int((left_top[1] + right_bottom[1]) / 2)]

        rot_angle = np.deg2rad(-angle)
        left_top = rotate_point(center_point, left_top, rot_angle)
        right_bottom = rotate_point(center_point, right_bottom, rot_angle)
        # print(left_top, right_bottom)

        rotated_image = rotate_image(img, -angle)
        center_x = center_point[0]
        center_y = center_point[1]

        ## Simple paste
        # test_image[center_y - crop_size // 2: center_y + crop_size // 2, center_x - crop_size // 2 : center_x + crop_size // 2] = rotated_image 
        
        ## Conditional paste
        test_image[center_y - crop_size // 2: center_y + crop_size // 2, center_x - crop_size // 2 : center_x + crop_size // 2] = \
            np.where(rotated_image != 0, rotated_image, test_image[center_y - crop_size // 2: center_y + crop_size // 2, center_x - crop_size // 2 : center_x + crop_size // 2])
        # if idx > 10:
        #     break
        # break
    fig = px.imshow(test_image)
    fig.show()

    if not osp.isdir(save_dir):
        os.makedirs(save_dir)
    Image.fromarray(test_image.astype(np.uint8)).save(osp.join(save_dir, 'test_image.png'))
        