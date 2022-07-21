import plotly.express as px
import argparse

import os
import os.path as osp

import json
from math import sqrt
import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-dir', type=str, default='figures')
    parser.add_argument('--annotation-dir', type=str, default='data/train_dataset_train/train/json')

    args = parser.parse_args()

    save_dir = args.save_dir
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)

    json_dir = args.annotation_dir

    x_coords = []
    y_coords = []
    for file in tqdm.tqdm(os.listdir(json_dir)):
        data = json.load(open(osp.join(json_dir, file)))
        x_1, y_1 = data['left_top']
        x_2, y_2 = data['right_bottom']
        center_x = (x_1 + x_2) / 2
        center_y = (y_1 + y_2) / 2
        x_coords.append(center_x)
        y_coords.append(center_y)
    fig = px.scatter(x=x_coords, y=y_coords)
    fig.write_image(osp.join(save_dir, 'center_scatter.png'))