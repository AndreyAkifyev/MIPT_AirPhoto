import torch
import json
from PIL import Image
import numpy as np
from torchvision import transforms

import os
import os.path as osp
from utils import rotate_point

class Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg):
        super().__init__()
        self.img_root = cfg.train_img
        self.images = sorted(os.listdir(cfg.train_img))
        self.norm_value = 10496 

        self.label_root = cfg.train_labels
        self.labels = sorted(os.listdir(cfg.train_labels))

        assert len(self.images) == len(self.labels)
        self.transforms = cfg.val_augs

    def __getitem__(self, index: int):
        image = Image.open(osp.join(self.img_root, self.images[index])).convert("RGB")
        labels = json.load(open(osp.join(self.label_root, self.labels[index])))
        left_top = labels['left_top']
        right_bottom = labels['right_bottom']
        center_point = [int((left_top[0] + right_bottom[0]) / 2), int((left_top[1] + right_bottom[1]) / 2)]

        angel = labels['angle']
        # rot_angle = np.deg2rad(-angel)
        # left_top = rotate_point(center_point, left_top, rot_angle)
        # right_bottom = rotate_point(center_point, right_bottom, rot_angle)
        # print(left_top, right_bottom)
                
        # print(labels, self.images[index], self.labels[index])
        return {
            'image' : self.transforms(image),
            'target' : torch.Tensor([
                    center_point[0] / self.norm_value,
                    center_point[1] / self.norm_value, 
                    angel / 360
                ])
            
        }

    def __len__(self) -> int:
        return len(self.images)
