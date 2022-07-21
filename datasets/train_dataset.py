from typing import Any, Dict
import torch
from torchvision import transforms

import numpy as np 
from PIL import Image
from utils.crop_rotated_rect import inside_rect, crop_rotated_rectangle
## https://www.freeconvert.com/tiff-to-png/download
## convert .tiff to .png

class Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg):
        super().__init__()
        self.image = np.array(Image.open(cfg.tiff_path).convert('RGB'))
        self.height, self.width = self.image.shape[:2]

        self.h_mult = self.height
        self.w_mult = self.width
        self.crop_size = 1024
        self.transforms = cfg.train_augs

    def __getitem__(self, dummy_index: int):
        center_x, center_y = (-1, -1)

        angle_in_degrees = np.random.randint(low=0, high=360)
        while not self._valid_rect(center_x, center_y, angle_in_degrees):
            center_x, center_y = np.random.uniform(size=2)
        
        return self._get_image(center_x, center_y, angle_in_degrees)
        
    def _get_image(self, x: float, y: float, angle_in_degrees: int): 
        rect = self._create_rect(x, y, angle_in_degrees)
        rotated_image = crop_rotated_rectangle(self.image, rect)
        return {
            'image' : self.transforms(Image.fromarray(rotated_image)),
            'target' : torch.Tensor([
                x, 
                y,
                angle_in_degrees / 360
            ])
        }

    def _valid_rect(self, x: float, y: float, angle: int) -> bool:
        rect = self._create_rect(x, y, angle)
        return inside_rect(rect, self.width, self.height)

    def _create_rect(self, x: float, y:float, angle: int):
        center = (int(x * self.w_mult), int(y * self.h_mult))
        rect = (center, (self.crop_size, self.crop_size), angle)
        return rect
         
    def __len__(self) -> int:
        return 1000

