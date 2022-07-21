import torch
from PIL import Image

import os
import os.path as osp

class Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg):
        super().__init__()
        self.img_root = cfg.test_img
        self.images = sorted(os.listdir(self.img_root))
        self.transforms = cfg.val_augs

    def __getitem__(self, index: int):
        image = Image.open(osp.join(self.img_root, self.images[index])).convert("RGB")
        return {
            'image' : self.transforms(image),
            'identifier' : self.images[index].split('.')[0]       
        }

    def __len__(self) -> int:
        return len(self.images)
