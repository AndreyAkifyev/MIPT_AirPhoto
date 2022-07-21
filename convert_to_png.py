import argparse
from PIL import Image
import os.path as osp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--tiff', type=str, default='./data/original.tiff')
    parser.add_argument('--save-dir', type=str, default='./data/')

    args = parser.parse_args()

    tiff = Image.open(args.tiff)
    tiff.save(osp.join(args.save_dir, 'original_new.png'), "PNG")

