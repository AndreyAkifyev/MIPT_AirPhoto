import argparse
from PIL import Image
import numpy as np

import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--path-to-train-canvas', type=str, default="./data/original.png")
    parser.add_argument('--path-to-test-canvas', type=str, default='./data/test_image.png')
    parser.add_argument('--save-dir', type=str, default='./data/')

    args = parser.parse_args()

    train_canvas = np.array(Image.open(args.path_to_train_canvas).convert("RGB"))
    test_canvas = np.array(Image.open(args.path_to_test_canvas))

    print(train_canvas.shape, test_canvas.shape)
    test_canvas = np.where(test_canvas != 0, test_canvas, train_canvas)
    Image.fromarray(test_canvas).save(os.path.join(args.save_dir, 'merged_canvas.png'))

