import argparse
import sys

import os
import os.path as osp

import importlib
import torch

import tqdm
import json
import shutil

def get_test_dl(test_ds, cfg):
    return torch.utils.data.DataLoader(
        test_ds,
        batch_size=cfg.test_bs,
        shuffle=False,
        num_workers=cfg.test_num_workers,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-root', type=str, default='./exps/')
    parser.add_argument('--exp', type=str, required=True)

    args = parser.parse_args()
    exp_dir = osp.join(args.exp_root, args.exp)
    save_dir = osp.join(exp_dir, 'test_preds')

    sys.path.append(exp_dir)
    sys.path.append('datasets')
    sys.path.append('models')

    config_name = list(filter(lambda x : x.startswith('config'), os.listdir(exp_dir)))[0].replace('.py', '')
    cfg = importlib.import_module(config_name).cfg
    
    test_ds = importlib.import_module(cfg.test_ds).Dataset(cfg)
    test_dl = get_test_dl(test_ds, cfg)
    
    model = importlib.import_module(cfg.model).Net(cfg).cuda()
    model.eval()
    
    weights = list(filter(lambda x : x.endswith('.pth'), os.listdir(exp_dir)))[0]
    status = model.load_state_dict(torch.load(osp.join(exp_dir, weights)))
    if status:
        print("Loaded weights successfully")
    else:
        print("Wrong weights") 
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():
        for data in tqdm.tqdm(test_dl):
            images = data['image'].cuda()
            identifiers = data['identifier']
            predictions = torch.sigmoid(model(images))
            for (x_coord, y_coord, angle), identifier in zip(predictions, identifiers):
                center_point = [int(10496 * x_coord), int(10496 * y_coord)]
                prediction = {
                    'left_top' : center_point,
                    'right_top' : center_point,
                    'left_bottom' : center_point,
                    'right_bottom' : center_point,
                    'angle' : int(360 * angle) 
                }
                with open(osp.join(save_dir, f'{identifier}.json'), 'w') as output_file:
                    json.dump(prediction, output_file, indent=4)
    shutil.make_archive('submission', 'zip', osp.join(exp_dir, 'test_preds'))
    shutil.copyfile('submission.zip', osp.join(exp_dir, 'submission.zip'))
    os.remove('./submission.zip')