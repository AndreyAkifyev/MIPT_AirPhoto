import torch
import torch.nn as nn
import numpy as np
import timm
from torch.cuda.amp import autocast, GradScaler
import importlib
import argparse
import sys
import tqdm
from pprint import pprint
import os
import os.path as osp
import shutil

sys.path.append('configs')
sys.path.append('models')
sys.path.append('datasets')

def train_one_epoch(train_dataloader: torch.utils.data.Dataset, model, opt, scheduler, criterion, scaler: GradScaler):
    pbar = tqdm.tqdm(train_dataloader)
    for data in pbar:
        opt.zero_grad()
        img = data['image'].cuda()
        target = data['target'].cuda()
        with autocast():
            preds = model(img)
            loss = criterion(preds, target)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        scheduler.step()
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        pbar.set_postfix(Train_loss = loss.item(),
                                LR = opt.param_groups[0]['lr'],
                                GPU_mem = f"{mem:.02f} GB" )
        # pbar.set_description(f"Loss : {loss:.4f}")
        # # break

@torch.no_grad()
def validate(val_dl, model, loss_fn):
    pbar = tqdm.tqdm(val_dl)
    loss = 0 
    count = 0

    val_preds = []
    targets = []
    for idx, data in enumerate(pbar):
        img = data['image'].cuda()
        target = data['target'].cuda()
        
        with autocast():
            preds = torch.sigmoid(model(img)).detach().cpu().numpy()
            val_preds.append(preds)

            target = target.detach().cpu().numpy()
            targets.append(target)
        # if idx > 2:
        #     break
    val_preds = np.concatenate(val_preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    # print(val_preds.shape, targets.shape)

    loss = loss_fn(targets, val_preds)
    return 1 - loss

def get_train_dataloader(dataset, cfg):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.train_bs,
        shuffle=True,
        num_workers=cfg.train_num_workers,
    )

def get_val_dataloader(dataset, cfg):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.val_bs,
        shuffle=False,
        num_workers=cfg.val_num_workers,
    )

class CompLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.coord_loss = nn.L1Loss(reduction='none')
        self.angle_loss = nn.L1Loss(reduction='none')
    
    def forward(self, target, preds):
        loss_1 = 0.7 * 1/2 * self.coord_loss(target[:, :2], preds[:, :2]).mean()

        angle_loss = torch.abs(self.angle_loss(target[:, 2], preds[:, 2]))
        loss_2 = 0.3 * torch.minimum(angle_loss, torch.abs(1 - angle_loss)).mean()
        return loss_1 + loss_2


def val_loss(target, preds):
    # print(target.shape, preds.shape)
    ## they were re-scaled to (0, 1), so rescale them back
    target_angles = 360 * target[:, 2]
    preds_angles = 360 * preds[:, 2]

    target_coords = target[:, :2]
    preds_coords = preds[:, :2]

    loss = 0
    
    angle_error = np.abs(target_angles - preds_angles)
    normalized_angel_error = (np.minimum(angle_error, np.abs(360 - angle_error)) / 360).mean()
    loss += 0.3 * normalized_angel_error
    
    # print(target_coords, preds_coords)
    normalized_center_error = 1/2 * np.abs(target_coords - preds_coords).mean()
    loss += 0.7 * normalized_center_error
    print(f"normalized angel error : {normalized_angel_error:.4f}, normalized center error : {normalized_center_error:.4f}")
    return loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    config_name = args.config.replace('.py', '')
    print(args)
    
    cfg = importlib.import_module(config_name).cfg
    pprint(cfg)

    train_ds = importlib.import_module(cfg.train_ds).Dataset(cfg)
    train_dl = get_train_dataloader(train_ds, cfg)

    val_ds = importlib.import_module(cfg.val_ds).Dataset(cfg)
    val_dl = get_val_dataloader(val_ds, cfg)

    model = importlib.import_module(cfg.model).Net(cfg).cuda()
    scaler = GradScaler()
    if cfg.opt == 'Adam':
        opt = torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.wd
        )
    else:
        raise NotImplementedError("Choose supported optimizer")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt,
        T_0=cfg.T_0,
        eta_min=cfg.min_lr
    )
    if cfg.criterion == 'MAE':
        criterion = nn.L1Loss()
    elif cfg.criterion == 'MSE':
        criterion = nn.MSELoss()
    elif cfg.criterion == 'CompLoss':
        criterion = CompLoss(cfg)
    elif cfg.criterion == 'BCELogits':
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError("Choose supported loss")
    exp_num = 1
    save_dir = osp.join('exps', f"exp_{exp_num}")
    while osp.isdir(save_dir):
        exp_num += 1
        save_dir = osp.join('exps', f"exp_{exp_num}")
    
    os.makedirs(save_dir)
    shutil.copyfile(f'configs/{config_name}.py', osp.join(save_dir, f"{config_name}.py"))
    shutil.copyfile(f'configs/default_config.py', osp.join(save_dir, "default_config.py"))
    best_metric = 0
    last_model_name = None
    for epoch in range(1, cfg.epochs + 1):
        train_one_epoch(train_dl, model, opt, scheduler, criterion, scaler)
        valid_metric = validate(val_dl, model, val_loss)
        print(f"Epoch : {epoch}, validation metric : {valid_metric:.4f}")
        if valid_metric > best_metric:
            best_metric = valid_metric
            model_name = f"{cfg.backbone_name}_{epoch}_{valid_metric:.4f}.pth"
            if last_model_name is not None:
                os.remove(osp.join(save_dir, last_model_name))
            last_model_name = model_name
            torch.save(model.state_dict(), osp.join(save_dir, model_name))