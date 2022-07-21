from types import SimpleNamespace
import os.path as osp
from torchvision import transforms

cfg = SimpleNamespace(**{})

cfg.data_dir = '/path/to/data'
cfg.tiff_path = osp.join(cfg.data_dir, 'original.png')
cfg.train_img = osp.join(cfg.data_dir, 'train_dataset_train', 'train', 'img')
cfg.train_labels = osp.join(cfg.data_dir, 'train_dataset_train', 'train', 'json')
cfg.test_img = osp.join(cfg.data_dir, 'test_dataset_test')
# cfg.test_img = cfg.train_img

cfg.mean = [0.485, 0.456, 0.406]
cfg.std = [0.229, 0.224, 0.225]

cfg.crop_size = 1024
cfg.train_augs = transforms.Compose([
    transforms.Resize((cfg.crop_size, cfg.crop_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=cfg.mean, std=cfg.std),
])

cfg.val_augs = transforms.Compose([
    transforms.Resize((cfg.crop_size, cfg.crop_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=cfg.mean, std=cfg.std),
])

cfg.train_ds = 'train_dataset'
cfg.train_bs = 4
cfg.train_num_workers = 6

cfg.val_ds = 'val_dataset'
cfg.val_bs = 32
cfg.val_num_workers = 6

cfg.test_ds = 'test_dataset'
cfg.test_bs = 32
cfg.test_num_workers = 6

cfg.backbone_name = 'regnety_008'
cfg.pretrained = True
cfg.model = 'model_1'
cfg.in_channels = 3
cfg.num_classes = 3

cfg.opt = 'Adam'
cfg.lr = 3e-4
cfg.wd = 0

cfg.T_0 = 20
cfg.min_lr = 1e-5

cfg.criterion = 'MAE'
 
cfg.epochs = 10

basic_cfg = cfg
