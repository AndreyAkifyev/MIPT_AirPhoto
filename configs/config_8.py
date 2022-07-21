from default_config import basic_cfg
from torchvision import transforms

cfg = basic_cfg

cfg.train_ds = 'train_dataset_2'
cfg.epochs = 20_000

cfg.model = 'model_2'
cfg.backbone_name = 'swin_base_patch4_window7_224'
cfg.crop_size = 224

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

cfg.train_bs = 8
cfg.val_bs = 8
cfg.criterion = 'CompLoss'

cfg.lr = 5e-5