from default_config import basic_cfg
from torchvision import transforms

cfg = basic_cfg

cfg.epochs = 500

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

cfg.lr = 5e-5