import timm
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = timm.create_model(
            model_name=cfg.backbone_name,
            pretrained=cfg.pretrained,
            num_classes=0,
        )
        self.head = nn.Linear(self.backbone.num_features, cfg.num_classes)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        x = torch.sigmoid(x)
        return x