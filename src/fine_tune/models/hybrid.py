import torch.nn as nn

from .resnet import ResNetBackbone

class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = ResNetBackbone()

        # self.adapter = nn.Sequential() (if dims needs to be matched across YOLO and ResNet)
        self.head = nn.Sequential() # Replace with YOLO neck and head

    def forward(self, x):
        features = self.backbone(x)

        # features = self.adapater(features)
        return self.head(features)