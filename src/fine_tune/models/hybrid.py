import torch
import torch.nn as nn

from .resnet import ResNetBackbone

from src.config.paths import WEIGHTS_DIR

class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = ResNetBackbone()

        # self.adapter = nn.Sequential() (if dims needs to be matched across YOLO and ResNet)
        self.head = nn.Sequential() # Replace with YOLO neck and head

        self._intitialise_weights()

    def forward(self, x):
        features = self.backbone(x)

        # features = self.adapater(features)
        return self.head(features)

    def _intitialise_weights(self):
        self.backbone.load_state_dict(
            torch.load(
                WEIGHTS_DIR / "resnet50.pt", 
                map_location="cpu", 
                weights_only=True
                )
            )

        # I think the other one or two bits should be random intialisation which 
        # i think yolo already provides, not sure though
        