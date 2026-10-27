import torch
import torch.nn as nn
from ultralytics.nn.modules import C2f, Conv, Detect

from .resnet import ResNetBackbone
from src.config.paths import WEIGHTS_DIR

class HybridModel(nn.Module):
    def __init__(self, num_classes=2, widths=(128, 256, 512)):
        super().__init__()
        c3, c4, c5 = widths

        self.backbone = ResNetBackbone()

        # 1x1 conv: ResNet channels 512, 1024, 2048 -> YOLO widths
        self.adapter = nn.ModuleList(Conv(c_in, c_out) for c_in, c_out in zip((512, 1024, 2048), widths))

        #YOLOv8 neck (FPN top-down + PAN bottom-up)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.top_down4 = C2f(c5 + c4, c4)
        self.top_down3 = C2f(c4 + c3, c3)
        self.downsample3 = Conv(c3, c3, 3, 2)
        self.bottom_up4 = C2f(c3 + c4, c4)
        self.downsample4 = Conv(c4, c4, 3, 2)
        self.bottom_up5 = C2f(c4 + c5, c5)

        self.detect = Detect(nc=num_classes, ch=widths)
        self.detect.stride = torch.tensor([8.0, 16.0, 32.0])
        self.detect.bias_init()

        self._intitialise_weights()

    def forward(self, x):
        p3, p4, p5 = (adapt(f) for adapt, f in zip(self.adapter, self.backbone(x)))

        t4 = self.top_down4(torch.cat([self.upsample(p5), p4], 1))
        n3 = self.top_down3(torch.cat([self.upsample(t4), p3], 1))
        n4 = self.bottom_up4(torch.cat([self.downsample3(n3), t4], 1))
        n5 = self.bottom_up5(torch.cat([self.downsample4(n4), p5], 1))

        return self.detect([n3, n4, n5])

    def _intitialise_weights(self):
        self.backbone.load_state_dict(
            torch.load(
                WEIGHTS_DIR / "resnet50.pt", 
                map_location="cpu", 
                weights_only=True
                )
            )
        