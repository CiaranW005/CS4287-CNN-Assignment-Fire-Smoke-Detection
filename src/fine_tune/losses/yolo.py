import torch

from .base import BaseLoss

class YOLOLoss(BaseLoss):
    def __init__(self) -> None:
        pass

    def loss(self, predicts, target):
        return sum(p.sum() for p in predicts) * 0.0 # dummy loss function