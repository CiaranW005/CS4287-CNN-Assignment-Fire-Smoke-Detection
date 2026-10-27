from types import SimpleNamespace
from ultralytics.utils.loss import v8DetectionLoss

from .base import BaseLoss

class YOLOLoss(BaseLoss):
    def __init__(self, model) -> None:
        shim = SimpleNamespace(
            args=SimpleNamespace(box=7.5, cls=0.5, dfl=1.5), #ulatrlytics default gains
            model=[model.detect],
            parameters=model.parameters,
        )
        self.criterion = v8DetectionLoss(shim)

    def loss(self, predicts, target):
        return self(predicts, target)[0]

    def __call__(self, predicts, target):
        # one pass gives both the total and the separate box/cls/dfl terms
        total, items = self.criterion(predicts, target)
        return total.sum(), {name: value.item() for name, value in items.items()}