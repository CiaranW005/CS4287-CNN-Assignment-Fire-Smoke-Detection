import torch

from abc import ABC, abstractmethod

class BaseLoss(ABC):

    @abstractmethod
    def loss(self, predicts, target):
        pass

    def metrics(self, predicts, target):
        return {}

    def __call__(self, predicts, target):
        loss = self.loss(predicts, target)

        with torch.no_grad():
            metrics = self.metrics(predicts, target)

        return loss, metrics