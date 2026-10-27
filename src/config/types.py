import torch

import numpy as np

from dataclasses import dataclass, field
from typing import TypedDict, Callable

class ModelParameters(TypedDict):
    seed: int 
    num_workers: int
    pin_memory: bool

    batch_size: int
    epochs: int
    accumulation_steps: int

    learning_rate: float
    weight_decay: float

    model_name: str

class ModelInfo(TypedDict):
    timestamp: str
    notes: list[str]
    git_commit: str
    parameters: ModelParameters

@dataclass
class TrainingResults:
    history: ModelHistory
    best_state: dict[str, torch.Tensor] | None
    best_eval_loss: float
    best_epoch: int | None
    error: Exception | None = None

@dataclass
class TrainingObjects:
    model: torch.nn.Module
    criterion: Callable
    optimizer: torch.optim.Optimizer
    device: torch.device

    accumulation_steps: int
    scaler: torch.amp.GradScaler

@dataclass
class EpochHistory:
    losses: list[float] = field(default_factory=list)
    metrics: list[float]  = field(default_factory=list)

    # set by Trainer.evaluate, computed over the whole split
    map50: float | None = None
    map50_95: float | None = None
    map_per_class: list[float] | None = None # [smoke, fire] 

    def append(self, loss, metrics):
        self.losses.append(loss)
        self.metrics.append(metrics)

    @property
    def avg_loss(self):
        return float(np.mean(self.losses)) if self.losses else 0.0

@dataclass
class ModelHistory:
    train: list[EpochHistory] = field(default_factory=list)
    eval: list[EpochHistory] = field(default_factory=list)

    def append(self, train_history, eval_history):
        self.train.append(train_history)
        self.eval.append(eval_history)

    @property
    def train_losses(self):
        return [batch_loss.avg_loss for batch_loss in self.train]
    
    @property
    def eval_losses(self):
        return [batch_loss.avg_loss for batch_loss in self.eval]
    
@dataclass
class BoundingBox:
    category: int   #Fire-1, Smoke-0
    coordinates: tuple[float, float, float, float] #x_center, y_center, width, height