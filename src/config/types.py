import torch

from dataclasses import dataclass
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

@dataclass
class TrainingResults:
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
class BoundingBox:
    category: int   #Fire-1, Smoke-0
    coordinates: tuple[float, float, float, float] #x_center, y_center, width, height