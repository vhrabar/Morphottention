from dataclasses import dataclass
from enum import IntEnum
from typing import Any



@dataclass(frozen=True)
class DataConfig:
    dataset: str = "MedMNIST"
    data_root: str = "./data"
    batch_size: int = 512
    num_workers: int = 12
    image_size: int = 32
    in_chans: int = 3
    num_classes: int = 100
    transform: Any = None
    synthetic_samples: int = 1024
    medmnist_size: int = 128


@dataclass(frozen=True)
class ModelConfig:
    image_size: int = 32
    patch_size: int = 4
    in_chans: int = 3
    embed_dim: int = 256
    depth: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    num_classes: int = 100
    window_size: int | None = None
    pruning_ratio: float = 0.5
    pruning_grid: int = 2

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2


@dataclass(frozen=True)
class OptimConfig:
    lr: float = 5e-4
    momentum: float = 0.9
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 100
    warmup_epochs: int = 3
