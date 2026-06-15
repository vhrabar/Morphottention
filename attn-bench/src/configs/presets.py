from dataclasses import dataclass, field, replace
from typing import Literal

from src.configs.params import  DataConfig, ModelConfig, OptimConfig, TrainConfig

ModelKind = Literal["vit"]


@dataclass(frozen=True)
class Runtime:
    name: str
    model_kind: ModelKind
    data: DataConfig
    model: ModelConfig
    optim: OptimConfig
    train: TrainConfig


_DEFAULT_OPTIM = OptimConfig()
_DEFAULT_TRAIN = TrainConfig()


@dataclass(frozen=True)
class _DatasetSpec:
    """
    Specs for each dataset
    """

    data: DataConfig
    model: ModelConfig
    optim: OptimConfig = field(default_factory=lambda: _DEFAULT_OPTIM)
    train: TrainConfig = field(default_factory=lambda: _DEFAULT_TRAIN)


_DATASETS: dict[str, _DatasetSpec] = {
    # CIFAR100
    "cifar100": _DatasetSpec(
        data=DataConfig(dataset="CIFAR100", image_size=32, in_chans=3, num_classes=100, batch_size=1024),
        model=ModelConfig(
            image_size=32, patch_size=4, in_chans=3, num_classes=100, embed_dim=256, depth=6, num_heads=8
        ),
        optim=OptimConfig(lr=1e-3),
    ),
    # FashionMNIST
    "fashionmnist": _DatasetSpec(
        data=DataConfig(dataset="FashionMNIST", image_size=28, in_chans=1, num_classes=10, batch_size=1024),
        model=ModelConfig(image_size=28, patch_size=7, in_chans=1, num_classes=10, embed_dim=192, depth=4, num_heads=6),
        optim=OptimConfig(lr=1e-3),
    ),
    # TinyImageNET
    "tinyimagenet": _DatasetSpec(
        data=DataConfig(dataset="TinyImageNET", image_size=64, in_chans=3, num_classes=200, batch_size=512),
        model=ModelConfig(
            image_size=64, patch_size=8, in_chans=3, num_classes=200, embed_dim=384, depth=8, num_heads=8
        ),
        optim=OptimConfig(lr=1e-3),
    ),
    # STL10
    "stl10": _DatasetSpec(
        data=DataConfig(dataset="STL10", image_size=96, in_chans=3, num_classes=10, batch_size=256),
        model=ModelConfig(image_size=96, patch_size=8, in_chans=3, num_classes=10, embed_dim=384, depth=8, num_heads=8),
    ),
    # MedMNIST: small
    "medmnist128": _DatasetSpec(
        data=DataConfig(
            dataset="MedMNIST", image_size=128, in_chans=3, num_classes=11, batch_size=128, medmnist_size=128
        ),
        model=ModelConfig(
            image_size=128, patch_size=16, in_chans=3, num_classes=11, embed_dim=384, depth=8, num_heads=8
        ),
        optim=OptimConfig(lr=1e-3),
    ),
    # MedMNIST: large
    "medmnist224": _DatasetSpec(
        data=DataConfig(
            dataset="MedMNIST", image_size=224, in_chans=3, num_classes=11, batch_size=40, medmnist_size=224
        ),
        model=ModelConfig(
            image_size=224, patch_size=16, in_chans=3, num_classes=11, embed_dim=384, depth=8, num_heads=8
        ),
    ),
    # HAM10000 — skin lesion classification (7 classes)
    "ham10k": _DatasetSpec(
        data=DataConfig(dataset="HAM10k", image_size=224, in_chans=3, num_classes=7, batch_size=128),
        model=ModelConfig(
            image_size=224, patch_size=16, in_chans=3, num_classes=7, embed_dim=384, depth=8, num_heads=8
        ),
        optim=OptimConfig(lr=1e-3),
    ),
    # ImageNet — 224x224
    "imagenet": _DatasetSpec(
        data=DataConfig(dataset="ImageNet", image_size=224, in_chans=3, num_classes=1000, batch_size=128),
        model=ModelConfig(
            image_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12
        ),
        optim=OptimConfig(lr=1e-3, weight_decay=5e-2),
        train=TrainConfig(epochs=300, warmup_epochs=20),
    ),
    # Synthetic
    "synthetic": _DatasetSpec(
        data=DataConfig(dataset="Synthetic", image_size=32, in_chans=3, num_classes=100, synthetic_samples=1024),
        model=ModelConfig(
            image_size=32, patch_size=4, in_chans=3, num_classes=100, embed_dim=128, depth=2, num_heads=4
        ),
        train=TrainConfig(epochs=20, warmup_epochs=3),
    ),
}


def _build_registry() -> dict[str, Runtime]:
    out: dict[str, Runtime] = {}
    for ds_name, spec in _DATASETS.items():
        out[ds_name] = Runtime(
            name=ds_name,
            model_kind="vit",
            data=spec.data,
            model=spec.model,
            optim=spec.optim,
            train=spec.train,
        )
    return out

DATASETS = tuple(_DATASETS.keys())
RUNTIMES = _build_registry()
