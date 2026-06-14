import multiprocessing as mp
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.configs import DataConfig
from src.utils import augmenter, eval_transform


def get_loader(config: DataConfig, split: str, device: torch.device) -> DataLoader[Any]:
    """
    Loads the data according to the configuration.
    :param: config: The configuration dt containing the data loading parameters.
    :param: split: The split of the data to load.
    :param: device: The device to load the data on.

    :return: The loaded data.
    """
    transform = config.transform
    if transform is None:
        transform = augmenter(config.image_size) if split == "train" else eval_transform(config.image_size)

    match config.dataset:
        case "TinyImageNET":
            from src.data.datasets.TinyImageNET import TinyImageNET

            dataset = TinyImageNET(root=config.data_root, split=split, transform=transform)

        case "MedMNIST":
            from src.data.datasets.MedMNIST import MedMNIST

            dataset = MedMNIST(
                root=config.data_root,
                split=split,
                transform=transform,
                size=config.medmnist_size,
            )
        case "HAM10k":
            from src.data.datasets.ham10k import HAM10k

            dataset = HAM10k(root=config.data_root, split=split, transform=transform)
        case "Synthetic":
            from src.data.datasets.Synthetic import Synthetic

            dataset = Synthetic(
                root=config.data_root,
                split=split,
                transform=transform,
                image_shape=(config.in_chans, config.image_size, config.image_size),
                num_classes=config.num_classes,
                length=config.synthetic_samples,
            )
        case _:
            raise ValueError(f"Unsupported dataset: {config.dataset}")

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=(split == "train"),
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(config.num_workers > 0),
        drop_last=(split == "train"),
        multiprocessing_context=(mp.get_context("fork") if config.num_workers > 0 else None),
    )

    return loader
