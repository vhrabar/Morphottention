from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """
    Dataset abstract class that defines the structure for loading and processing datasets.
    """

    _images = None
    _labels = None

    def __init__(self, root: str, split: str, transform=None) -> None:
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform

        self.samples: Any = []
        self._load_samples()

    @abstractmethod
    def _load_samples(self) -> None:
        pass

    def _get_raw(self, index: int) -> tuple[Any, int]:
        """
        Return raw (image, label) for the given index.
        :param index: index of the sample
        :return: raw image and label
        """
        return self._images[index], int(self._labels[index])

    def _load_sample(self, index: int) -> tuple[Any, Any]:
        image, label = self._get_raw(index)

        if isinstance(image, Image.Image):
            if image.mode != "RGB":
                image = image.convert("RGB")
            image = np.asarray(image)

        if self.transform:
            if image.dtype != np.uint8:
                image = (image * 255).clip(0, 255).astype(np.uint8)
            image = self.transform(image=image)["image"]
        else:
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            tensor = torch.from_numpy(image)
            tensor = tensor.unsqueeze(0) if tensor.ndim == 2 else tensor.permute(2, 0, 1)
            image = tensor.contiguous()

        return image, label

    def __len__(self) -> int:
        if self._labels is not None:
            return len(self._labels)
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        return self._load_sample(index)
