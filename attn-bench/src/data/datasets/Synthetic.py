import torch
from PIL import Image

from src.data.datasets.BaseDataset import BaseDataset


class Synthetic(BaseDataset):
    def __init__(
        self,
        root: str = "",
        split: str = "train",
        transform=None,
        image_shape: tuple[int, int, int] = (1, 1, 1),
        num_classes: int = 1000,
        length: int = 10_000,
        seed: int = 1,
        as_pil: bool = False,
    ) -> None:
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.length = length
        self.seed = seed
        self.as_pil = as_pil
        super().__init__(root=root, split=split, transform=None)

    def _load_samples(self) -> None:
        self.samples = range(self.length)

    def _load_sample(self, index: int):
        gen = torch.Generator().manual_seed(self.seed + index)

        if self.as_pil:
            c, h, w = self.image_shape
            arr = torch.randint(0, 256, (h, w, c), generator=gen, dtype=torch.uint8).numpy()
            image = Image.fromarray(arr.squeeze() if c == 1 else arr)
        else:
            image = torch.randn(self.image_shape, generator=gen)

        label = index % self.num_classes  # irrelevant
        return image, label
