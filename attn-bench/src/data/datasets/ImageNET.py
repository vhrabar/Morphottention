from pathlib import Path
from typing import Any

from datasets import Dataset, concatenate_datasets

from src.data.datasets.BaseDataset import BaseDataset


class ImageNET(BaseDataset):
    SPLIT_MAP = {
        "train": "train",
        "val": "validation",
        "validation": "validation",
        "test": "test",
    }
    _SUBDIR = "ImageNET"
    _SHARD_GLOB = "imagenet-1k-{split}-*.arrow"

    def __init__(self, root: str, split: str = "train", transform=None) -> None:
        self._hf_split = self.SPLIT_MAP[split]
        super().__init__(root=root, split=split, transform=transform)

    def _find_shards(self) -> list[Path]:
        """
        Locate parquet shards for the requested split.
        :return: List of paths to parquet shards
        """
        root = Path(self.root) / self._SUBDIR
        shards = sorted(root.glob(self._SHARD_GLOB.format(split=self._hf_split)))
        if shards:
            return shards
        else:
            raise FileNotFoundError(f"No shards found for split {self._hf_split}")

    def _load_samples(self) -> None:
        """
        Build an HF Dataset from local parquet shards.
        """
        shards = self._find_shards()

        parts = [Dataset.from_file(str(p)) for p in shards]
        self.samples = parts[0] if len(parts) == 1 else concatenate_datasets(parts)

        label_feat = self.samples.features.get("label")
        self.classes = getattr(label_feat, "names", None)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)} if self.classes else None

    def _get_raw(self, index: int) -> tuple[Any, int]:
        item = self.samples[index]
        return item["image"], int(item["label"])
