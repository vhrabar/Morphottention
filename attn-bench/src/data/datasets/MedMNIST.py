import shutil
import zipfile
from pathlib import Path

import numpy as np

from src.data.datasets.BaseDataset import BaseDataset


def _ensure_npy_cache(npz_path: Path, cache_dir: Path, x_key: str, y_key: str) -> tuple[Path, Path]:
    """
    cache npz sources in npy format
    """
    images_path = cache_dir / f"{x_key}.npy"
    labels_path = cache_dir / f"{y_key}.npy"

    if images_path.exists() and labels_path.exists():
        return images_path, labels_path

    cache_dir.mkdir(parents=True, exist_ok=True)

    if not images_path.exists():
        tmp_path = images_path.with_suffix(".npy.tmp")
        with zipfile.ZipFile(npz_path) as zf, zf.open(f"{x_key}.npy") as src, open(tmp_path, "wb") as dst:
            shutil.copyfileobj(src, dst, length=1 << 20)
        tmp_path.replace(images_path)

    if not labels_path.exists():
        with np.load(npz_path) as data:
            labels = np.asarray(data[y_key])
        if labels.ndim > 1:
            labels = labels.squeeze(-1)
        np.save(labels_path, labels.astype(np.int64))

    return images_path, labels_path


class MedMNIST(BaseDataset):
    _SPLIT_KEYS = {
        "train": ("train_images", "train_labels"),
        "val": ("val_images", "val_labels"),
        "test": ("test_images", "test_labels"),
    }

    def __init__(self, root: str, split: str = "train", transform=None, size: int = 128):
        """
        size: image resolution, 128 or 224
        """
        self.size = size
        super().__init__(root=root, split=split, transform=transform)

    def _load_samples(self) -> None:
        root = Path(self.root) / "MedMNIST"
        npz_path = root / f"pathmnist_{self.size}.npz"
        cache_dir = root / f"pathmnist_{self.size}_cache"
        x_key, y_key = self._SPLIT_KEYS[self.split]

        images_path, labels_path = _ensure_npy_cache(npz_path, cache_dir, x_key, y_key)

        self._images = np.load(images_path, mmap_mode="r")
        self._labels = np.load(labels_path)
