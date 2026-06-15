import csv
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.data.datasets.BaseDataset import BaseDataset

CLASS_TO_IDX: dict[str, int] = {
    "akiec": 0,
    "bcc": 1,
    "bkl": 2,
    "df": 3,
    "mel": 4,
    "nv": 5,
    "vasc": 6,
}


def _read_labels(csv_path: Path) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["image_id"], CLASS_TO_IDX[row["dx"]]))
    return rows


def _ensure_extracted(zip_path: Path, inner_dir_name: str, out_root: Path) -> Path:
    """
    Extract zip on first load. Returns the directory containing the .jpg files.
    """
    target = out_root / inner_dir_name
    if target.exists() and any(target.glob("*.jpg")):
        return target

    out_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if member.startswith("__MACOSX/") or not member.endswith(".jpg"):
                continue
            zf.extract(member, out_root)
    return target


def _build_image_index(image_dirs: list[Path]) -> dict[str, Path]:
    index = {}
    for d in image_dirs:
        for p in d.glob("*.jpg"):
            index[p.stem] = p
    return index


class HAM10k(BaseDataset):
    """
    HAM10000 skin lesion dataset.
    """

    def _load_samples(self) -> None:
        root = Path(self.root) / "HAM10k"

        if self.split == "train":
            csv_path = root / "HAM10000_metadata"
            image_dirs = [
                root / "HAM10000_images_part_1",
                root / "HAM10000_images_part_2",
            ]
        else:
            csv_path = root / "ISIC2018_Task3_Test_GroundTruth.csv"
            test_dir = _ensure_extracted(
                root / "ISIC2018_Task3_Test_Images.zip",
                "ISIC2018_Task3_Test_Images",
                root,
            )
            image_dirs = [test_dir]

        index = _build_image_index([d for d in image_dirs if d.exists()])
        entries = _read_labels(csv_path)

        paths = []
        labels = []
        for image_id, label in entries:
            path = index.get(image_id)
            if path is None:
                continue
            paths.append(path)
            labels.append(label)

        if not paths:
            raise RuntimeError(f"No HAM10k images matched labels for split={self.split}")

        self._paths = paths
        self._labels = np.asarray(labels, dtype=np.int64)

    def _get_raw(self, index: int) -> tuple[Any, int]:
        image = Image.open(self._paths[index])
        return image, int(self._labels[index])
