from src.data.datasets.ImageNET import ImageNET


class TinyImageNET(ImageNET):
    SPLIT_MAP = {
        "train": "train",
        "val": "valid",
        "valid": "valid",
    }
    _SUBDIR = "Tiny-ImageNET"
    _SHARD_GLOB = "tiny-imagenet-{split}.arrow"

    NUM_CLASSES = 200
