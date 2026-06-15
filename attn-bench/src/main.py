#!/usr/bin/env python3

import argparse
import time
import torch
from torch import nn

from configs import DATASETS, RUNTIMES
from data import get_loader
from utils import build_model, apply_overrides, resolve_runtime, train, build_scheduler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--runtime", choices=sorted(RUNTIMES))
    parser.add_argument("--dataset", choices=DATASETS)
    args = parser.parse_args()

    runtime = apply_overrides(resolve_runtime(args), args)


    if torch.cuda.is_available():
        cuda_index = torch.cuda.current_device()
        device = torch.device("cuda", cuda_index)
        print("GPU available:")
        print(f"GPU: {torch.cuda.get_device_name(cuda_index)}")
        print(f"CC: {torch.cuda.get_device_capability(cuda_index)}")
        print(f"VRAM: {torch.cuda.get_device_properties(cuda_index).total_memory / 1024**3:.2f} GiB")
    else:
        print("GPU error!")
        exit(1)

    model = build_model(runtime)
    model.to(device)

    train_loader, val_loader = (
        get_loader(runtime.data, "train", device),
        get_loader(runtime.data, "val", device),
    )
    optimizer = torch.optim.AdamW
    scheduler = build_scheduler(optimizer, runtime.train, len(train_loader))
    criterion = nn.CrossEntropyLoss(label_smoothing=runtime.optim.label_smoothing)
    scaler = torch.amp.GradScaler(device="cuda", enabled=True)

    train(runtime, model, train_loader, val_loader, optimizer, scheduler, scaler, criterion, device)

if __name__ == "__main__":
    main()
