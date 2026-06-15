import math
import time

import torch
from torch import nn, Device
from torch.cpu.amp import GradScaler
from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from configs import TrainConfig
from networks import ViT


def build_scheduler(optimizer: Optimizer, params: TrainConfig, steps_per_epoch: int) -> LRScheduler:
    """
    Cosine decay with linear warmup scheduler
    :param optimizer: optimizer to be scheduled
    :param params: training parameters
    :param steps_per_epoch: number of steps per epoch
    :return: scheduler
    """
    total_steps = params.epochs * steps_per_epoch
    warmup_steps = params.warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model: ViT, loader: DataLoader, optimizer: Optimizer,scheduler: LRScheduler,scaler: GradScaler, criterion: CrossEntropyLoss, device,epoch: int,)-> tuple[float, float]:
    """
    SIngle epoch training loop
    :param model:
    :param loader:
    :param optimizer:
    :param scheduler:
    :param scaler:
    :param criterion:
    :param device:
    :param epoch:
    :return: average loss and accuracy for the epoch
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for step, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(imgs)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()

        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)

        if (step + 1) % max(1, len(loader) // 2) == 0:
            lr = scheduler.get_last_lr()[0]
            print(
                f"[epoch {epoch} | step {step + 1}/{len(loader)}]  "
                f"loss={total_loss / total:.4f}  acc={100 * correct / total:.2f}%  lr={lr:.2e}"
            )

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model: ViT, loader: DataLoader, criterion: CrossEntropyLoss, device) -> tuple[float, float]:
    """
    Evaluation loop
    :param model:
    :param loader:
    :param criterion:
    :param device:
    :return:
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(imgs)
            loss = criterion(logits, labels)

        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total

def train(runtime, model: ViT, train_loader: DataLoader, val_loader: DataLoader, optimizer: Optimizer, scheduler: LRScheduler, scaler: GradScaler, criterion: CrossEntropyLoss, device: Device) -> None:
    best_acc = 0.0
    ckpt_path = f"models/{runtime.name}_best.pt"

    for epoch in range(1, runtime.train.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, criterion, device,epoch)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - t0
        print(
            f"\nEpoch {epoch:>3}/{runtime.train.epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={100 * train_acc:.2f}%  |  "
            f"val_loss={val_loss:.4f}  val_acc={100 * val_acc:.2f}%  "
            f"({elapsed:.1f}s)\n"
        )


        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "runtime": runtime.name,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_acc": val_acc,
                },
                ckpt_path,
            )
            print(f"New best saved to {ckpt_path}  (val_acc={100 * best_acc:.2f}%)")