"""
Trains a lung CT classifier (CNN baseline, ResNet50, or EfficientNet-B0).

Usage:
    python -m src.training.train --model resnet50
    python -m src.training.train --model all          # trains all 3 back-to-back
    python -m src.training.train --model cnn_baseline --epochs 15 --lr 5e-4

Each run writes:
    checkpoints/<model_name>_best.pt
    outputs/logs/<model_name>_history.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    CHECKPOINT_DIR,
    LOG_DIR,
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE,
    LR_SCHEDULER_PATIENCE,
    LR_SCHEDULER_FACTOR,
    AVAILABLE_MODELS,
)
from src.data.dataset import get_dataloaders
from src.models.architectures import build_model
from src.training.utils import EarlyStopping, CheckpointManager


def get_device() -> torch.device:
    if DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def train_one_model(model_name: str, epochs: int, lr: float, batch_size: int):
    device = get_device()
    print(f"\n{'=' * 60}\nTraining {model_name} on {device}\n{'=' * 60}")

    train_loader, val_loader, _, class_to_idx = get_dataloaders(batch_size=batch_size)
    print(f"Class mapping: {class_to_idx}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = build_model(model_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE
    )

    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)
    checkpoint_path = CHECKPOINT_DIR / f"{model_name}_best.pt"
    checkpointer = CheckpointManager(checkpoint_path)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

    for epoch in range(1, epochs + 1):
        start = time.time()

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        improved = checkpointer.step(
            model, val_acc, extra={"class_to_idx": class_to_idx, "model_name": model_name}
        )
        marker = " (best)" if improved else ""

        elapsed = time.time() - start
        print(
            f"Epoch {epoch:>3}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}{marker} | "
            f"lr={current_lr:.2e} | {elapsed:.1f}s"
        )

        if early_stopping.step(val_loss):
            print(f"Early stopping triggered at epoch {epoch} (no improvement for {EARLY_STOPPING_PATIENCE} epochs).")
            break

    log_path = LOG_DIR / f"{model_name}_history.json"
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val_acc for {model_name}: {checkpointer.best_val_acc:.4f}")
    print(f"Checkpoint saved to: {checkpoint_path}")
    print(f"Training history saved to: {log_path}")

    return checkpointer.best_val_acc


def main():
    parser = argparse.ArgumentParser(description="Train lung CT classification models")
    parser.add_argument(
        "--model", type=str, default="resnet50",
        choices=AVAILABLE_MODELS + ["all"],
        help="Which architecture to train, or 'all' to train each in sequence",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    models_to_run = AVAILABLE_MODELS if args.model == "all" else [args.model]

    results = {}
    for model_name in models_to_run:
        results[model_name] = train_one_model(
            model_name, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size
        )

    if len(results) > 1:
        print(f"\n{'=' * 60}\nModel comparison (best val accuracy)\n{'=' * 60}")
        for name, acc in sorted(results.items(), key=lambda kv: -kv[1]):
            print(f"  {name:<18} {acc:.4f}")


if __name__ == "__main__":
    main()
