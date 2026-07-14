"""
PyTorch Dataset + DataLoader builders for the lung CT classification task.

Train-time augmentation includes random rotation, horizontal/vertical flip,
random resized crop, and brightness/contrast jitter, per the project spec.
Val/test transforms are deterministic (resize + normalize only).
"""

import sys
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    PROCESSED_DATA_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    CLASS_NAMES,
)


def build_transforms():
    train_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.85, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ]
    )

    return train_transform, eval_transform


def build_class_balanced_sampler(dataset: datasets.ImageFolder) -> WeightedRandomSampler:
    """
    Handles class imbalance (IQ-OTH/NCCD is not evenly split across
    Normal/Benign/Malignant) by oversampling minority classes during training.
    """
    targets = [label for _, label in dataset.samples]
    class_counts = Counter(targets)
    num_samples = len(targets)

    class_weights = {
        cls_idx: num_samples / count for cls_idx, count in class_counts.items()
    }
    sample_weights = [class_weights[label] for label in targets]

    return WeightedRandomSampler(
        weights=sample_weights, num_samples=num_samples, replacement=True
    )


def get_dataloaders(batch_size: int = BATCH_SIZE, num_workers: int = NUM_WORKERS):
    if not (PROCESSED_DATA_DIR / "train").exists():
        raise FileNotFoundError(
            f"{PROCESSED_DATA_DIR / 'train'} not found. "
            "Run `python -m src.data.prepare_dataset` first."
        )

    train_transform, eval_transform = build_transforms()

    train_dataset = datasets.ImageFolder(
        PROCESSED_DATA_DIR / "train", transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        PROCESSED_DATA_DIR / "val", transform=eval_transform
    )
    test_dataset = datasets.ImageFolder(
        PROCESSED_DATA_DIR / "test", transform=eval_transform
    )

    # Sanity check: ImageFolder assigns class indices alphabetically, so this
    # should line up with CLASS_NAMES sorted — verify rather than assume.
    expected = {cls: idx for idx, cls in enumerate(sorted(CLASS_NAMES))}
    assert train_dataset.class_to_idx == expected, (
        f"Class index mismatch: {train_dataset.class_to_idx} vs expected {expected}"
    )

    sampler = build_class_balanced_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader, train_dataset.class_to_idx
