"""
Central configuration for the lung cancer image-classification pipeline (Module 1).

Everything that other modules (training, evaluation, Grad-CAM, API) need to agree on
lives here so there is exactly one place to change it.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Expected raw dataset layout (IQ-OTH/NCCD style):
#   data/raw/Normal/*.jpg
#   data/raw/Benign/*.jpg
#   data/raw/Malignant/*.jpg
# Download the dataset yourself and place it here — this repo does not ship data.
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# After running src/data/prepare_dataset.py, split data lives here:
#   data/processed/train/<class>/*.jpg
#   data/processed/val/<class>/*.jpg
#   data/processed/test/<class>/*.jpg
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
HEATMAP_DIR = PROJECT_ROOT / "outputs" / "heatmaps"

for d in [CHECKPOINT_DIR, LOG_DIR, METRICS_DIR, HEATMAP_DIR, PROCESSED_DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------
CLASS_NAMES = ["Normal", "Benign", "Malignant"]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}

# ---------------------------------------------------------------------------
# Image / training hyperparameters
# ---------------------------------------------------------------------------
IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 2  # lower this to 0 on Windows if you hit multiprocessing issues

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# ImageNet normalization stats — used because all three backbones (CNN baseline
# is trained from scratch but kept consistent for comparability, ResNet50 and
# EfficientNet are ImageNet-pretrained) expect inputs in this range.
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 6
LR_SCHEDULER_PATIENCE = 3
LR_SCHEDULER_FACTOR = 0.5

# Which backbones to train when you run train.py with --model all
AVAILABLE_MODELS = ["cnn_baseline", "resnet50", "efficientnet_b0"]

DEVICE = "cuda"  # train.py falls back to "cpu" automatically if CUDA isn't available
