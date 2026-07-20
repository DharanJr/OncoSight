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
BATCH_SIZE = 32
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

# ---------------------------------------------------------------------------
# Module 2 — Clinical Risk Prediction
# ---------------------------------------------------------------------------
# The CSV can live directly in data/raw/ or in data/raw/clinical/ — the loader
# (src/data/clinical_dataset.py) searches both and matches by filename pattern,
# so exact placement/naming isn't fragile the way early image-folder naming was.
CLINICAL_DATA_DIR = PROJECT_ROOT / "data" / "raw"
CLINICAL_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "clinical"
CLINICAL_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

RISK_MODEL_DIR = PROJECT_ROOT / "checkpoints" / "risk_models"
RISK_MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Target column in the "Cancer Patient Data Sets" Kaggle dataset is "Level"
# with values Low / Medium / High. Ordinal on purpose (not one-hot) — risk
# has a natural order, and metrics/plots read better as an ordered scale.
RISK_TARGET_COLUMN = "Level"
RISK_CLASSES = ["Low", "Medium", "High"]
RISK_CLASS_TO_IDX = {name: idx for idx, name in enumerate(RISK_CLASSES)}

# Columns to drop if present — identifier columns carry no predictive signal
# and risk leaking a patient index into the model as a "feature".
CLINICAL_ID_COLUMNS = ["index", "Patient Id", "Patient Id ", "Unnamed: 0"]

CLINICAL_TEST_SPLIT = 0.20
CLINICAL_CV_FOLDS = 5  # stratified k-fold, used instead of a single train/test
                        # split for model selection — more reliable on a
                        # smaller tabular dataset than one lucky/unlucky split
RISK_AVAILABLE_MODELS = ["logistic_regression", "random_forest", "xgboost", "lightgbm"]

# ---------------------------------------------------------------------------
# Module 4 — Multimodal Fusion
# ---------------------------------------------------------------------------
# IMPORTANT: the CT image dataset and clinical risk dataset have NO shared
# patients — they come from unrelated sources. There is no real paired
# multimodal data to fuse. This module builds a SYNTHETIC demonstration:
# randomly pairs one image-model prediction with one clinical-model
# prediction to simulate what a combined patient record would look like,
# then trains a meta-classifier on those pairs. This demonstrates the fusion
# architecture and technique correctly, but the trained fusion model's
# accuracy does NOT represent real-world combined-diagnosis performance —
# state this explicitly in any report/viva discussion of this module.
FUSION_DIR = PROJECT_ROOT / "checkpoints" / "fusion"
FUSION_DIR.mkdir(parents=True, exist_ok=True)
FUSION_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "fusion"
FUSION_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

FUSION_N_SYNTHETIC_PAIRS = 800
FUSION_TEST_SPLIT = 0.20

# Maps each modality's classes onto a shared 0/1/2 severity scale so the two
# independent ground truths can be combined into one target for the
# synthetic pairs. Rule: combined severity = the MORE severe of the two
# assessments (clinically conservative — if either signal suggests high
# risk, treat the combined case as high risk).
IMAGE_SEVERITY = {"Normal": 0, "Benign": 1, "Malignant": 2}
CLINICAL_SEVERITY = {"Low": 0, "Medium": 1, "High": 2}
FUSION_CLASSES = ["Low", "Medium", "High"]  # combined severity labels, reusing risk naming

# ---------------------------------------------------------------------------
# Module 5 — LLM Report Generation (local, FREE — no API billing required)
# ---------------------------------------------------------------------------
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Uses Ollama (https://ollama.com) running locally — no API key, no billing,
# no internet required after the model is downloaded once. Falls back to a
# template-based generator automatically (src/reports/template_report.py)
# if Ollama isn't running, so report generation never hard-fails.
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:3b"  # ~2GB quantized — comfortable on 6GB VRAM
LLM_TIMEOUT_SECONDS = 60

# ---------------------------------------------------------------------------
# Module 6 — RAG Medical Assistant
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"
RAG_INDEX_DIR = PROJECT_ROOT / "data" / "processed" / "rag"
RAG_INDEX_DIR.mkdir(parents=True, exist_ok=True)

RAG_CHUNK_SIZE = 500       # characters per chunk
RAG_CHUNK_OVERLAP = 100    # characters shared between consecutive chunks
RAG_TOP_K = 4              # how many chunks to retrieve per query
RAG_RELEVANCE_THRESHOLD = 0.25  # below this similarity, treat as "no good answer"
                                  # rather than let the LLM improvise — this is
                                  # what prevents hallucination on off-topic questions
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # per project spec

# Second, independent guard against off-topic questions slipping through on a
# lucky similarity score alone — the query must ALSO contain at least one of
# these terms. Belt-and-suspenders: similarity score catches semantic
# mismatches, this keyword check catches accidental term-overlap false
# positives (e.g. a question about "risk" in an unrelated context).
RAG_TOPIC_KEYWORDS = {
    "lung", "cancer", "tumor", "tumour", "smoking", "smoke", "carcinoma",
    "nodule", "chest", "biopsy", "oncology", "metastasis", "screening",
    "chemotherapy", "radiation", "staging", "malignant", "benign", "cough",
    "breath", "respiratory", "pulmonary", "diagnosis", "treatment", "symptom",
}

# Maps filename keywords to a proper display name for citations, so answers
# say "World Health Organization" instead of a raw PDF filename. Add entries
# here if you download PDFs with different naming than these organizations'
# typical export names.
RAG_SOURCE_DISPLAY_NAMES = {
    "who": "World Health Organization (WHO)",
    "nci": "National Cancer Institute (NCI)",
    "cdc": "Centers for Disease Control and Prevention (CDC)",
    "cancer.org": "American Cancer Society (ACS)",
    "acs": "American Cancer Society (ACS)",
}