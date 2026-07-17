"""
Builds SYNTHETIC paired multimodal samples for Module 4 fusion.

The CT image dataset (IQ-OTH/NCCD) and the clinical risk dataset (Cancer
Patient Data Sets) have no shared patients — there is no real record
linking a specific person's scan to their specific symptom profile. This
script does NOT pretend otherwise: it randomly pairs one image-model
prediction with one clinical-model prediction to simulate what a combined
record would look like, purely to demonstrate the fusion architecture.

Every output file from this script is prefixed `synthetic_` and this
should be stated plainly in any report/viva discussion of Module 4.

Usage:
    python -m src.fusion.build_fusion_dataset
"""

import sys
from pathlib import Path

import numpy as np
import joblib
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    CHECKPOINT_DIR,
    RISK_MODEL_DIR,
    CLINICAL_PROCESSED_DIR,
    FUSION_PROCESSED_DIR,
    FUSION_N_SYNTHETIC_PAIRS,
    IMAGE_SEVERITY,
    CLINICAL_SEVERITY,
    RISK_CLASSES,
    RANDOM_SEED,
)
from src.data.dataset import get_dataloaders
from src.models.architectures import build_model
from src.training.train import get_device

IMAGE_MODEL_NAME = "resnet50"      # Module 1's best model
CLINICAL_MODEL_NAME = "random_forest"  # Module 2's model used for SHAP/fusion


def get_image_model_test_outputs():
    """
    Runs the trained image model on its real test set and returns:
      probs        (N_img, 3) softmax probabilities, column order = model's
                   own class_to_idx order (NOT assumed — read from checkpoint)
      true_classes list of N_img true class name strings
      ordered_class_names  list of 3 class names, in the same column order
                   as `probs` (fixes the same ordering trap Module 1's
                   evaluate.py had — never assume CLASS_NAMES order)
    """
    device = get_device()
    checkpoint_path = CHECKPOINT_DIR / f"{IMAGE_MODEL_NAME}_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint_path}. Train Module 1 first."
        )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(IMAGE_MODEL_NAME).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    class_to_idx = checkpoint["class_to_idx"]
    ordered_class_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]

    _, _, test_loader, _ = get_dataloaders()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.extend(labels.numpy().tolist())

    probs = np.concatenate(all_probs, axis=0)
    true_classes = [ordered_class_names[i] for i in all_labels]
    return probs, true_classes, ordered_class_names


def get_clinical_model_test_outputs():
    """
    Returns (probs, true_classes) for Module 2's saved model on its own
    real test set. Column order matches RISK_CLASSES directly, since
    RISK_CLASS_TO_IDX is a config-defined mapping we control (Low=0,
    Medium=1, High=2) rather than something sklearn assigns alphabetically
    — no reordering trap here, unlike the image side.
    """
    model_path = RISK_MODEL_DIR / f"{CLINICAL_MODEL_NAME}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"No saved model at {model_path}. Train Module 2 first.")
    model = joblib.load(model_path)

    X_test = np.load(CLINICAL_PROCESSED_DIR / "X_test.npy")
    y_test = np.load(CLINICAL_PROCESSED_DIR / "y_test.npy")

    probs = model.predict_proba(X_test)
    true_classes = [RISK_CLASSES[i] for i in y_test]
    return probs, true_classes


def build_synthetic_pairs(n_pairs: int = FUSION_N_SYNTHETIC_PAIRS):
    print("Loading real test-set outputs from both modules...")
    img_probs, img_true, img_class_names = get_image_model_test_outputs()
    clin_probs, clin_true = get_clinical_model_test_outputs()
    print(f"  Image model: {len(img_true)} real test predictions, classes={img_class_names}")
    print(f"  Clinical model: {len(clin_true)} real test predictions, classes={RISK_CLASSES}")

    rng = np.random.default_rng(RANDOM_SEED)
    img_indices = rng.integers(0, len(img_true), size=n_pairs)
    clin_indices = rng.integers(0, len(clin_true), size=n_pairs)

    X = np.zeros((n_pairs, img_probs.shape[1] + clin_probs.shape[1]))
    y = np.zeros(n_pairs, dtype=int)

    for i in range(n_pairs):
        img_idx, clin_idx = img_indices[i], clin_indices[i]
        X[i] = np.concatenate([img_probs[img_idx], clin_probs[clin_idx]])

        img_severity = IMAGE_SEVERITY[img_true[img_idx]]
        clin_severity = CLINICAL_SEVERITY[clin_true[clin_idx]]
        y[i] = max(img_severity, clin_severity)  # conservative: worse signal wins

    feature_names = [f"img_prob_{c.lower()}" for c in img_class_names] + \
                     [f"clin_prob_{c.lower()}" for c in RISK_CLASSES]

    print(
        f"\n[SYNTHETIC] Built {n_pairs} randomly-paired samples "
        "(NOT real linked patient records — see module docstring)."
    )
    print("Combined severity distribution:")
    for i, cls in enumerate(["Low", "Medium", "High"]):
        print(f"  {cls}: {(y == i).sum()}")

    np.save(FUSION_PROCESSED_DIR / "synthetic_X.npy", X)
    np.save(FUSION_PROCESSED_DIR / "synthetic_y.npy", y)
    joblib.dump(feature_names, FUSION_PROCESSED_DIR / "synthetic_feature_names.joblib")
    print(f"\nSaved to: {FUSION_PROCESSED_DIR}")

    return X, y, feature_names


if __name__ == "__main__":
    build_synthetic_pairs()