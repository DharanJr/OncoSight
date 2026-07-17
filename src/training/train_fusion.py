"""
Trains the Module 4 fusion meta-classifier: a small Logistic Regression on
top of [image model's 3 class probabilities + clinical model's 3 class
probabilities] -> combined severity (Low/Medium/High).

Reminder (see build_fusion_dataset.py docstring): trained on SYNTHETIC
randomly-paired samples, since the two source datasets share no real
patients. This demonstrates the fusion architecture; its accuracy number
does not represent real combined-diagnosis performance.

Usage:
    python -m src.fusion.train_fusion
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save-only, no GUI window — see evaluate.py for why
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    FUSION_DIR,
    FUSION_PROCESSED_DIR,
    FUSION_TEST_SPLIT,
    FUSION_CLASSES,
    METRICS_DIR,
    RANDOM_SEED,
)
from src.fusion.build_fusion_dataset import build_synthetic_pairs


def load_or_build_pairs():
    x_path = FUSION_PROCESSED_DIR / "synthetic_X.npy"
    y_path = FUSION_PROCESSED_DIR / "synthetic_y.npy"
    names_path = FUSION_PROCESSED_DIR / "synthetic_feature_names.joblib"

    if x_path.exists() and y_path.exists() and names_path.exists():
        print("Loading previously-built synthetic pairs...")
        X = np.load(x_path)
        y = np.load(y_path)
        feature_names = joblib.load(names_path)
    else:
        X, y, feature_names = build_synthetic_pairs()
    return X, y, feature_names


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Purples",
        xticklabels=FUSION_CLASSES, yticklabels=FUSION_CLASSES,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual (combined severity)")
    plt.title("Fusion Meta-Classifier — Confusion Matrix\n(SYNTHETIC paired data)")
    plt.tight_layout()
    out_path = METRICS_DIR / "synthetic_fusion_confusion_matrix.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_feature_weights(model, feature_names):
    """Logistic Regression coefficients — shows how much each modality's
    signal contributes to the final fused decision, per class."""
    coefs = model.coef_  # (n_classes, n_features)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(feature_names))
    width = 0.25
    for i, cls in enumerate(FUSION_CLASSES):
        ax.bar(x + i * width, coefs[i], width, label=cls)
    ax.set_xticks(x + width)
    ax.set_xticklabels(feature_names, rotation=30, ha="right")
    ax.set_ylabel("Coefficient weight")
    ax.set_title("Fusion Meta-Classifier — Feature Weights per Class\n(SYNTHETIC paired data)")
    ax.legend()
    plt.tight_layout()
    out_path = METRICS_DIR / "synthetic_fusion_feature_weights.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def main():
    print("=" * 60)
    print("Module 4 — Multimodal Fusion (SYNTHETIC demonstration)")
    print("=" * 60)
    print(
        "NOTE: image and clinical datasets share no real patients. Training "
        "on randomly-paired synthetic samples to demonstrate the fusion "
        "architecture — see build_fusion_dataset.py docstring.\n"
    )

    X, y, feature_names = load_or_build_pairs()
    print(f"\nTotal synthetic pairs: {X.shape[0]} | Features: {feature_names}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=FUSION_TEST_SPLIT, random_state=RANDOM_SEED, stratify=y
    )
    print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    model = LogisticRegression(max_iter=2000, random_state=RANDOM_SEED, class_weight="balanced")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report_text = classification_report(y_test, y_pred, target_names=FUSION_CLASSES, zero_division=0)

    print(f"\nTest accuracy: {accuracy:.4f}")
    print(report_text)

    report_path = METRICS_DIR / "synthetic_fusion_classification_report.txt"
    report_path.write_text(
        "NOTE: evaluated on SYNTHETIC randomly-paired data (no real linked "
        "patient records exist between the image and clinical datasets).\n\n"
        + report_text
    )

    cm_path = plot_confusion_matrix(y_test, y_pred)
    weights_path = plot_feature_weights(model, feature_names)

    model_path = FUSION_DIR / "fusion_meta_classifier.joblib"
    joblib.dump(model, model_path)

    print(f"\nConfusion matrix saved to: {cm_path}")
    print(f"Feature weights plot saved to: {weights_path}")
    print(f"Classification report saved to: {report_path}")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()