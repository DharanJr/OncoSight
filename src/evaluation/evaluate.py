"""
Evaluates a trained checkpoint on the held-out test split.

Usage:
    python -m src.evaluation.evaluate --model resnet50
    python -m src.evaluation.evaluate --model all

Produces, per model:
    outputs/metrics/<model_name>_classification_report.txt
    outputs/metrics/<model_name>_confusion_matrix.png
    outputs/metrics/<model_name>_roc_curves.png
    outputs/metrics/<model_name>_metrics.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — avoids a Tkinter/PIL DLL
                        # crash some locked-down Windows setups hit
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_fscore_support,
    accuracy_score,
)
from sklearn.preprocessing import label_binarize

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import CHECKPOINT_DIR, METRICS_DIR, CLASS_NAMES, AVAILABLE_MODELS
from src.data.dataset import get_dataloaders
from src.models.architectures import build_model
from src.training.train import get_device


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)

        all_labels.extend(labels.numpy())
        all_preds.extend(preds)
        all_probs.extend(probs)

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred, model_name, ordered_class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=ordered_class_names, yticklabels=ordered_class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    out_path = METRICS_DIR / f"{model_name}_confusion_matrix.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_roc_curves(y_true, y_probs, model_name, ordered_class_names):
    y_true_bin = label_binarize(y_true, classes=range(len(ordered_class_names)))

    plt.figure(figsize=(6, 5))
    for i, cls in enumerate(ordered_class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        auc = roc_auc_score(y_true_bin[:, i], y_probs[:, i])
        plt.plot(fpr, tpr, label=f"{cls} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves (One-vs-Rest) — {model_name}")
    plt.legend()
    plt.tight_layout()
    out_path = METRICS_DIR / f"{model_name}_roc_curves.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def evaluate_model(model_name: str):
    device = get_device()
    checkpoint_path = CHECKPOINT_DIR / f"{model_name}_best.pt"
    if not checkpoint_path.exists():
        print(f"[SKIP] No checkpoint found for {model_name} at {checkpoint_path}")
        return None

    print(f"\n{'=' * 60}\nEvaluating {model_name}\n{'=' * 60}")

    # weights_only=False is required here because our checkpoints bundle a
    # plain dict (class_to_idx, model_name) alongside the tensor weights, not
    # just tensors. Safe in this project since we only ever load checkpoints
    # this same codebase wrote — never load a checkpoint from an untrusted
    # source with weights_only=False.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(model_name).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    _, _, test_loader, class_to_idx = get_dataloaders()
    # class_to_idx is the ground truth for which label index means which
    # class (ImageFolder assigns indices alphabetically, e.g. Benign=0,
    # Malignant=1, Normal=2) — never assume CLASS_NAMES' written order
    # matches this, always derive display order from it directly.
    ordered_class_names = [
        name for name, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])
    ]
    y_true, y_pred, y_probs = collect_predictions(model, test_loader, device)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    y_true_bin = label_binarize(y_true, classes=range(len(ordered_class_names)))
    try:
        roc_auc_macro = roc_auc_score(y_true_bin, y_probs, average="macro", multi_class="ovr")
    except ValueError:
        roc_auc_macro = None  # can happen if a class is entirely absent from the test split

    report_text = classification_report(y_true, y_pred, target_names=ordered_class_names, zero_division=0)
    report_path = METRICS_DIR / f"{model_name}_classification_report.txt"
    report_path.write_text(report_text)

    cm_path = plot_confusion_matrix(y_true, y_pred, model_name, ordered_class_names)
    roc_path = plot_roc_curves(y_true, y_probs, model_name, ordered_class_names)

    metrics = {
        "model_name": model_name,
        "accuracy": accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "roc_auc_macro": roc_auc_macro,
        "val_acc_at_checkpoint": checkpoint.get("val_acc"),
    }
    metrics_path = METRICS_DIR / f"{model_name}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(report_text)
    print(f"ROC-AUC (macro, OvR): {roc_auc_macro}")
    print(f"Confusion matrix saved to: {cm_path}")
    print(f"ROC curves saved to: {roc_path}")
    print(f"Metrics JSON saved to: {metrics_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate lung CT classification models")
    parser.add_argument(
        "--model", type=str, default="resnet50",
        choices=AVAILABLE_MODELS + ["all"],
    )
    args = parser.parse_args()

    models_to_run = AVAILABLE_MODELS if args.model == "all" else [args.model]

    all_metrics = []
    for model_name in models_to_run:
        result = evaluate_model(model_name)
        if result:
            all_metrics.append(result)

    if len(all_metrics) > 1:
        print(f"\n{'=' * 60}\nModel comparison (test set)\n{'=' * 60}")
        print(f"{'Model':<18}{'Accuracy':>10}{'F1':>10}{'ROC-AUC':>10}")
        for m in sorted(all_metrics, key=lambda x: -x["accuracy"]):
            auc_str = f"{m['roc_auc_macro']:.4f}" if m["roc_auc_macro"] else "N/A"
            print(f"{m['model_name']:<18}{m['accuracy']:>10.4f}{m['f1_weighted']:>10.4f}{auc_str:>10}")


if __name__ == "__main__":
    main()