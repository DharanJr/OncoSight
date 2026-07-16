"""
Evaluates trained clinical risk models on the held-out test set, and
produces feature importance analysis (required by the project spec).

Usage:
    python -m src.evaluation.evaluate_risk --model random_forest
    python -m src.evaluation.evaluate_risk --model all
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — we only ever save PNGs to
                        # disk, never display a window, so this avoids the
                        # Tkinter/PIL DLL crash some locked-down Windows
                        # setups hit with the default backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    RISK_MODEL_DIR,
    METRICS_DIR,
    CLINICAL_PROCESSED_DIR,
    RISK_CLASSES,
    RISK_AVAILABLE_MODELS,
)


def load_test_data():
    X_test = np.load(CLINICAL_PROCESSED_DIR / "X_test.npy")
    y_test = np.load(CLINICAL_PROCESSED_DIR / "y_test.npy")
    feature_names = joblib.load(CLINICAL_PROCESSED_DIR / "feature_names.joblib")
    return X_test, y_test, feature_names


def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Oranges",
        xticklabels=RISK_CLASSES, yticklabels=RISK_CLASSES,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Risk Confusion Matrix — {model_name}")
    plt.tight_layout()
    out_path = METRICS_DIR / f"risk_{model_name}_confusion_matrix.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def get_feature_importance(model, model_name, feature_names):
    """
    Returns a (feature_name, importance) list, sorted descending.
    Logistic Regression: uses mean absolute coefficient across classes.
    Tree-based models (RF/XGBoost/LightGBM): uses built-in feature_importances_.
    """
    if hasattr(model, "coef_"):
        # coef_ shape is (n_classes, n_features) for multi-class LogisticRegression
        importances = np.abs(model.coef_).mean(axis=0)
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        return None

    pairs = sorted(zip(feature_names, importances), key=lambda x: -x[1])
    return pairs


def plot_feature_importance(pairs, model_name, top_n=15):
    top_pairs = pairs[:top_n]
    names = [p[0] for p in top_pairs][::-1]
    values = [p[1] for p in top_pairs][::-1]

    plt.figure(figsize=(7, max(4, len(names) * 0.35)))
    plt.barh(names, values, color="#2c7fb8")
    plt.xlabel("Importance")
    plt.title(f"Top {len(names)} Feature Importances — {model_name}")
    plt.tight_layout()
    out_path = METRICS_DIR / f"risk_{model_name}_feature_importance.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def evaluate_risk_model(model_name: str):
    model_path = RISK_MODEL_DIR / f"{model_name}.joblib"
    if not model_path.exists():
        print(f"[SKIP] No saved model found for {model_name} at {model_path}")
        return None

    print(f"\n{'=' * 60}\nEvaluating {model_name}\n{'=' * 60}")

    model = joblib.load(model_path)
    X_test, y_test, feature_names = load_test_data()

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    report_text = classification_report(y_test, y_pred, target_names=RISK_CLASSES, zero_division=0)
    report_path = METRICS_DIR / f"risk_{model_name}_classification_report.txt"
    report_path.write_text(report_text)

    cm_path = plot_confusion_matrix(y_test, y_pred, model_name)

    importance_pairs = get_feature_importance(model, model_name, feature_names)
    importance_path = None
    if importance_pairs:
        importance_path = plot_feature_importance(importance_pairs, model_name)
        importance_json_path = METRICS_DIR / f"risk_{model_name}_feature_importance.json"
        with open(importance_json_path, "w") as f:
            json.dump([{"feature": n, "importance": float(v)} for n, v in importance_pairs], f, indent=2)

    metrics = {
        "model_name": model_name,
        "accuracy": accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
    }
    metrics_path = METRICS_DIR / f"risk_{model_name}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(report_text)
    print(f"Confusion matrix saved to: {cm_path}")
    if importance_path:
        print(f"Feature importance plot saved to: {importance_path}")
        print("\nTop 5 most important features:")
        for name, val in importance_pairs[:5]:
            print(f"  {name:<30} {val:.4f}")
    print(f"Metrics JSON saved to: {metrics_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate clinical risk models")
    parser.add_argument(
        "--model", type=str, default="random_forest",
        choices=RISK_AVAILABLE_MODELS + ["all"],
    )
    args = parser.parse_args()

    models_to_run = RISK_AVAILABLE_MODELS if args.model == "all" else [args.model]

    all_metrics = []
    for model_name in models_to_run:
        result = evaluate_risk_model(model_name)
        if result:
            all_metrics.append(result)

    if len(all_metrics) > 1:
        print(f"\n{'=' * 60}\nModel comparison (test set)\n{'=' * 60}")
        print(f"{'Model':<20}{'Accuracy':>10}{'F1':>10}")
        for m in sorted(all_metrics, key=lambda x: -x["accuracy"]):
            print(f"{m['model_name']:<20}{m['accuracy']:>10.4f}{m['f1_weighted']:>10.4f}")


if __name__ == "__main__":
    main()