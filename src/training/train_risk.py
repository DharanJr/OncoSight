"""
Trains clinical risk prediction models (Logistic Regression, Random Forest,
XGBoost, LightGBM).

Uses stratified 5-fold cross-validation for model selection/reporting —
chosen over a single train/test split because the clinical dataset is
tabular and modest-sized, where one split can be a lucky or unlucky draw.
After CV, each model is also fit once on the full training set and saved,
since evaluate_risk.py needs an actual fitted model to score the held-out
test set against.

Usage:
    python -m src.training.train_risk --model random_forest
    python -m src.training.train_risk --model all
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold, cross_validate

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    RISK_MODEL_DIR,
    LOG_DIR,
    CLINICAL_CV_FOLDS,
    RANDOM_SEED,
    RISK_AVAILABLE_MODELS,
)
from src.data.clinical_dataset import prepare_clinical_data
from src.models.risk_models import build_risk_model

SCORING = ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"]


def run_cross_validation(model_name: str, X_train, y_train):
    # With duplicates removed, the training set is much smaller (~120 rows
    # rather than 800), so the smallest class might not have enough samples
    # for the configured fold count. Reduce folds rather than crash.
    min_class_count = np.bincount(y_train).min()
    n_folds = min(CLINICAL_CV_FOLDS, min_class_count)
    if n_folds < CLINICAL_CV_FOLDS:
        print(
            f"[WARN] Smallest class has only {min_class_count} training samples — "
            f"reducing cross-validation from {CLINICAL_CV_FOLDS} to {n_folds} folds."
        )

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    model = build_risk_model(model_name)

    cv_results = cross_validate(
        model, X_train, y_train, cv=skf, scoring=SCORING, n_jobs=-1, return_train_score=False
    )

    summary = {
        metric: {
            "mean": float(np.mean(cv_results[f"test_{metric}"])),
            "std": float(np.std(cv_results[f"test_{metric}"])),
        }
        for metric in SCORING
    }
    return summary


def train_final_model(model_name: str, X_train, y_train):
    model = build_risk_model(model_name)
    model.fit(X_train, y_train)
    save_path = RISK_MODEL_DIR / f"{model_name}.joblib"
    joblib.dump(model, save_path)
    return model, save_path


def train_one(model_name: str, X_train, y_train):
    print(f"\n{'=' * 60}\nTraining {model_name}\n{'=' * 60}")

    print(f"Running {CLINICAL_CV_FOLDS}-fold stratified cross-validation...")
    cv_summary = run_cross_validation(model_name, X_train, y_train)
    for metric, stats in cv_summary.items():
        print(f"  {metric:<20} {stats['mean']:.4f} (+/- {stats['std']:.4f})")

    print("\nFitting final model on full training set...")
    _, save_path = train_final_model(model_name, X_train, y_train)
    print(f"Saved to: {save_path}")

    log_path = LOG_DIR / f"risk_{model_name}_cv_results.json"
    with open(log_path, "w") as f:
        json.dump(cv_summary, f, indent=2)
    print(f"CV results saved to: {log_path}")

    return cv_summary


def main():
    parser = argparse.ArgumentParser(description="Train clinical risk prediction models")
    parser.add_argument(
        "--model", type=str, default="random_forest",
        choices=RISK_AVAILABLE_MODELS + ["all"],
    )
    args = parser.parse_args()

    print("Preparing clinical dataset...")
    X_train, X_test, y_train, y_test, feature_names = prepare_clinical_data()
    print(f"Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]} | Features: {len(feature_names)}")

    models_to_run = RISK_AVAILABLE_MODELS if args.model == "all" else [args.model]

    results = {}
    for model_name in models_to_run:
        results[model_name] = train_one(model_name, X_train, y_train)

    if len(results) > 1:
        print(f"\n{'=' * 60}\nCross-validation comparison (mean accuracy)\n{'=' * 60}")
        ranked = sorted(results.items(), key=lambda kv: -kv[1]["accuracy"]["mean"])
        for name, summary in ranked:
            acc = summary["accuracy"]
            f1 = summary["f1_weighted"]
            print(f"  {name:<20} acc={acc['mean']:.4f} (+/-{acc['std']:.4f})  f1={f1['mean']:.4f}")


if __name__ == "__main__":
    main()