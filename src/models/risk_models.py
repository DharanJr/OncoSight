"""
Builders for the four clinical risk classifiers: Logistic Regression,
Random Forest, XGBoost, LightGBM. Kept as simple factory functions so
train_risk.py can loop over RISK_AVAILABLE_MODELS generically.
"""

import sys
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import RANDOM_SEED, RISK_CLASSES


def build_logistic_regression():
    # multi_class defaults to 'auto' -> softmax for 3+ classes in recent
    # sklearn; max_iter raised since scaled features + 3 classes sometimes
    # need more than the default 100 iterations to converge cleanly.
    return LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_SEED,
        class_weight="balanced",  # dataset's risk levels are rarely perfectly balanced
    )


def build_random_forest():
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


def build_xgboost():
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=len(RISK_CLASSES),
        eval_metric="mlogloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


def build_lightgbm():
    return LGBMClassifier(
        n_estimators=300,
        max_depth=-1,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multiclass",
        num_class=len(RISK_CLASSES),
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )


MODEL_BUILDERS = {
    "logistic_regression": build_logistic_regression,
    "random_forest": build_random_forest,
    "xgboost": build_xgboost,
    "lightgbm": build_lightgbm,
}


def build_risk_model(model_name: str):
    if model_name not in MODEL_BUILDERS:
        raise ValueError(
            f"Unknown model_name '{model_name}'. Expected one of: {list(MODEL_BUILDERS)}"
        )
    return MODEL_BUILDERS[model_name]()