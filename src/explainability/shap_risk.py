"""
Per-prediction explainability for the clinical risk model (Module 3) —
implemented WITHOUT the `shap` / `numba` packages.

Why: on this machine, `shap` transitively imports `numba`, whose compiled
DLL gets blocked by a Windows Application Control policy (the same class of
issue that blocked matplotlib's Tkinter backend earlier). Rather than
depend on a package that keeps hitting blocked DLLs, this implements the
same underlying idea directly with NumPy: the Saabas method (Saabas, 2014)
— the direct mathematical predecessor to SHAP for tree ensembles, and the
same algorithm the `treeinterpreter` package implements.

How it works, per tree in the Random Forest: walk the decision path from
root to leaf for a given patient. At each split, the class-probability
prediction shifts by some amount — attribute that shift to whichever
feature caused the split. Summing these shifts across the path (and
averaging across all trees in the forest) decomposes the final prediction
into: bias (the forest's average prediction with no information) + each
feature's individual contribution. This is exact for a single tree and a
close, well-established approximation for a forest.

Module 2's feature importance answered "which features matter most
overall." This answers "why did THIS specific patient get predicted as
High risk" — the per-patient explanation a doctor actually needs.

Usage:
    python -m src.explainability.shap_risk
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save-only, no GUI window — see evaluate.py for why
import matplotlib.pyplot as plt
import numpy as np
import joblib

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import RISK_MODEL_DIR, CLINICAL_PROCESSED_DIR, METRICS_DIR, RISK_CLASSES

TREE_MODEL_NAME = "random_forest"


def load_model_and_data():
    model_path = RISK_MODEL_DIR / f"{TREE_MODEL_NAME}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No saved model at {model_path}. Train Module 2 first: "
            "python -m src.training.train_risk --model random_forest"
        )
    model = joblib.load(model_path)
    X_test = np.load(CLINICAL_PROCESSED_DIR / "X_test.npy")
    y_test = np.load(CLINICAL_PROCESSED_DIR / "y_test.npy")
    feature_names = joblib.load(CLINICAL_PROCESSED_DIR / "feature_names.joblib")
    return model, X_test, y_test, feature_names


def _tree_contributions_single(tree, x, n_features, n_classes):
    """
    Walks one fitted DecisionTreeClassifier's decision path for a single
    sample x, returns (bias, contributions) where contributions has shape
    (n_features, n_classes). bias + contributions.sum(axis=0) == the leaf's
    predicted class-probability vector, exactly, for this one tree.
    """
    t = tree.tree_
    node_id = 0
    path = [0]
    while t.children_left[node_id] != -1:  # -1 (TREE_LEAF) means it's a leaf
        feature_idx = t.feature[node_id]
        threshold = t.threshold[node_id]
        if x[feature_idx] <= threshold:
            node_id = t.children_left[node_id]
        else:
            node_id = t.children_right[node_id]
        path.append(node_id)

    def node_probs(nid):
        raw = t.value[nid, 0, :]
        total = raw.sum()
        return raw / total if total > 0 else np.zeros(n_classes)

    probs_path = [node_probs(n) for n in path]
    bias = probs_path[0]

    contributions = np.zeros((n_features, n_classes))
    for i in range(1, len(path)):
        split_feature = t.feature[path[i - 1]]
        contributions[split_feature] += probs_path[i] - probs_path[i - 1]

    return bias, contributions


def compute_forest_contributions(model, X):
    """
    Averages per-tree Saabas contributions across every tree in the forest.
    Returns (bias, contributions) where contributions has shape
    (n_samples, n_features, n_classes).
    """
    n_samples = X.shape[0]
    n_features = X.shape[1]
    n_classes = len(model.classes_)
    n_trees = len(model.estimators_)

    all_contributions = np.zeros((n_samples, n_features, n_classes))
    bias_sum = np.zeros(n_classes)

    for tree in model.estimators_:
        for i in range(n_samples):
            bias, contrib = _tree_contributions_single(tree, X[i], n_features, n_classes)
            all_contributions[i] += contrib
            if i == 0:
                bias_sum += bias

    all_contributions /= n_trees
    bias_avg = bias_sum / n_trees
    return bias_avg, all_contributions


def plot_summary_per_class(contributions, feature_names):
    n_classes = contributions.shape[2]
    for class_idx in range(n_classes):
        class_name = RISK_CLASSES[class_idx]
        mean_abs = np.abs(contributions[:, :, class_idx]).mean(axis=0)
        order = np.argsort(mean_abs)[::-1][:15]

        plt.figure(figsize=(7, max(4, len(order) * 0.35)))
        plt.barh(
            [feature_names[i] for i in order][::-1],
            [mean_abs[i] for i in order][::-1],
            color="#2c7fb8",
        )
        plt.xlabel("Mean |contribution| to prediction")
        plt.title(f"Feature Contribution Summary — Risk = {class_name}")
        plt.tight_layout()
        out_path = METRICS_DIR / f"shap_summary_{class_name.lower()}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path}")


def plot_individual_explanation(contributions, X, y, feature_names, sample_idx, save_path):
    true_class_idx = int(y[sample_idx])
    class_contrib = contributions[sample_idx, :, true_class_idx]
    values = X[sample_idx]

    pairs = sorted(zip(feature_names, class_contrib, values), key=lambda x: -abs(x[1]))[:10]
    labels = [f"{name} = {val:.2f}" for name, _, val in pairs][::-1]
    contrib_values = [c for _, c, _ in pairs][::-1]
    colors = ["#d62728" if c > 0 else "#1f77b4" for c in contrib_values]

    plt.figure(figsize=(7, 5))
    plt.barh(labels, contrib_values, color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Contribution to this prediction")
    plt.title(
        f"Why this patient was predicted '{RISK_CLASSES[true_class_idx]}' risk\n"
        "(red = pushes toward this class, blue = pushes away)"
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    print(f"Loading {TREE_MODEL_NAME} model and test data...")
    model, X_test, y_test, feature_names = load_model_and_data()

    print("Computing per-prediction feature contributions (Saabas method — numba/shap-free)...")
    bias, contributions = compute_forest_contributions(model, X_test)
    print(f"Forest bias (baseline prediction with no patient info): {dict(zip(RISK_CLASSES, bias.round(3)))}")

    print("\nGenerating per-class summary plots (overall feature impact per risk level)...")
    plot_summary_per_class(contributions, feature_names)

    print("\nGenerating individual patient explanations (one example per risk class)...")
    shown_classes = set()
    for i in range(len(y_test)):
        cls = int(y_test[i])
        if cls in shown_classes:
            continue
        save_path = METRICS_DIR / f"shap_patient_example_{RISK_CLASSES[cls].lower()}.png"
        plot_individual_explanation(contributions, X_test, y_test, feature_names, i, save_path)
        print(f"  Saved: {save_path}")
        shown_classes.add(cls)
        if len(shown_classes) == len(RISK_CLASSES):
            break

    print(f"\nDone. All outputs saved to: {METRICS_DIR}")


if __name__ == "__main__":
    main()