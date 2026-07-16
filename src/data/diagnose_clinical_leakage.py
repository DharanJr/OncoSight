"""
Diagnostic: checks WHY the risk models hit 100% accuracy — almost always
either (a) the dataset is synthetic and the target is a direct formula of
the features, or (b) a leaked/duplicate column. Run this once, read the
output, don't skip it — "why is my accuracy suspiciously perfect" is
exactly the kind of question a reviewer or interviewer will ask you.

Usage:
    python -m src.data.diagnose_clinical_leakage
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.data.clinical_dataset import load_raw_dataframe, clean_dataframe
from src.config import RISK_TARGET_COLUMN, RISK_CLASS_TO_IDX


def main():
    df = load_raw_dataframe()
    df = clean_dataframe(df)

    y_numeric = df[RISK_TARGET_COLUMN].map(RISK_CLASS_TO_IDX)
    numeric_df = df.select_dtypes(include=[np.number]).copy()

    print("=" * 60)
    print("1. Correlation of each numeric feature with the target")
    print("=" * 60)
    correlations = numeric_df.corrwith(y_numeric).abs().sort_values(ascending=False)
    print(correlations.to_string())
    print(
        "\nAny value above ~0.9 here is suspicious — a real symptom rarely "
        "correlates that strongly, alone, with a diagnosis."
    )

    print("\n" + "=" * 60)
    print("2. Does the sum of all numeric columns perfectly predict risk?")
    print("=" * 60)
    row_sum = numeric_df.sum(axis=1)
    sum_corr = np.corrcoef(row_sum, y_numeric)[0, 1]
    print(f"Correlation of (row sum of all features) with target: {sum_corr:.4f}")
    if abs(sum_corr) > 0.95:
        print(
            "This is very likely it — the target looks like it was generated "
            "as a threshold/bucket of a weighted sum of the other columns, "
            "meaning this dataset is synthetic, not real patient data."
        )

    print("\n" + "=" * 60)
    print("3. Duplicate rows check")
    print("=" * 60)
    dup_count = df.duplicated().sum()
    print(f"Exact duplicate rows: {dup_count}")

    print("\n" + "=" * 60)
    print("4. Class separation check — do risk levels occupy non-overlapping ranges?")
    print("=" * 60)
    if len(correlations) > 0:
        top_feature = correlations.index[0]
        print(f"Value ranges of top-correlated feature ('{top_feature}') per class:")
        print(df.groupby(RISK_TARGET_COLUMN)[top_feature].agg(["min", "max", "mean"]))


if __name__ == "__main__":
    main()