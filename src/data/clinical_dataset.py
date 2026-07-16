"""
Loads and preprocesses the clinical risk dataset ("Cancer Patient Data Sets"
style — Age, Gender, symptom/lifestyle columns, target column "Level" with
values Low/Medium/High).

Handles, in order:
    1. Auto-locating the CSV (searches data/raw/ and data/raw/clinical/,
       matches by filename keywords rather than requiring an exact name)
    2. Dropping identifier columns that carry no signal
    3. Missing value handling (median for numeric, mode for categorical)
    4. Outlier handling (clipping to the 1st/99th percentile per numeric
       column — chosen over dropping rows, since this dataset is small and
       every row matters)
    5. Encoding the target (Low/Medium/High -> 0/1/2, ordinal)
    6. One-hot encoding any remaining non-numeric feature columns
    7. Feature scaling (StandardScaler) — fit on train only, applied to both
       train and test, to avoid leaking test-set statistics into training

Usage:
    python -m src.data.clinical_dataset
    (running directly prints a preprocessing summary — useful as a sanity
    check before training)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    CLINICAL_DATA_DIR,
    CLINICAL_PROCESSED_DIR,
    RISK_TARGET_COLUMN,
    RISK_CLASSES,
    RISK_CLASS_TO_IDX,
    CLINICAL_ID_COLUMNS,
    CLINICAL_TEST_SPLIT,
    RANDOM_SEED,
)

# Filename keywords used to auto-locate the CSV — matches things like
# "cancer patient data sets.csv", "cancer_patient_datasets.csv",
# "lung_cancer_data.csv", etc. without requiring an exact name.
FILENAME_KEYWORDS = ["cancer", "lung", "patient", "risk"]
SEARCH_DIRS = [CLINICAL_DATA_DIR, CLINICAL_DATA_DIR / "clinical"]


def find_clinical_csv() -> Path:
    candidates = []
    for search_dir in SEARCH_DIRS:
        if not search_dir.exists():
            continue
        for path in search_dir.glob("*.csv"):
            candidates.append(path)
        # Also catch .xlsx in case someone saves the Kaggle download as Excel
        for path in search_dir.glob("*.xlsx"):
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            f"No clinical dataset CSV/XLSX found in {SEARCH_DIRS}. "
            "Download the clinical risk dataset and place it in data/raw/ "
            "or data/raw/clinical/ — any filename is fine."
        )

    # Prefer a filename that matches known keywords; fall back to the first
    # CSV found if nothing matches (better to try than to hard-fail here).
    for path in candidates:
        if any(kw in path.stem.lower() for kw in FILENAME_KEYWORDS):
            return path

    print(
        f"[WARN] No filename matched expected keywords {FILENAME_KEYWORDS}; "
        f"using first file found: {candidates[0].name}"
    )
    return candidates[0]


def load_raw_dataframe() -> pd.DataFrame:
    csv_path = find_clinical_csv()
    print(f"[INFO] Loading clinical dataset from: {csv_path}")

    if csv_path.suffix.lower() == ".xlsx":
        df = pd.read_excel(csv_path)
    else:
        df = pd.read_csv(csv_path)

    # Normalize column names: strip whitespace (the source dataset is known
    # to have trailing spaces on some headers, e.g. "Patient Id ")
    df.columns = [c.strip() for c in df.columns]
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop identifier columns if present
    cols_to_drop = [c.strip() for c in CLINICAL_ID_COLUMNS if c.strip() in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # De-duplicate BEFORE splitting into train/test. This dataset ships with
    # a large number of exact duplicate rows (verified: ~85% of rows in the
    # "Cancer Patient Data Sets" release are copies of other rows). Without
    # this step, the same patient record ends up in both the train and test
    # split, which means test accuracy measures memorization, not
    # generalization — inflating every model to a meaningless ~100%.
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_after = len(df)
    if n_before != n_after:
        print(
            f"[INFO] Removed {n_before - n_after} duplicate rows "
            f"({n_before} -> {n_after} unique patient records). "
            "This is expected for this dataset and prevents train/test leakage."
        )

    if RISK_TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Expected target column '{RISK_TARGET_COLUMN}' not found. "
            f"Columns present: {list(df.columns)}\n"
            "If your dataset uses a different target column name, update "
            "RISK_TARGET_COLUMN in src/config.py."
        )

    # Normalize target values (dataset sometimes has trailing spaces / case
    # differences, e.g. "Low " or "low")
    df[RISK_TARGET_COLUMN] = df[RISK_TARGET_COLUMN].astype(str).str.strip().str.title()
    unknown = set(df[RISK_TARGET_COLUMN].unique()) - set(RISK_CLASSES)
    if unknown:
        raise ValueError(
            f"Unexpected target values {unknown} in '{RISK_TARGET_COLUMN}'. "
            f"Expected only: {RISK_CLASSES}"
        )

    # --- Missing values ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [
        c for c in df.select_dtypes(exclude=[np.number]).columns if c != RISK_TARGET_COLUMN
    ]

    for col in numeric_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    for col in categorical_cols:
        if df[col].isna().any():
            mode_val = df[col].mode(dropna=True)
            df[col] = df[col].fillna(mode_val[0] if len(mode_val) else "Unknown")

    # --- Outlier clipping (1st/99th percentile) ---
    for col in numeric_cols:
        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)
        df[col] = df[col].clip(lower=lower, upper=upper)

    return df


def encode_and_split(df: pd.DataFrame):
    y = df[RISK_TARGET_COLUMN].map(RISK_CLASS_TO_IDX).values
    X = df.drop(columns=[RISK_TARGET_COLUMN])

    # One-hot encode any remaining non-numeric columns (e.g. "Gender" as
    # text rather than pre-coded 1/2). Most versions of this dataset already
    # encode symptom severity as integers 1-8, so this often no-ops.
    X = pd.get_dummies(X, drop_first=True)
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y,
        test_size=CLINICAL_TEST_SPLIT,
        random_state=RANDOM_SEED,
        stratify=y,  # preserve Low/Medium/High proportions in both splits
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # fit on train only — no leakage

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler


def prepare_clinical_data():
    """
    Full pipeline: load -> clean -> encode -> split -> scale.
    Saves the fitted scaler and feature names so evaluation/inference can
    apply the exact same transform later.
    Returns (X_train, X_test, y_train, y_test, feature_names).
    """
    df = load_raw_dataframe()
    df = clean_dataframe(df)
    X_train, X_test, y_train, y_test, feature_names, scaler = encode_and_split(df)

    joblib.dump(scaler, CLINICAL_PROCESSED_DIR / "scaler.joblib")
    joblib.dump(feature_names, CLINICAL_PROCESSED_DIR / "feature_names.joblib")
    np.save(CLINICAL_PROCESSED_DIR / "X_train.npy", X_train)
    np.save(CLINICAL_PROCESSED_DIR / "X_test.npy", X_test)
    np.save(CLINICAL_PROCESSED_DIR / "y_train.npy", y_train)
    np.save(CLINICAL_PROCESSED_DIR / "y_test.npy", y_test)

    return X_train, X_test, y_train, y_test, feature_names


def main():
    X_train, X_test, y_train, y_test, feature_names = prepare_clinical_data()

    print("\nClinical dataset preprocessing summary")
    print("-" * 60)
    print(f"Features ({len(feature_names)}): {feature_names}")
    print(f"Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")
    print("\nClass distribution:")
    for split_name, y in [("Train", y_train), ("Test", y_test)]:
        counts = {RISK_CLASSES[i]: int((y == i).sum()) for i in range(len(RISK_CLASSES))}
        print(f"  {split_name}: {counts}")
    print("-" * 60)
    print(f"Processed arrays + scaler saved to: {CLINICAL_PROCESSED_DIR}")


if __name__ == "__main__":
    main()