"""
Splits the raw IQ-OTH/NCCD-style dataset (one folder per class) into a
stratified train/val/test layout under data/processed/.

Usage:
    python -m src.data.prepare_dataset

Expects raw class folders inside data/raw/. The IQ-OTH/NCCD dataset as
distributed on Kaggle uses inconsistent naming across releases (e.g. the
"benign" folder is spelled "Bengin", and some releases append "cases" to
every folder name), so this script auto-detects the raw folder for each
canonical class instead of assuming an exact name:

    canonical "Normal"    matches: Normal, normal, Normal cases, ...
    canonical "Benign"    matches: Benign, Bengin, benign cases, ...
    canonical "Malignant" matches: Malignant, malignant cases, ...

Produces (canonical names, regardless of raw folder naming):
    data/processed/train/<Normal|Benign|Malignant>/*.jpg
    data/processed/val/<Normal|Benign|Malignant>/*.jpg
    data/processed/test/<Normal|Benign|Malignant>/*.jpg

Files are copied, not moved, so data/raw/ is left untouched and you can
re-run this safely (it wipes and rebuilds data/processed/ each time).
"""

import shutil
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    CLASS_NAMES,
    TRAIN_SPLIT,
    VAL_SPLIT,
    TEST_SPLIT,
    RANDOM_SEED,
)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Known raw-folder spellings seen across IQ-OTH/NCCD releases, keyed by the
# canonical class name our pipeline uses everywhere downstream.
RAW_FOLDER_ALIASES = {
    "Normal": ["normal", "normal cases", "normal case"],
    "Benign": ["benign", "bengin", "benign cases", "bengin cases", "benign case"],
    "Malignant": ["malignant", "malignant cases", "malignant case"],
}


def find_raw_class_dir(canonical_class: str) -> Path | None:
    """
    Finds the actual folder in data/raw/ for a canonical class name,
    matching case-insensitively against known aliases. Returns None if no
    matching folder exists.
    """
    if not RAW_DATA_DIR.exists():
        return None

    aliases = set(RAW_FOLDER_ALIASES.get(canonical_class, [canonical_class.lower()]))
    for entry in RAW_DATA_DIR.iterdir():
        if entry.is_dir() and entry.name.strip().lower() in aliases:
            return entry
    return None


def collect_class_files(class_dir: Path) -> list[Path]:
    return sorted(
        p for p in class_dir.iterdir() if p.suffix.lower() in VALID_EXTENSIONS
    )


def main():
    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {RAW_DATA_DIR}\n"
            "Download the IQ-OTH/NCCD dataset and place its class folders "
            "inside data/raw/."
        )

    assert abs(TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT - 1.0) < 1e-6, (
        "TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT must sum to 1.0"
    )

    # Resolve each canonical class to its actual raw folder before touching
    # anything, so we fail fast with a clear message if a class is missing
    # rather than partway through copying files.
    resolved_dirs: dict[str, Path] = {}
    for cls in CLASS_NAMES:
        found = find_raw_class_dir(cls)
        if found is None:
            print(
                f"[WARN] Could not find a raw folder for class '{cls}' "
                f"(looked for: {RAW_FOLDER_ALIASES.get(cls, [cls])}) — skipping."
            )
            continue
        resolved_dirs[cls] = found
        if found.name != cls:
            print(f"[INFO] Mapping raw folder '{found.name}' -> canonical class '{cls}'")

    if not resolved_dirs:
        raise FileNotFoundError(
            f"No class folders found under {RAW_DATA_DIR}. "
            "Check that the dataset was extracted correctly."
        )

    # Wipe and rebuild processed dir so re-runs don't mix stale splits
    if PROCESSED_DATA_DIR.exists():
        shutil.rmtree(PROCESSED_DATA_DIR)
    for split in ["train", "val", "test"]:
        for cls in CLASS_NAMES:
            (PROCESSED_DATA_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    summary = {}
    for cls, class_dir in resolved_dirs.items():
        files = collect_class_files(class_dir)
        if len(files) == 0:
            print(f"[WARN] No images found for class '{cls}' in {class_dir}")
            continue

        train_files, temp_files = train_test_split(
            files, train_size=TRAIN_SPLIT, random_state=RANDOM_SEED
        )
        relative_val_size = VAL_SPLIT / (VAL_SPLIT + TEST_SPLIT)
        val_files, test_files = train_test_split(
            temp_files, train_size=relative_val_size, random_state=RANDOM_SEED
        )

        for split_name, split_files in [
            ("train", train_files),
            ("val", val_files),
            ("test", test_files),
        ]:
            dest_dir = PROCESSED_DATA_DIR / split_name / cls
            for src_path in split_files:
                shutil.copy2(src_path, dest_dir / src_path.name)

        summary[cls] = {
            "total": len(files),
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files),
        }

    print("\nDataset split summary")
    print("-" * 60)
    print(f"{'Class':<12}{'Total':>10}{'Train':>10}{'Val':>10}{'Test':>10}")
    for cls, counts in summary.items():
        print(
            f"{cls:<12}{counts['total']:>10}{counts['train']:>10}"
            f"{counts['val']:>10}{counts['test']:>10}"
        )
    print("-" * 60)
    print(f"Processed data written to: {PROCESSED_DATA_DIR}")


if __name__ == "__main__":
    main()