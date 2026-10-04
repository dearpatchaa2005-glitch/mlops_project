"""Generate synthetic "live" prediction logs to demonstrate
src/monitoring/drift.py actually catching drift -- one unshifted sample
(should NOT flag) and one deliberately shifted sample (SHOULD flag), plus a
concept-drift demo that flips labels to crater pAUC on purpose.

This mirrors a real scenario: tbp_tile_type is the imaging device/tile
setting -- a clinic switching imaging hardware would show up exactly as a
shift in that column's proportions (data drift). Flipping labels simulates
what "the model is wrong in a new, systematic way" looks like for the
concept-drift check.

    python scripts/simulate_drift.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

META_PATH = Path("data/raw/images/metadata.csv")
# The label lives in a separate ground-truth file, not in metadata.csv
# (same join src/data/make_splits.py does).
GT_PATH = Path("data/raw/metadata/ISIC_2024_Training_GroundTruth.csv")
OUT_DIR = Path("runtime/logs")


def main():
    if not META_PATH.exists():
        print(f"missing {META_PATH}; run this from the project root with metadata available")
        sys.exit(1)

    meta = pd.read_csv(META_PATH, low_memory=False)
    rng = np.random.default_rng(0)

    sample = meta.sample(n=min(500, len(meta)), random_state=0).copy()
    sample["score"] = rng.uniform(0, 0.05, size=len(sample))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    unshifted_path = OUT_DIR / "predictions_unshifted.jsonl"
    with unshifted_path.open("w", encoding="utf-8") as f:
        for _, row in sample.iterrows():
            f.write(json.dumps(row.to_dict(), default=str) + "\n")
    print("wrote", unshifted_path, "(should NOT flag data drift)")

    shifted = sample.copy()
    # simulate a hardware/clinic change: push age up and heavily skew the
    # imaging-tile categorical toward one value.
    shifted["age_approx"] = shifted["age_approx"] + 25
    shifted["tbp_tile_type"] = "3D: white"
    shifted_path = OUT_DIR / "predictions_shifted.jsonl"
    with shifted_path.open("w", encoding="utf-8") as f:
        for _, row in shifted.iterrows():
            f.write(json.dumps(row.to_dict(), default=str) + "\n")
    print("wrote", shifted_path, "(SHOULD flag data drift: age_approx PSI + tile-type shift)")

    # --- concept drift demo: needs labels, which live in the ground-truth file
    if "malignant" not in meta.columns and GT_PATH.exists():
        gt = pd.read_csv(GT_PATH)
        meta = meta.merge(gt, on="isic_id", how="inner")
        meta["malignant"] = meta["malignant"].astype(int)

    if "malignant" not in meta.columns:
        print(f"skipped concept-drift files: no 'malignant' column and {GT_PATH} not found")
        return

    # Plain random sampling almost never contains a malignant case
    # (prevalence ~0.1%), which breaks pAUC -- stratify so "recent" always
    # has enough malignant cases for a meaningful comparison.
    mal_pool = meta[meta["malignant"] == 1]
    ben_pool = meta[meta["malignant"] == 0]
    n_mal = min(60, len(mal_pool))
    recent = pd.concat([
        mal_pool.sample(n_mal, random_state=1),
        ben_pool.sample(400 - n_mal, random_state=1),
    ]).sample(frac=1, random_state=1).reset_index(drop=True)

    flipped = recent.copy()
    flip_mask = rng.random(len(flipped)) < 0.4
    flipped.loc[flip_mask, "malignant"] = 1 - flipped.loc[flip_mask, "malignant"]
    flipped_path = OUT_DIR / "recent_outcomes_flipped.csv"
    flipped.to_csv(flipped_path, index=False)
    print("wrote", flipped_path, "(SHOULD flag concept drift: ~40% of labels flipped)")

    unflipped_path = OUT_DIR / "recent_outcomes_unflipped.csv"
    recent.to_csv(unflipped_path, index=False)
    print("wrote", unflipped_path, f"(should NOT flag concept drift; {n_mal} malignant of {len(recent)})")


if __name__ == "__main__":
    main()