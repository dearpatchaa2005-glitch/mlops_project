import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.evaluation.metrics import partial_auc

meta = pd.read_csv("data/raw/images/metadata.csv", low_memory=False)
sp = pd.read_csv("data/splits/splits.csv", usecols=["isic_id", "malignant", "split", "fold"])
df = meta.merge(sp, on="isic_id")
df = df[df["split"] == "trainval"]

cols = {"tbp_lv_nevi_confidence": -1, "clin_size_long_diam_mm": 1,
        "tbp_lv_areaMM2": 1, "tbp_lv_norm_color": 1}
for c, sign in cols.items():
    vals = []
    for f in range(5):
        d = df[df["fold"] == f]
        s = sign * d[c].fillna(df[c].median())
        vals.append(partial_auc(d["malignant"], s))
    print(f"{c:28s} pAUC mean={np.mean(vals):.4f}  folds={[round(v, 4) for v in vals]}")