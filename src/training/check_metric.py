import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from sklearn.metrics import roc_auc_score

from src.evaluation.metrics import partial_auc

rng = np.random.default_rng(0)
n, pos = 200_000, 200
y = np.zeros(n, dtype=int)
y[:pos] = 1

cases = {
    "random": rng.random(n),
    "perfect": y + rng.random(n) * 0.01,
    "good (AUC~0.85)": y * 1.5 + rng.normal(size=n),
    "weak (AUC~0.65)": y * 0.55 + rng.normal(size=n),
}
for name, s in cases.items():
    print(f"{name:18s} pAUC={partial_auc(y, s):.4f}  ROC-AUC={roc_auc_score(y, s):.3f}")