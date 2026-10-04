import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.evaluation.metrics import partial_auc
from src.evaluation.run_context import RUN_ID
from src.preprocessing.features import (
    CAT_COLS,
    fit_categorical_dtypes,
    get_feature_columns,
)

N_BOOT = 1000
SEED = 42

meta = pd.read_csv("data/raw/images/metadata.csv", low_memory=False)
sp = pd.read_csv("data/splits/splits.csv", usecols=["isic_id", "malignant", "split", "fold"])
df = meta.merge(sp, on="isic_id")
df = fit_categorical_dtypes(df, CAT_COLS)
features = get_feature_columns(df)
tv = df[df["split"] == "trainval"].reset_index(drop=True)

# out-of-fold predictions
tv["oof"] = np.nan
mdir = Path("artifacts/models") / RUN_ID
for f in range(5):
    m = lgb.Booster(model_file=str(mdir / f"fold_{f}.txt"))
    idx = tv["fold"] == f
    tv.loc[idx, "oof"] = m.predict(tv.loc[idx, features])

# sanity check: ต้องตรงกับ CV เดิม (0.1582)
per_fold = [partial_auc(g["malignant"], g["oof"]) for _, g in tv.groupby("fold")]
print(f"check: mean per-fold pAUC = {np.mean(per_fold):.4f} (expect 0.1582)")

out = Path("artifacts/evaluation") / RUN_ID / "oof_predictions.csv"
out.parent.mkdir(parents=True, exist_ok=True)
tv[["isic_id", "patient_id", "fold", "malignant", "oof"]].to_csv(out, index=False)
print("saved:", out)

tv["age_group"] = (pd.cut(tv["age_approx"], bins=[0, 50, 65, 90],
                          labels=["<=50", "51-65", ">65"])
                   .astype(object).where(tv["age_approx"].notna(), "unknown")
                   .astype(str))
tv["sex"] = tv["sex"].astype(object).where(tv["sex"].notna(), "unknown").astype(str)
tv["tbp_tile_type"] = tv["tbp_tile_type"].astype(str)

# bootstrap ระดับผู้ป่วย
rng = np.random.default_rng(SEED)
pids = tv["patient_id"].unique()
by_pat = {p: g for p, g in tv.groupby("patient_id")}


def boot_ci(col):
    for name in sorted(tv[col].astype(str).unique()):
        sub = tv[tv[col].astype(str) == name]
        if sub["malignant"].sum() < 10:
            print(f"  {col}={name}: too few malignant, skipped")
            continue
        point = partial_auc(sub["malignant"], sub["oof"])
        sub_p = {p: g for p, g in sub.groupby("patient_id")}
        keys = np.array(list(sub_p))
        vals = []
        for _ in range(N_BOOT):
            pick = rng.choice(keys, size=len(keys), replace=True)
            b = pd.concat([sub_p[p] for p in pick])
            if b["malignant"].nunique() < 2:
                continue
            vals.append(partial_auc(b["malignant"], b["oof"]))
        lo, hi = np.percentile(vals, [2.5, 97.5])
        n_pos = int(sub["malignant"].sum())
        print(f"  {col}={name:8s} malignant={n_pos:4d}  pAUC={point:.4f}  "
              f"95% CI {lo:.4f}-{hi:.4f}")


print("\nGroup-wise OOF pAUC (patient-level bootstrap):")
boot_ci("sex")
boot_ci("age_group")
boot_ci("tbp_tile_type")