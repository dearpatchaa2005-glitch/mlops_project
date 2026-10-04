import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.serving.predictor import RUN_ID, Predictor

ref = pd.read_csv(Path("artifacts/evaluation") / RUN_ID / "test_predictions.csv")
meta = pd.read_csv("data/raw/images/metadata.csv", low_memory=False)
df = ref[["isic_id"]].merge(meta, on="isic_id", how="left")

p = Predictor()
pred = p.predict(df)
diff = np.abs(pred - ref["prediction_score"].to_numpy())
print("rows:", len(df), "| max abs diff:", float(diff.max()))

# single-row path (what the API will do)
one = p.predict(df.iloc[[0]])
print("row0 batch vs single:", float(pred[0]), float(one[0]))