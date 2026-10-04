import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import lightgbm as lgb
import numpy as np

from src.evaluation.run_context import RUN_ID
from src.preprocessing.features import (
    CAT_COLS,
    categories_from_booster,
    prepare_for_inference,
)

MODEL_DIR = Path("artifacts/models") / RUN_ID
N_FOLDS = 5


class Predictor:
    """Loads the 5 fold models from one MLflow run and averages their
    predictions. All feature preparation goes through
    src/preprocessing/features.py -- the same module the training script
    uses -- so a request is transformed exactly like a training row was,
    by construction rather than by two scripts happening to agree.
    """

    def __init__(self, model_dir=MODEL_DIR, n_folds: int = N_FOLDS):
        self.run_id = Path(model_dir).name
        self.models = [
            lgb.Booster(model_file=str(Path(model_dir) / f"fold_{i}.txt"))
            for i in range(n_folds)
        ]
        self.features = self.models[0].feature_name()
        self.categories = categories_from_booster(self.models[0], CAT_COLS)

    def prepare(self, df):
        return prepare_for_inference(df, self.features, self.categories)

    def predict(self, df):
        x = self.prepare(df)
        return np.mean([m.predict(x) for m in self.models], axis=0)
