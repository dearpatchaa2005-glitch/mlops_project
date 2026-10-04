import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from src.evaluation.metrics import partial_auc


def test_partial_auc_perfect_scores_hits_theoretical_max():
    """Max possible pAUC above TPR>=0.80 is (1 - 0.80) = 0.20 -- the same
    sanity check documented in docs/experiment_log.md ("perfect = 0.20")."""
    rng = np.random.default_rng(0)
    n, pos = 2000, 50
    y = np.zeros(n, dtype=int)
    y[:pos] = 1
    scores = y + rng.random(n) * 0.001  # near-perfect separation
    assert partial_auc(y, scores) > 0.195


def test_partial_auc_random_scores_near_theoretical_random_value():
    """Random scores -> pAUC near 0.02, per experiment_log.md."""
    rng = np.random.default_rng(1)
    n, pos = 50_000, 50
    y = np.zeros(n, dtype=int)
    y[:pos] = 1
    scores = rng.random(n)
    assert 0.0 <= partial_auc(y, scores) < 0.05
