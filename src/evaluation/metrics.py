"""Shared metric definitions.

partial_auc lives here, once, because it was previously copy-pasted (via a
regex that extracted it out of train_tabular_baseline.py's source text) into
check_gates.py, fairness_report.py, oof_fairness.py and src/monitoring/drift.py.
That worked but was fragile -- renaming the function, or a run_id embedded
right next to it, could silently break extraction. Importing from here
removes the fragility; train_tabular_baseline.py imports it too, so there is
exactly one implementation.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_curve


def partial_auc(y_true, y_score, min_tpr: float = 0.80) -> float:
    """pAUC เหนือ TPR >= min_tpr (เต็ม = 1 - min_tpr)"""
    v_gt = np.abs(np.asarray(y_true) - 1)
    v_pred = -np.asarray(y_score)
    max_fpr = abs(1 - min_tpr)
    fpr, tpr, _ = roc_curve(v_gt, v_pred)
    stop = np.searchsorted(fpr, max_fpr, "right")
    x_interp = [fpr[stop - 1], fpr[stop]]
    y_interp = [tpr[stop - 1], tpr[stop]]
    tpr = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
    fpr = np.append(fpr[:stop], max_fpr)
    return float(np.trapezoid(tpr, fpr))
