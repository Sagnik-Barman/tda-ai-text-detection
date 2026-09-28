"""Evaluation metrics. Scores are oriented so that higher = more likely AI."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, roc_curve


def tpr_at_fpr(y: np.ndarray, s: np.ndarray, fpr_target: float = 0.01) -> float:
    """True-positive rate when at most ``fpr_target`` of human texts are flagged.

    This is the number that matters in practice: a detector that wrongly accuses
    5% of human writers is unusable even with a high AUROC.
    """
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.interp(fpr_target, fpr, tpr))


def youden_threshold(y: np.ndarray, s: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y, s)
    return float(thr[np.argmax(tpr - fpr)])


def evaluate(y: np.ndarray, s: np.ndarray, threshold: float) -> dict[str, float]:
    y, s = np.asarray(y), np.asarray(s, dtype=np.float64)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    if len(np.unique(y)) < 2:
        return dict(auroc=np.nan, tpr_at_1fpr=np.nan, acc=np.nan, f1=np.nan, n=len(y))
    pred = (s >= threshold).astype(int)
    return dict(
        auroc=float(roc_auc_score(y, s)),
        tpr_at_1fpr=tpr_at_fpr(y, s, 0.01),
        acc=float((pred == y).mean()),
        f1=float(f1_score(y, pred)),
        n=int(len(y)),
        coverage=float(ok.mean()) if len(ok) else np.nan,
    )


def bootstrap_auroc_ci(y: np.ndarray, s: np.ndarray, n_boot: int = 1000, alpha: float = 0.05,
                       seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            vals.append(roc_auc_score(y[i], s[i]))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
