"""Detectors evaluated in the grid.

Zero-shot detectors are a single feature column -- no training beyond choosing
the sign and threshold on the train split. Trained detectors are sklearn models
on one or more feature groups (selected by column prefix).

Feature-group prefixes (produced by scripts/02_extract_features.py):
  cloud_  topology / intrinsic dimension of the token-embedding cloud (RoBERTa)
  attn_   topology of the 144 attention maps (RoBERTa)
  ppl_    likelihood statistics from a small causal LM (Qwen2.5-0.5B)
  zs_     zero-shot scores: Fast-DetectGPT, Binoculars
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import youden_threshold

ZERO_SHOT = {
    "PHD (zero-shot)": "cloud_phd",
    "MLE dim (zero-shot)": "cloud_mle",
    "H1 loop count (zero-shot)": "cloud_h1_n_bars",
    "H1 total persistence (zero-shot)": "cloud_h1_total",
    "Length only (sanity baseline)": "cloud_n_tokens",
    "log-likelihood": "ppl_mean_ll",
    "log-rank": "ppl_mean_log_rank",
    "Fast-DetectGPT": "zs_fast_detectgpt",
    "Binoculars": "zs_binoculars",
}

TRAINED = {
    "Embedding topology [LR]": (("cloud_",), "logreg"),
    "Attention topology [LR]": (("attn_",), "logreg"),
    "All topology [GBM]": (("cloud_", "attn_"), "hgb"),
    "Likelihood feats [LR]": (("ppl_",), "logreg"),
    "Hybrid topo+likelihood [GBM]": (("cloud_", "attn_", "ppl_", "zs_"), "hgb"),
}


def build_model(kind: str, seed: int = 0):
    if kind == "logreg":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=0.05, max_iter=3000))
    if kind == "hgb":
        return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                                              l2_regularization=1.0, early_stopping=True,
                                              validation_fraction=0.15, random_state=seed)
    raise ValueError(kind)


def select_columns(columns: list[str], prefixes) -> list[str]:
    # length is a confounder, not a signal we want (h0_n_bars is always n_tokens - 1)
    drop = {"cloud_n_tokens", "ppl_n_tokens", "cloud_h0_n_bars"}
    return [c for c in columns if c.startswith(tuple(prefixes)) and c not in drop]


@dataclass
class ZeroShot:
    column: str
    sign: float = 1.0
    threshold: float = 0.0

    def fit(self, X, y):
        s = X[self.column].to_numpy(dtype=float)
        ok = np.isfinite(s)
        if len(np.unique(y[ok])) < 2:  # e.g. PHD undefined for every short text
            self.sign, self.threshold = 1.0, np.inf
            return self
        self.sign = 1.0 if roc_auc_score(y[ok], s[ok]) >= 0.5 else -1.0
        self.threshold = youden_threshold(y[ok], self.sign * s[ok])
        return self

    def score(self, X):
        return self.sign * X[self.column].to_numpy(dtype=float)


@dataclass
class Trained:
    prefixes: tuple
    kind: str
    seed: int = 0
    columns: list = field(default_factory=list)
    model: object = None
    threshold: float = 0.5

    def fit(self, X, y):
        self.columns = select_columns(list(X.columns), self.prefixes)
        self.model = build_model(self.kind, self.seed).fit(X[self.columns].to_numpy(np.float32), y)
        self.threshold = youden_threshold(y, self.score(X))
        return self

    def score(self, X):
        return self.model.predict_proba(X[self.columns].to_numpy(np.float32))[:, 1]


def available_detectors(columns: list[str], seed: int = 0, X=None) -> dict[str, object]:
    """Detectors whose feature columns exist (and, if ``X`` is given, are not all-NaN)."""
    if X is not None:
        columns = [c for c in columns if X[c].notna().any()]
    dets: dict[str, object] = {}
    for name, col in ZERO_SHOT.items():
        if col in columns:
            dets[name] = ZeroShot(col)
    for name, (prefixes, kind) in TRAINED.items():
        if select_columns(columns, prefixes):
            if len(prefixes) > 1 and not all(select_columns(columns, [p]) for p in prefixes if p != "zs_"):
                continue  # a hybrid needs all its core groups
            dets[name] = Trained(prefixes, kind, seed)
    return dets
