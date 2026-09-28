"""Is a topological signal real, or a proxy for text length / domain?

For each zero-shot score, reports
  * overall AUROC (direction chosen automatically, and printed)
  * AUROC inside each token-length quintile  -> survives length control?
  * AUROC inside each domain                 -> survives domain control?
  * Spearman correlation with length
and compares with a length-only baseline. Output: results/<dataset>/confounds.md

    python scripts/06_confound_checks.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from topodetect import data as D  # noqa: E402
from topodetect.config import load_config, paths  # noqa: E402
from topodetect.featstore import load_features  # noqa: E402

SCORES = {
    "PHD": "cloud_phd", "MLE dim": "cloud_mle", "TwoNN dim": "cloud_twonn",
    "H1 loop count": "cloud_h1_n_bars", "H1 total persistence": "cloud_h1_total",
    "H0 persistence entropy": "cloud_h0_norm_entropy", "log-likelihood": "ppl_mean_ll",
    "Fast-DetectGPT": "zs_fast_detectgpt", "Binoculars": "zs_binoculars", "Length (tokens)": "cloud_n_tokens",
}


def oriented_auc(y, s):
    ok = np.isfinite(s)
    if len(np.unique(y[ok])) < 2:
        return np.nan, ""
    a = roc_auc_score(y[ok], s[ok])
    return max(a, 1 - a), ("AI higher" if a >= 0.5 else "AI lower")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    P = paths(cfg)
    meta = pd.DataFrame(D.read_jsonl(P["dataset"]))
    meta = meta[meta["attack"] == "none"]
    X = load_features(P["features"])
    df = meta.set_index("id").join(X, how="inner")
    y = df["label"].to_numpy()
    df["len_bin"] = pd.qcut(df["cloud_n_tokens"], 5, duplicates="drop")

    rows = []
    for name, col in SCORES.items():
        if col not in df:
            continue
        s = df[col].to_numpy(float)
        auc, direction = oriented_auc(y, s)
        sign = 1 if direction == "AI higher" else -1
        per_len = [roc_auc_score(g["label"], sign * g[col]) for _, g in df.groupby("len_bin", observed=True)
                   if g["label"].nunique() == 2 and g[col].notna().all()]
        per_dom = {d: roc_auc_score(g["label"], sign * g[col]) for d, g in df.groupby("domain")
                   if g["label"].nunique() == 2 and g[col].notna().all()}
        rho = pd.Series(s).corr(pd.Series(df["cloud_n_tokens"].to_numpy()), method="spearman")
        med = df.groupby("label")[col].median()
        rows.append({"score": name, "AUROC": auc, "direction": direction,
                     "median human": float(med.get(0, np.nan)), "median AI": float(med.get(1, np.nan)),
                     "min AUROC in length quintile": min(per_len) if per_len else np.nan,
                     "mean AUROC in length quintiles": np.mean(per_len) if per_len else np.nan,
                     "min AUROC in a domain": min(per_dom.values()) if per_dom else np.nan,
                     "Spearman with length": rho})
    out = pd.DataFrame(rows).set_index("score").round(3)
    txt = ("# Confound checks\n\nA signal is trustworthy if its AUROC stays high *within* length quintiles and "
           "*within* each domain, and clearly beats the length-only baseline.\n\n")
    cols = list(out.columns)
    txt += "| score | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 1) + "\n"
    for i, r in out.iterrows():
        txt += f"| {i} | " + " | ".join("" if pd.isna(v) else str(v) for v in r.values) + " |\n"
    P["results"].mkdir(parents=True, exist_ok=True)
    (P["results"] / "confounds.md").write_text(txt, encoding="utf-8")
    print(txt)
    print(f"-> {P['results'] / 'confounds.md'}")


if __name__ == "__main__":
    main()
