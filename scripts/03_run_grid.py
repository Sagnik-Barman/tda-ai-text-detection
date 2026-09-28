"""Run every detector on every split and write the robustness grid.

Outputs (under results/<dataset>/):
  grid.csv       one row per (split, detector): AUROC, TPR@1%FPR, accuracy, F1, n
  summary.md     AUROC table + robustness gap (in-domain minus mean out-of-distribution)
and models/demo_model.joblib -- the hybrid detector trained on the in-domain split,
used by the Gradio demo.

    python scripts/03_run_grid.py
    python scripts/03_run_grid.py --families in_domain loo_gen   # subset
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from topodetect import data as D  # noqa: E402
from topodetect.config import load_config, paths  # noqa: E402
from topodetect.detectors import Trained, available_detectors  # noqa: E402
from topodetect.featstore import load_features  # noqa: E402
from topodetect.metrics import evaluate  # noqa: E402
from topodetect.splits import make_splits  # noqa: E402


def load_table(cfg, features_dir=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    P = paths(cfg)
    meta = pd.DataFrame(D.read_jsonl(P["dataset"]))
    feats = load_features(features_dir or P["features"])
    meta = meta[meta["id"].isin(feats.index)].reset_index(drop=True)
    X = feats.loc[meta["id"]].reset_index(drop=True)
    return meta, X


def to_md(df: pd.DataFrame) -> str:
    """Markdown table without the optional 'tabulate' dependency."""
    df = df.round(3)
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        cells = ["" if (isinstance(v, float) and np.isnan(v)) else f"{v}" for v in row.values]
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines)


def summarize(res: pd.DataFrame) -> str:
    res = res.copy()
    piv = res.pivot_table(index="detector", columns="split", values="auroc")
    cols = ["in_domain"] + sorted(c for c in piv.columns if c != "in_domain")
    piv = piv[[c for c in cols if c in piv.columns]]
    fam = res[res["family"] != "in_domain"].groupby(["detector", "family"])["auroc"].mean().unstack()
    rob = pd.DataFrame({"in_domain": piv.get("in_domain")})
    if not fam.empty:
        rob = rob.join(fam)
        rob["mean_OOD"] = fam.mean(axis=1)
        rob["gap"] = rob["in_domain"] - rob["mean_OOD"]
    rob = rob.sort_values("mean_OOD" if "mean_OOD" in rob else "in_domain", ascending=False)
    tpr = res[res["split"] == "in_domain"].set_index("detector")["tpr_at_1fpr"]
    rob["TPR@1%FPR (in-domain)"] = tpr
    return ("## Robustness summary (AUROC)\n\n" + to_md(rob) +
            "\n\n## AUROC by split\n\n" + to_md(piv) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--features-dir")
    ap.add_argument("--families", nargs="+")
    ap.add_argument("--detectors", nargs="+", help="substring filter on detector names")
    args = ap.parse_args()
    cfg = load_config(args.config)
    P = paths(cfg)
    meta, X = load_table(cfg, args.features_dir)
    y = meta["label"].to_numpy()
    print(f"{len(meta)} texts with features, {X.shape[1]} feature columns")

    sc = cfg["splits"]
    kw = dict(test_size=sc["test_size"], seed=cfg["seed"], min_per_class=sc["min_per_class"],
              min_per_generator=sc["min_per_generator"], reference_temps=tuple(sc["reference_temps"]),
              heldout_sources=tuple(sc["heldout_sources"]),
              attacker_generators=(cfg["generation"]["paraphrase_model"],))
    if args.families:
        kw["families"] = tuple(args.families)
    splits = make_splits(meta, **kw)
    dets = available_detectors(list(X.columns), cfg["seed"], X)
    if args.detectors:
        dets = {k: v for k, v in dets.items() if any(s.lower() in k.lower() for s in args.detectors)}
    print(f"{len(splits)} splits x {len(dets)} detectors")

    rows = []
    for sp in splits:
        Xtr, Xte, ytr, yte = X.iloc[sp.train], X.iloc[sp.test], y[sp.train], y[sp.test]
        for name, proto in dets.items():
            t0 = time.time()
            det = copy.deepcopy(proto).fit(Xtr, ytr)
            m = evaluate(yte, det.score(Xte), det.threshold)
            rows.append(dict(split=sp.name, family=sp.family, detector=name, **m,
                             n_train=len(sp.train), seconds=round(time.time() - t0, 1)))
            print(f"  {sp.name:<28} {name:<32} AUROC={m['auroc']:.3f}  TPR@1%={m['tpr_at_1fpr']:.3f}")
            if sp.name == "in_domain" and name.startswith("Hybrid"):
                import joblib

                P["models"].mkdir(parents=True, exist_ok=True)
                joblib.dump(det, P["models"] / "demo_model.joblib")

    res = pd.DataFrame(rows)
    P["results"].mkdir(parents=True, exist_ok=True)
    extra = P["results"] / "roberta.csv"
    if extra.exists():
        res = pd.concat([res, pd.read_csv(extra)], ignore_index=True)
    res.to_csv(P["results"] / "grid.csv", index=False)
    (P["results"] / "summary.md").write_text(summarize(res), encoding="utf-8")
    print(f"\n-> {P['results'] / 'grid.csv'}\n-> {P['results'] / 'summary.md'}\n")
    print(summarize(res))


if __name__ == "__main__":
    main()
