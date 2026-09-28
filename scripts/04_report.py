"""Figures for the write-up (results/<dataset>/figures/):

  auroc_heatmap.png     detector x split AUROC
  robustness_gap.png    in-domain vs mean out-of-distribution AUROC per detector
  phd_by_generator.png  PHD distribution: human vs each generator (the core TDA finding)
  phd_vs_temperature.png  PHD of AI text as sampling temperature rises (if available)
  phd_by_attack.png     PHD shift under paraphrase / humanize attacks (if available)

    python scripts/04_report.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from topodetect.config import load_config, paths  # noqa: E402

HUMAN_C, AI_C = "#2a6f97", "#d1495b"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    P = paths(cfg)
    figs = P["results"] / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    res = pd.read_csv(P["results"] / "grid.csv")

    # 1) heatmap
    piv = res.pivot_table(index="detector", columns="split", values="auroc")
    cols = ["in_domain"] + sorted(c for c in piv.columns if c != "in_domain")
    piv = piv[[c for c in cols if c in piv.columns]]
    piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(1.0 + 0.75 * piv.shape[1], 0.5 + 0.42 * piv.shape[0]))
    im = ax.imshow(piv.values, cmap="viridis", vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(piv.shape[1]), piv.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(piv.shape[0]), piv.index, fontsize=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v < 0.8 else "black")
    fig.colorbar(im, ax=ax, label="AUROC")
    ax.set_title("AUROC by detector and split")
    fig.tight_layout()
    fig.savefig(figs / "auroc_heatmap.png", dpi=150)
    plt.close(fig)

    # 2) robustness gap
    ind = res[res["family"] == "in_domain"].set_index("detector")["auroc"]
    ood = res[res["family"] != "in_domain"].groupby("detector")["auroc"].mean()
    both = pd.DataFrame({"in-domain": ind, "mean OOD": ood}).dropna().sort_values("mean OOD")
    if not both.empty:
        fig, ax = plt.subplots(figsize=(7, 0.5 + 0.4 * len(both)))
        yy = np.arange(len(both))
        ax.hlines(yy, both["mean OOD"], both["in-domain"], color="#999")
        ax.scatter(both["in-domain"], yy, color="#333", label="in-domain", zorder=3)
        ax.scatter(both["mean OOD"], yy, color=AI_C, label="mean out-of-distribution", zorder=3)
        ax.set_yticks(yy, both.index, fontsize=8)
        ax.set_xlabel("AUROC")
        ax.set_title("In-domain vs mean out-of-distribution AUROC", fontsize=11)
        ax.legend(loc="lower left", fontsize=8)
        fig.tight_layout()
        fig.savefig(figs / "robustness_gap.png", dpi=150)
        plt.close(fig)

    # 3-5) PHD distributions (needs features)
    try:
        from importlib import import_module

        meta, X = import_module("03_run_grid").load_table(cfg)
    except Exception as e:  # features missing
        print("skipping PHD plots:", e)
        return
    if "cloud_phd" not in X:
        return
    df = meta.assign(phd=X["cloud_phd"].to_numpy()).dropna(subset=["phd"])
    clean = df[df["attack"] == "none"]

    gens = ["human"] + sorted(g for g in clean["generator"].unique() if g != "human")
    data = [clean.loc[clean["generator"] == g, "phd"].to_numpy() for g in gens]
    fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(gens)), 4))
    bp = ax.boxplot(data, showfliers=False, patch_artist=True)
    for k, b in enumerate(bp["boxes"]):
        b.set_facecolor(HUMAN_C if k == 0 else AI_C)
        b.set_alpha(0.7)
    ax.set_xticks(range(1, len(gens) + 1), gens, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("PHD of RoBERTa token cloud")
    ax.set_title("Intrinsic dimension: human vs each generator")
    fig.tight_layout()
    fig.savefig(figs / "phd_by_generator.png", dpi=150)
    plt.close(fig)

    ai_t = clean[(clean["label"] == 1) & clean["temperature"].notna()]
    if ai_t["temperature"].nunique() >= 2:
        fig, ax = plt.subplots(figsize=(6, 4))
        for g, sub in ai_t.groupby("generator"):
            m = sub.groupby("temperature")["phd"].median()
            ax.plot(m.index, m.values, "o-", label=g)
        ax.axhline(clean.loc[clean["label"] == 0, "phd"].median(), color=HUMAN_C, ls="--", label="human (median)")
        ax.set(xlabel="sampling temperature", ylabel="median PHD", title="Does high temperature make AI text look human?")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figs / "phd_vs_temperature.png", dpi=150)
        plt.close(fig)

    if (df["attack"] != "none").any():
        groups = ["human", "AI clean"] + sorted(set(df["attack"]) - {"none"})
        vals = [clean.loc[clean["label"] == 0, "phd"], clean.loc[clean["label"] == 1, "phd"]]
        vals += [df.loc[(df["attack"] == a) & (df["label"] == 1), "phd"] for a in groups[2:]]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot([v.to_numpy() for v in vals], showfliers=False)
        ax.set_xticks(range(1, len(groups) + 1), groups, rotation=20)
        ax.set(ylabel="PHD", title="Effect of adversarial rewriting on PHD")
        fig.tight_layout()
        fig.savefig(figs / "phd_by_attack.png", dpi=150)
        plt.close(fig)
    print(f"figures -> {figs}")


if __name__ == "__main__":
    main()
