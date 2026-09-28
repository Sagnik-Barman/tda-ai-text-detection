"""Gradio demo: paste a paragraph, see its topology and an AI-likelihood score.

Shows
  * PHD / MLE intrinsic dimension of the token-embedding cloud, against the human
    and AI reference ranges measured on the training data
  * the H0 barcode (MST edge lengths) and the log-log MST scaling fit behind PHD
  * the hybrid detector's probability (if scripts/03_run_grid.py saved models/demo_model.joblib)

    pip install gradio
    python app/demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from topodetect.config import load_config, paths  # noqa: E402
from topodetect.encoders import CausalScorer, TextEncoder  # noqa: E402
from topodetect.features.attention import attention_features  # noqa: E402
from topodetect.features.attention import feature_names as attn_names  # noqa: E402
from topodetect.features.cloud import cloud_features  # noqa: E402
from topodetect.features.perplexity import binoculars, fast_detectgpt, perplexity_features  # noqa: E402
from topodetect.persistence import mst_weights, pairwise_distances  # noqa: E402
from topodetect.phd import phd  # noqa: E402

CFG = load_config()
FC = CFG["features"]
P = paths(CFG)
ENC = TextEncoder(FC["encoder"], need_attentions=True)
MODEL = None
if (P["models"] / "demo_model.joblib").exists():
    import joblib

    MODEL = joblib.load(P["models"] / "demo_model.joblib")
NEEDS_LM = MODEL is not None and any(c.startswith(("ppl_", "zs_")) for c in MODEL.columns)
SCORER = CausalScorer(FC["scorer"], load_in_4bit=FC["load_in_4bit"]) if NEEDS_LM else None
PERF = (CausalScorer(FC["bino_performer"], load_in_4bit=FC["load_in_4bit"])
        if NEEDS_LM and "zs_binoculars" in MODEL.columns else None)


def reference_ranges():
    """Median PHD of human and AI texts in the extracted training features, if present."""
    try:
        import pandas as pd

        from topodetect import data as D
        from topodetect.featstore import load_group

        cloud = load_group(P["features"], "cloud")
        meta = pd.DataFrame(D.read_jsonl(P["dataset"])).set_index("id")
        j = cloud.join(meta["label"], how="inner")
        return {lab: j.loc[j["label"] == lab, "cloud_phd"].median() for lab in (0, 1)}
    except Exception:
        return None


REF = reference_ranges()


def analyse(text: str):
    text = (text or "").strip()
    if len(text.split()) < 40:
        return "Please paste at least ~60 words -- PHD needs enough tokens to estimate a dimension.", None
    e = ENC.encode(text, max_length=FC["cloud_max_tokens"])
    X = e.hidden[-1]
    feats = cloud_features(X, h1=FC["cloud_h1"])
    d, det = phd(X, return_details=True)

    ea = ENC.encode(text, max_length=FC["attn_max_tokens"], attentions=True)
    cfg_m = ENC.model.config
    feats.update(dict(zip(attn_names(cfg_m.num_hidden_layers, cfg_m.num_attention_heads),
                          attention_features(ea.attentions, ea.tokens))))
    if SCORER is not None:
        ids = SCORER.tokenize(text, 512)
        tg, lg = SCORER.logits_for_ids(ids)
        feats.update(perplexity_features(tg, lg))
        feats["zs_fast_detectgpt"] = fast_detectgpt(tg, lg)
        if PERF is not None:
            _, lp = PERF.logits_for_ids(ids)
            feats["zs_binoculars"] = binoculars(tg, lg, lp)

    lines = [f"### Intrinsic dimension (PHD): **{d:.2f}**",
             f"MLE: {feats['cloud_mle']:.2f} · TwoNN: {feats['cloud_twonn']:.2f} · tokens: {X.shape[0]}"]
    if REF:
        lines.append(f"Reference medians on training data -- human: {REF[0]:.2f}, AI: {REF[1]:.2f} "
                     "(AI text tends to have *lower* intrinsic dimension)")
    if MODEL is not None:
        import pandas as pd

        row = pd.DataFrame([{c: feats.get(c, np.nan) for c in MODEL.columns}])
        p = float(MODEL.score(row)[0])
        verdict = "likely AI-generated" if p >= MODEL.threshold else "likely human-written"
        lines.append(f"\n### Hybrid detector: **{p:.1%}** AI probability → {verdict}")
    lines.append("\n*Detectors are probabilistic and can be wrong, especially on short, edited, "
                 "or non-native-English text. Never use a single score to accuse someone.*")

    D_ = pairwise_distances(X)
    w = np.sort(mst_weights(D_))[::-1]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].hlines(np.arange(len(w)), 0, w, lw=1, color="#2a6f97")
    ax[0].set(title="H0 barcode (MST edge lengths)", xlabel="filtration scale ε", yticks=[])
    ax[1].loglog(det["sizes"], det["lengths"], "o-", color="#d1495b")
    ax[1].set(title=f"MST scaling: slope m={det['slope']:.3f} → PHD = 1/(1-m) = {d:.2f}",
              xlabel="subsample size n", ylabel="Σ MST edge lengths")
    fig.tight_layout()
    return "\n\n".join(lines), fig


def main() -> None:
    import gradio as gr

    demo = gr.Interface(
        fn=analyse,
        inputs=gr.Textbox(lines=12, label="Paste a paragraph (60+ words)"),
        outputs=[gr.Markdown(), gr.Plot()],
        title="Topological AI-text detector",
        description="Persistent homology of RoBERTa token embeddings and attention maps.",
    )
    demo.launch()


if __name__ == "__main__":
    main()
