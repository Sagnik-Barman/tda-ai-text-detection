"""Supervised baseline: fine-tuned RoBERTa-base, evaluated on the same splits as the grid.

This is the strong in-domain baseline that topological detectors should be
compared against -- it usually wins in-domain and loses more off-distribution.
Results go to results/<dataset>/roberta.csv, which 03_run_grid.py merges in.

Fits a 6 GB GPU: fp16 autocast, max_length 256, batch 16 (~3-5 GB).

    python scripts/05_finetune_roberta.py                       # in_domain + loo_gen + attack + heldout
    python scripts/05_finetune_roberta.py --families in_domain  # quickest
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from topodetect import data as D  # noqa: E402
from topodetect.config import load_config, paths  # noqa: E402
from topodetect.metrics import evaluate, youden_threshold  # noqa: E402
from topodetect.splits import make_splits  # noqa: E402

NAME = "RoBERTa fine-tuned (supervised)"


def batches(n, bs, shuffle, rng):
    idx = rng.permutation(n) if shuffle else np.arange(n)
    for k in range(0, n, bs):
        yield idx[k:k + bs]


@torch.no_grad()
def predict(model, tok, texts, rc, device):
    model.eval()
    out = []
    for b in batches(len(texts), rc["batch_size"] * 2, False, None):
        enc = tok([texts[i] for i in b], truncation=True, max_length=rc["max_length"], padding=True,
                  return_tensors="pt").to(device)
        with torch.autocast(device_type=device.split(":")[0], enabled=device.startswith("cuda")):
            logits = model(**enc).logits.float()
        out.append(torch.softmax(logits, -1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def train_eval(texts, y, sp, rc, device, seed):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tok = AutoTokenizer.from_pretrained(rc["model"])
    model = AutoModelForSequenceClassification.from_pretrained(rc["model"], num_labels=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=rc["lr"], weight_decay=0.01)
    steps = rc["epochs"] * int(np.ceil(len(sp.train) / rc["batch_size"]))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=rc["lr"], total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=device.startswith("cuda"))
    tr_texts, tr_y = [texts[i] for i in sp.train], y[sp.train]
    for ep in range(rc["epochs"]):
        model.train()
        for b in batches(len(tr_texts), rc["batch_size"], True, rng):
            enc = tok([tr_texts[i] for i in b], truncation=True, max_length=rc["max_length"], padding=True,
                      return_tensors="pt").to(device)
            labels = torch.tensor(tr_y[b], device=device)
            with torch.autocast(device_type=device.split(":")[0], enabled=device.startswith("cuda")):
                loss = model(**enc, labels=labels).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
    s_tr = predict(model, tok, tr_texts, rc, device)
    thr = youden_threshold(tr_y, s_tr)
    s_te = predict(model, tok, [texts[i] for i in sp.test], rc, device)
    del model
    torch.cuda.empty_cache()
    return evaluate(y[sp.test], s_te, thr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--families", nargs="+", default=["in_domain", "loo_gen", "attack", "heldout"])
    args = ap.parse_args()
    cfg = load_config(args.config)
    P, rc, sc = paths(cfg), cfg["roberta"], cfg["splits"]
    meta = pd.DataFrame(D.read_jsonl(P["dataset"]))
    y, texts = meta["label"].to_numpy(), meta["text"].tolist()
    splits = make_splits(meta, test_size=sc["test_size"], seed=cfg["seed"], min_per_class=sc["min_per_class"],
                         min_per_generator=sc["min_per_generator"], reference_temps=tuple(sc["reference_temps"]),
                         heldout_sources=tuple(sc["heldout_sources"]), families=tuple(args.families),
                         attacker_generators=(cfg["generation"]["paraphrase_model"],))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for sp in splits:
        t0 = time.time()
        m = train_eval(texts, y, sp, rc, device, cfg["seed"])
        rows.append(dict(split=sp.name, family=sp.family, detector=NAME, **m, n_train=len(sp.train),
                         seconds=round(time.time() - t0, 1)))
        print(f"  {sp.name:<28} AUROC={m['auroc']:.3f}  TPR@1%={m['tpr_at_1fpr']:.3f}  ({time.time() - t0:.0f}s)")
        P["results"].mkdir(parents=True, exist_ok=True)
        out = pd.DataFrame(rows)
        f = P["results"] / "roberta.csv"
        if f.exists():  # keep results for splits not re-run this time
            old = pd.read_csv(f)
            out = pd.concat([old[~old["split"].isin(out["split"])], out], ignore_index=True)
        out.to_csv(f, index=False)  # save as we go
    print(f"-> {P['results'] / 'roberta.csv'}  (re-run 03_run_grid.py to merge)")


if __name__ == "__main__":
    main()
