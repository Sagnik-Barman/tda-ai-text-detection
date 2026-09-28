"""Extract all feature groups for the dataset, in resumable chunks.

Groups
  cloud  RoBERTa token-embedding cloud: PHD, MLE, TwoNN, H0/H1 summaries
  attn   RoBERTa attention maps: Betti numbers at 6 thresholds, H0 barcodes, patterns (4320 feats)
  ppl    Qwen2.5-0.5B likelihood statistics + Fast-DetectGPT score
  bino   Binoculars score (Qwen2.5-0.5B observer / -Instruct performer)

GPU work (forward passes) runs in the main process; the topology maths runs in
a pool of CPU worker processes, so the GPU and CPU overlap.

    python scripts/02_extract_features.py                     # all groups
    python scripts/02_extract_features.py --groups cloud attn  # just topology
    python scripts/02_extract_features.py --limit 200          # quick smoke run (random sample)
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from topodetect import data as D  # noqa: E402
from topodetect.config import load_config, paths  # noqa: E402
from topodetect.featstore import part_path, save_part  # noqa: E402
from topodetect.features.attention import attention_features  # noqa: E402
from topodetect.features.attention import feature_names as attn_names  # noqa: E402
from topodetect.features.cloud import cloud_features  # noqa: E402


def chunks(rows, size):
    for k in range(0, len(rows), size):
        yield k // size, rows[k:k + size]


class Bounded:
    """Submit jobs to a pool with at most ``max_inflight`` pending (caps RAM for 37 MB attention tensors)."""

    def __init__(self, pool, max_inflight):
        self.pool, self.max, self.q, self.results = pool, max_inflight, deque(), {}

    def submit(self, key, fn, *a, **kw):
        while len(self.q) >= self.max:
            k, f = self.q.popleft()
            self.results[k] = f.result()
        self.q.append((key, self.pool.submit(fn, *a, **kw)))

    def drain(self):
        while self.q:
            k, f = self.q.popleft()
            self.results[k] = f.result()
        out, self.results = self.results, {}
        return out


def encoder_groups(rows, groups, fc, root, pool):
    from topodetect.encoders import TextEncoder

    enc = TextEncoder(fc["encoder"], need_attentions="attn" in groups)
    cfg_m = enc.model.config
    a_names = attn_names(cfg_m.num_hidden_layers, cfg_m.num_attention_heads, h1=fc["attn_h1"])
    runner = Bounded(pool, max_inflight=2 * fc["workers"])
    t0, n_done = time.time(), 0
    for k, chunk in chunks(rows, fc["chunk_size"]):
        todo = [g for g in groups if not part_path(root, g, k).exists()]
        if not todo:
            continue
        ids = [r["id"] for r in chunk]
        for i, r in enumerate(chunk):
            if "cloud" in todo:
                e = enc.encode(r["text"], max_length=fc["cloud_max_tokens"])
                runner.submit(("cloud", i), cloud_features, e.hidden[-1], None, fc["cloud_h1"])
            if "attn" in todo:
                e = enc.encode(r["text"], max_length=fc["attn_max_tokens"], attentions=True)
                runner.submit(("attn", i), attention_features, e.attentions, e.tokens, h1=fc["attn_h1"])
        res = runner.drain()
        if "cloud" in todo:
            dicts = [res[("cloud", i)] for i in range(len(chunk))]
            names = list(dicts[0].keys())
            save_part(part_path(root, "cloud", k), ids, names, [[d[n] for n in names] for d in dicts])
        if "attn" in todo:
            save_part(part_path(root, "attn", k), ids, a_names, np.stack([res[("attn", i)] for i in range(len(chunk))]))
        n_done += len(chunk)
        rate = n_done / (time.time() - t0)
        print(f"  [{'+'.join(todo)}] part {k}: {n_done}/{len(rows)} texts, {rate:.2f} texts/s, "
              f"ETA {(len(rows) - n_done) / max(rate, 1e-9) / 60:.0f} min")
    del enc


def ppl_group(rows, fc, root):
    import torch

    from topodetect.encoders import CausalScorer
    from topodetect.features.perplexity import fast_detectgpt, perplexity_features

    sc = CausalScorer(fc["scorer"], load_in_4bit=fc["load_in_4bit"])
    for k, chunk in chunks(rows, fc["chunk_size"]):
        p = part_path(root, "ppl", k)
        if p.exists():
            continue
        dicts = []
        for r in chunk:
            tg, lg = sc.logits_for_ids(sc.tokenize(r["text"], 512))
            f = perplexity_features(tg, lg)
            f["zs_fast_detectgpt"] = fast_detectgpt(tg, lg)
            dicts.append(f)
        names = list(dicts[0].keys())
        save_part(p, [r["id"] for r in chunk], names, [[d[n] for n in names] for d in dicts])
        print(f"  [ppl] part {k} done")
    del sc
    torch.cuda.empty_cache()


def bino_group(rows, fc, root):
    import torch

    from topodetect.encoders import CausalScorer
    from topodetect.features.perplexity import binoculars

    obs = CausalScorer(fc["bino_observer"], load_in_4bit=fc["load_in_4bit"])
    perf = CausalScorer(fc["bino_performer"], load_in_4bit=fc["load_in_4bit"])
    for k, chunk in chunks(rows, fc["chunk_size"]):
        p = part_path(root, "bino", k)
        if p.exists():
            continue
        vals = []
        for r in chunk:
            ids = obs.tokenize(r["text"], 512)
            tg, lo = obs.logits_for_ids(ids)
            _, lp = perf.logits_for_ids(ids)
            vals.append([binoculars(tg, lo, lp)])
        save_part(p, [r["id"] for r in chunk], ["zs_binoculars"], vals)
        print(f"  [bino] part {k} done")
    del obs, perf
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--groups", nargs="+", default=["cloud", "attn", "ppl", "bino"],
                    choices=["cloud", "attn", "ppl", "bino"])
    ap.add_argument("--limit", type=int, help="only the first N rows (smoke test)")
    ap.add_argument("--threads", action="store_true", help="thread pool instead of processes")
    args = ap.parse_args()
    cfg = load_config(args.config)
    fc, P = cfg["features"], paths(cfg)
    rows = D.read_jsonl(P["dataset"])
    if args.limit:  # random sample, so a smoke run sees both classes and all domains
        import random

        rows = random.Random(cfg["seed"]).sample(rows, min(args.limit, len(rows)))
    root = P["features"] if not args.limit else P["features"].with_name(P["features"].name + f"_smoke{args.limit}")
    print(f"{len(rows)} texts -> {root}")

    enc_groups = [g for g in args.groups if g in ("cloud", "attn")]
    if enc_groups:
        Pool = ThreadPoolExecutor if args.threads else ProcessPoolExecutor
        with Pool(max_workers=fc["workers"]) as pool:
            encoder_groups(rows, enc_groups, fc, root, pool)
    if "ppl" in args.groups:
        ppl_group(rows, fc, root)
    if "bino" in args.groups:
        bino_group(rows, fc, root)
    print("done.")


if __name__ == "__main__":
    main()
