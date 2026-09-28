"""Adversarial rewrites of AI texts (paraphrase / 'humanize') with a local Ollama model.

The rewritten rows keep label=1, their original generator and group, and get
attack="paraphrase" or "humanize". The robustness grid trains on clean text only
and tests on these, measuring how much each detector degrades under attack.

    python scripts/01c_paraphrase.py --n 600
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from topodetect import data as D  # noqa: E402
from topodetect import genutils as G  # noqa: E402
from topodetect.config import load_config  # noqa: E402


def main() -> None:
    cfg = load_config()
    gc = cfg["generation"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", default=[str(ROOT / "data" / "generated" / "ollama.jsonl")])
    ap.add_argument("--include-hc3", action="store_true", help="also rewrite HC3 ChatGPT answers")
    ap.add_argument("--attacks", nargs="+", default=["paraphrase", "humanize"], choices=list(G.PARAPHRASE_PROMPTS))
    ap.add_argument("--model", default=gc["paraphrase_model"])
    ap.add_argument("--n", type=int, default=600, help="texts to rewrite per attack")
    ap.add_argument("--out", default=str(ROOT / "data" / "generated" / "paraphrased.jsonl"))
    args = ap.parse_args()

    src = []
    for p in args.inputs:
        if Path(p).exists():
            src += [r for r in D.read_jsonl(p) if r["label"] == 1 and r["attack"] == "none"]
    if args.include_hc3:
        src += [r for r in D.load_hc3(cfg["data"]["hc3"]["domains"], 200, cfg["data"]["min_words"]) if r["label"] == 1]
    src = [r for r in src if r["generator"] != args.model]  # don't let a model rewrite itself
    random.Random(cfg["seed"]).shuffle(src)
    print(f"{len(src)} candidate AI texts; rewriting {args.n} per attack with {args.model}")

    out = Path(args.out)
    done = G.done_ids(out)
    for attack in args.attacks:
        k = 0
        for r in src:
            if k >= args.n:
                break
            rid = f"{attack}-{r['id']}"
            if rid in done:
                k += 1
                continue
            prompt = G.PARAPHRASE_PROMPTS[attack].format(text=r["text"])
            try:
                text = G.strip_thinking(G.ollama_generate(gc["ollama_url"], args.model, prompt, 0.7, gc["max_tokens"] + 100))
            except Exception as e:
                print(f"  ! {e}")
                continue
            text = D.clean_text(text)
            if D.n_words(text) < 20:
                continue
            G.append_row(out, D.make_row(text, 1, r["generator"], r["domain"], r["source"], r["group"],
                                         attack=attack, temperature=r.get("temperature"), id=rid))
            k += 1
            if k % 25 == 0:
                print(f"  {attack}: {k}/{args.n}")
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
