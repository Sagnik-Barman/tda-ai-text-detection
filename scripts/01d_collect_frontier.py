"""Collect a small held-out set from current frontier models (the "2026-era LLM" test).

Two ways:

1) Any OpenAI-compatible API (OpenAI, Gemini, OpenRouter, Groq, ...):
       set FRONTIER_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
       set FRONTIER_API_KEY=...
       python scripts/01d_collect_frontier.py api --models gemini-2.5-flash --per-domain 40

2) Manually (paste answers from chat UIs into a CSV with columns
   text,generator,domain,question):
       python scripts/01d_collect_frontier.py csv data/frontier/manual.csv

Questions are sampled with a different seed from the Ollama generation so the
held-out set does not reuse training prompts. Rows get source="frontier", which
the grid holds out entirely (split "heldout:frontier").
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from topodetect import data as D  # noqa: E402
from topodetect import genutils as G  # noqa: E402
from topodetect.config import load_config  # noqa: E402

OUT = ROOT / "data" / "generated" / "frontier.jsonl"


def from_api(args, cfg) -> None:
    base, key = os.environ.get("FRONTIER_BASE_URL"), os.environ.get("FRONTIER_API_KEY")
    if not base or not key:
        sys.exit("set FRONTIER_BASE_URL and FRONTIER_API_KEY")
    qs = G.hc3_questions(cfg["data"]["hc3"]["domains"], args.per_domain, cfg["data"]["min_words"], seed=cfg["seed"] + 1000)
    done = G.done_ids(OUT)
    for model in args.models:
        for q in qs:
            rid = f"frontier-{D._h(model, q['group'])}"
            if rid in done:
                continue
            prompt = G.QA_PROMPT.format(n_words=G.target_words(q["human"]), question=q["question"])
            try:
                text = G.strip_thinking(G.openai_chat(base, key, model, prompt, args.temperature, 600))
            except Exception as e:
                print(f"  ! {model}: {e}")
                continue
            G.append_row(OUT, D.make_row(D.clean_text(text), 1, model, q["domain"], "frontier", q["group"],
                                         temperature=args.temperature, id=rid))
        print(f"{model}: done")


def from_csv(args) -> None:
    n = 0
    with open(args.path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            q = r.get("question", "")
            grp = f"hc3-{r['domain']}-{D._h(q)}" if q else f"frontier-{D._h(r['text'][:200])}"
            G.append_row(OUT, D.make_row(D.clean_text(r["text"]), 1, r["generator"], r["domain"], "frontier", grp))
            n += 1
    print(f"imported {n} rows -> {OUT}")


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("api")
    a.add_argument("--models", nargs="+", required=True)
    a.add_argument("--per-domain", type=int, default=40)
    a.add_argument("--temperature", type=float, default=0.7)
    c = sub.add_parser("csv")
    c.add_argument("path")
    args = ap.parse_args()
    from_api(args, cfg) if args.mode == "api" else from_csv(args)


if __name__ == "__main__":
    main()
