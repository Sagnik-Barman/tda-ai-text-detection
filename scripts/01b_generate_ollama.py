"""Generate modern-LLM answers locally with Ollama (fits a 6 GB GPU with 4-bit 7-9B models).

For each sampled HC3 question, every model answers at every temperature, with a
length target matched to the human answer (so length is not a giveaway). Output
rows share the question's group id with the human answers, which keeps splits
leak-free. Resumable: rows already written are skipped.

    ollama pull qwen2.5:7b && ollama pull llama3.1:8b && ollama pull mistral:7b && ollama pull gemma3:4b
    python scripts/01b_generate_ollama.py
    python scripts/01b_generate_ollama.py --models qwen2.5:7b --no-sweep --per-domain 10   # quick test
"""

from __future__ import annotations

import argparse
import sys
import time
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
    ap.add_argument("--models", nargs="+", default=gc["models"])
    ap.add_argument("--temperatures", nargs="+", type=float, default=gc["temperatures"])
    ap.add_argument("--per-domain", type=int, default=gc["prompts_per_domain"])
    ap.add_argument("--out", default=str(ROOT / "data" / "generated" / "ollama.jsonl"))
    ap.add_argument("--no-human", action="store_true", help="don't also write the paired human answers")
    ap.add_argument("--no-sweep", action="store_true", help="skip the temperature sweep")
    args = ap.parse_args()

    jobs = [(m, t) for m in args.models for t in args.temperatures]
    sw = gc.get("temperature_sweep")
    if sw and not args.no_sweep:
        jobs += [(sw["model"], t) for t in sw["temperatures"] if (sw["model"], t) not in jobs]

    out = Path(args.out)
    done = G.done_ids(out)
    qs = G.hc3_questions(cfg["data"]["hc3"]["domains"], args.per_domain, cfg["data"]["min_words"], cfg["seed"])
    total = len(qs) * len(jobs)
    print(f"{len(qs)} questions x {len(jobs)} (model, temperature) jobs = {total} generations "
          f"({len(done)} rows already done)")

    if not args.no_human:
        for q in qs:
            row = D.make_row(q["human"], 0, "human", q["domain"], "local", q["group"],
                             id=f"human-{D._h(q['group'], q['human'][:200])}")
            if row["id"] not in done:
                G.append_row(out, row)
                done.add(row["id"])

    t0, k = time.time(), 0
    for model, temp in jobs:
        for i, q in enumerate(qs):
            rid = f"ollama-{D._h(model, temp, q['group'])}"
            if rid in done:
                continue
            prompt = G.QA_PROMPT.format(n_words=G.target_words(q["human"]), question=q["question"])
            try:
                text = G.strip_thinking(G.ollama_generate(gc["ollama_url"], model, prompt, temp,
                                                          gc["max_tokens"], seed=cfg["seed"] + i))
            except Exception as e:  # keep going; a rerun fills gaps
                print(f"  ! {model} T={temp}: {e}")
                continue
            text = D.clean_text(text)
            if D.n_words(text) < 20:
                continue
            G.append_row(out, D.make_row(text, 1, model, q["domain"], "local", q["group"],
                                         temperature=temp, id=rid))
            k += 1
            if k % 25 == 0:
                rate = k / (time.time() - t0)
                print(f"  {k} new rows  ({rate * 60:.1f}/min)  current: {model} T={temp}")
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
