"""Dataset loading into one unified schema.

Every sample is a dict with these keys:

    id           unique string
    text         the passage
    label        1 = AI-generated, 0 = human
    generator    "human" or model name (e.g. "chatgpt", "qwen2.5:7b", "gpt-4o")
    domain       e.g. "reddit_eli5", "news", "abstracts"
    attack       "none" or attack name (e.g. "paraphrase")
    temperature  sampling temperature if known, else None
    group        prompt/source id -- human text and the AI texts written for the
                 same prompt share a group, so splits never leak a prompt
                 across train/test
    source       dataset name ("hc3", "raid", "mage", "local", "frontier")

Supported sources:
  * HC3  (Guo et al. 2023) -- human vs ChatGPT answers, 5 domains
  * RAID (Dugan et al., ACL 2024) -- 11 generators, 8 domains, 11 adversarial attacks
  * MAGE (Li et al., ACL 2024) -- 27 generators, 10 domains
  * local JSONL files written by scripts/01b_generate_ollama.py, 01c_paraphrase.py,
    01d_collect_frontier.py (same schema)
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

HC3_DOMAINS = ("finance", "medicine", "open_qa", "reddit_eli5", "wiki_csai")
REQUIRED = ("id", "text", "label", "generator", "domain", "attack", "temperature", "group", "source")


def _h(*parts) -> str:
    return hashlib.md5("||".join(map(str, parts)).encode("utf-8")).hexdigest()[:12]


def make_row(text, label, generator, domain, source, group, attack="none", temperature=None, id=None) -> dict:
    return dict(
        id=id or _h(source, generator, domain, attack, text[:200]),
        text=text.strip(), label=int(label), generator=str(generator), domain=str(domain),
        attack=str(attack), temperature=temperature, group=str(group), source=str(source),
    )


def clean_text(t: str) -> str:
    t = re.sub(r"\s+", " ", t or "").strip()
    # HC3 artefacts: spaces before punctuation from its tokenised human answers
    return re.sub(r" ([,.;:!?)])", r"\1", t)


_PREAMBLE = re.compile(
    r"^\s*(?:(?:sure|certainly|of course|okay|ok)[!,.]?\s+)?"
    r"(?:here(?:'s| is| are)\b[^:\n]{0,120}:|below is\b[^:\n]{0,120}:)\s*", re.I)


def strip_generation_artifacts(t: str) -> str:
    """Remove meta-text a human would never write, so attacks/generations aren't trivially detectable.

    e.g. "Here's the rewritten text: ..." (paraphrase preambles), markdown bold/headers.
    Genuine stylistic openers ("Okay, let's break this down") are kept -- that's real model style.
    """
    t = _PREAMBLE.sub("", t, count=1)
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t)          # **bold** -> bold
    t = re.sub(r"(^|\s)#{1,6}\s+", r"\1", t)          # markdown headers
    t = t.strip().strip('"').strip()
    return t


def n_words(t: str) -> int:
    return len(t.split())


# ----------------------------------------------------------------------------
# JSONL helpers
# ----------------------------------------------------------------------------
def write_jsonl(rows: Iterable[dict], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def validate(rows: list[dict]) -> list[dict]:
    for r in rows:
        missing = [k for k in REQUIRED if k not in r]
        if missing:
            raise ValueError(f"row {r.get('id')} missing {missing}")
    return rows


# ----------------------------------------------------------------------------
# HC3
# ----------------------------------------------------------------------------
def load_hc3(domains=HC3_DOMAINS, max_per_domain: int | None = 1000, min_words: int = 60,
             seed: int = 0) -> list[dict]:
    """Balanced human/ChatGPT answers, grouped by question."""
    from huggingface_hub import hf_hub_download

    rng = random.Random(seed)
    rows = []
    for dom in domains:
        path = hf_hub_download("Hello-SimpleAI/HC3", f"{dom}.jsonl", repo_type="dataset")
        items = read_jsonl(path)
        rng.shuffle(items)
        hum, ai = [], []
        for it in items:
            g = f"hc3-{dom}-{_h(it.get('question', ''))}"
            for t in it.get("human_answers") or []:
                t = clean_text(t)
                if n_words(t) >= min_words:
                    hum.append(make_row(t, 0, "human", dom, "hc3", g))
            for t in it.get("chatgpt_answers") or []:
                t = clean_text(t)
                if n_words(t) >= min_words:
                    ai.append(make_row(t, 1, "chatgpt", dom, "hc3", g))
        k = min(len(hum), len(ai), max_per_domain or 10**9)
        rows += hum[:k] + ai[:k]
    return rows


# ----------------------------------------------------------------------------
# RAID
# ----------------------------------------------------------------------------
def load_raid(split: str = "train", models=None, domains=None, attacks=("none",),
              decodings=None, n_per_cell: int = 150, min_words: int = 60,
              max_scan: int = 3_000_000, seed: int = 0) -> list[dict]:
    """Stream RAID from the HF hub and keep up to ``n_per_cell`` rows per
    (model, domain, attack, decoding) cell. RAID is large (millions of rows), so
    it is streamed; ``max_scan`` bounds the pass.

    Human rows have model == "human". Attacked machine rows point to their
    un-attacked parent via ``adv_source_id``; human/AI rows share ``source_id``.
    """
    from datasets import load_dataset

    try:
        ds = load_dataset("liamdugan/raid", split=split, streaming=True)
    except Exception:
        ds = load_dataset("liamdugan/raid", "raid", split=split, streaming=True)

    need = {"model", "domain", "attack", "generation"}
    counts: dict[tuple, int] = defaultdict(int)
    rows = []
    for i, r in enumerate(ds):
        if i == 0 and not need.issubset(r.keys()):
            raise KeyError(f"RAID schema changed; got columns {sorted(r.keys())}, need {sorted(need)}")
        if i >= max_scan:
            break
        m, d, a = r["model"], r["domain"], r.get("attack", "none")
        dec = r.get("decoding") or "none"
        if models and m not in models and m != "human":
            continue
        if domains and d not in domains:
            continue
        if attacks and a not in attacks:
            continue
        if decodings and m != "human" and dec not in decodings:
            continue
        cell = (m, d, a, dec)
        if counts[cell] >= n_per_cell:
            continue
        t = clean_text(r["generation"])
        if n_words(t) < min_words:
            continue
        counts[cell] += 1
        grp = r.get("source_id") or r.get("adv_source_id") or r.get("id")
        rows.append(make_row(t, int(m != "human"), m, d, "raid", f"raid-{grp}", attack=a,
                             temperature=None, id=f"raid-{r.get('id', i)}"))
    return rows


# ----------------------------------------------------------------------------
# MAGE
# ----------------------------------------------------------------------------
def _parse_mage_src(src: str) -> tuple[str, str]:
    """'xsum_human' -> ('xsum', 'human'); 'xsum_machine_continuation_gpt-3.5-turbo' -> ('xsum', 'gpt-3.5-turbo')."""
    parts = src.split("_")
    domain = parts[0]
    if "human" in parts:
        return domain, "human"
    return domain, parts[-1] if len(parts) > 1 else "unknown"


def load_mage(split: str = "test", n: int | None = 4000, min_words: int = 60, seed: int = 0) -> list[dict]:
    """MAGE; the label is derived from ``src`` (contains 'human' => human), not
    from the numeric label column, so we never depend on its 0/1 convention."""
    from datasets import load_dataset

    ds = load_dataset("yaful/MAGE", split=split)
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    rows = []
    for i in idx:
        r = ds[i]
        t = clean_text(r["text"])
        if n_words(t) < min_words:
            continue
        dom, gen = _parse_mage_src(r["src"])
        rows.append(make_row(t, int(gen != "human"), gen, dom, "mage", f"mage-{i}", id=f"mage-{split}-{i}"))
        if n and len(rows) >= n:
            break
    return rows


# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------
def dedupe(rows: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in rows:
        k = _h(r["text"][:500])
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def summarize(rows: list[dict]) -> str:
    from collections import Counter

    c = Counter((r["source"], r["generator"], r["attack"]) for r in rows)
    lines = [f"{len(rows)} rows  ({sum(r['label'] for r in rows)} AI / {sum(1 - r['label'] for r in rows)} human)"]
    lines += [f"  {s:<9} {g:<24} {a:<14} {n}" for (s, g, a), n in sorted(c.items())]
    return "\n".join(lines)
