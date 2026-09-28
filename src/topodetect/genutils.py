"""Shared helpers for the generation scripts (Ollama / OpenAI-compatible APIs)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import requests

from . import data as D

QA_PROMPT = (
    "Answer the following question in roughly {n_words} words, the way a knowledgeable "
    "person would answer on a Q&A forum. Write in plain paragraphs: no headings, no bullet "
    "points, no preamble.\n\nQuestion: {question}"
)

PARAPHRASE_PROMPTS = {
    "paraphrase": ("Paraphrase the following text. Keep the meaning and roughly the same length, "
                   "but change the wording and sentence structure. Output only the rewritten text."
                   "\n\n{text}"),
    "humanize": ("Rewrite the following text so it sounds like a real person typed it quickly: "
                 "vary sentence length, use a casual tone, and keep the meaning. Output only the "
                 "rewritten text.\n\n{text}"),
}


def hc3_questions(domains, per_domain: int, min_words: int, seed: int) -> list[dict]:
    """Sample HC3 questions that have at least one usable human answer."""
    from huggingface_hub import hf_hub_download

    rng = random.Random(seed)
    out = []
    for dom in domains:
        items = D.read_jsonl(hf_hub_download("Hello-SimpleAI/HC3", f"{dom}.jsonl", repo_type="dataset"))
        rng.shuffle(items)
        k = 0
        for it in items:
            hum = [D.clean_text(t) for t in it.get("human_answers") or []]
            hum = [t for t in hum if D.n_words(t) >= min_words]
            if not hum or not it.get("question"):
                continue
            out.append(dict(domain=dom, question=it["question"], human=hum[0],
                            group=f"hc3-{dom}-{D._h(it['question'])}"))
            k += 1
            if k >= per_domain:
                break
    return out


def target_words(human_text: str) -> int:
    return int(min(max(D.n_words(human_text), 80), 300))


def ollama_generate(url: str, model: str, prompt: str, temperature: float, max_tokens: int,
                    seed: int = 0, timeout: int = 600) -> str:
    r = requests.post(f"{url}/api/generate", timeout=timeout, json=dict(
        model=model, prompt=prompt, stream=False,
        options=dict(temperature=temperature, num_predict=max_tokens, seed=seed)))
    r.raise_for_status()
    return r.json().get("response", "")


def openai_chat(base_url: str, api_key: str, model: str, prompt: str, temperature: float,
                max_tokens: int, timeout: int = 300) -> str:
    """Any OpenAI-compatible endpoint (OpenAI, Gemini's /v1beta/openai, OpenRouter, Groq, ...)."""
    r = requests.post(f"{base_url.rstrip('/')}/chat/completions", timeout=timeout,
                      headers={"Authorization": f"Bearer {api_key}"},
                      json=dict(model=model, temperature=temperature, max_tokens=max_tokens,
                                messages=[{"role": "user", "content": prompt}]))
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def strip_thinking(text: str) -> str:
    """Drop <think>...</think> blocks emitted by reasoning models."""
    import re

    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(l)["id"] for l in path.open(encoding="utf-8") if l.strip()}


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
