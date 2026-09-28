"""Tiny token helpers (kept torch-free so worker processes start fast)."""

from __future__ import annotations

import string

_PUNCT = set(string.punctuation) | {"``", "''", "...", "--", "’", "“", "”"}


def clean_token(tok: str) -> str:
    """Strip BPE / WordPiece markers (RoBERTa 'Ġ', SentencePiece '▁', BERT '##')."""
    return tok.replace("Ġ", "").replace("▁", "").replace("##", "").strip()


def is_punct(tok: str) -> bool:
    t = clean_token(tok)
    return bool(t) and all(c in _PUNCT for c in t)
