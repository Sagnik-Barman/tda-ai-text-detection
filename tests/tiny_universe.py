"""A tiny offline 'universe' for integration tests: synthetic texts, a BPE tokenizer
trained on them, and small randomly-initialised RoBERTa / GPT-2 models saved to disk.

Lets the full pipeline (build dataset -> extract features -> grid -> report ->
fine-tune) run in CI or on a machine with no internet, in about a minute.
"Human" texts use a broad Zipf vocabulary; "AI" texts a steeper, more repetitive
one -- so the topology genuinely differs and detectors have something to find.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

WORDS = [f"w{i}" for i in range(1500)]


def synth_text(rng, zipf_a: float, vocab: int, n_words: int) -> str:
    ranks = np.minimum(rng.zipf(zipf_a, size=n_words), vocab) - 1
    words = [WORDS[r] for r in ranks]
    out, k = [], 0
    while k < len(words):
        L = int(rng.integers(6, 18))
        out.append(" ".join(words[k:k + L]).capitalize() + ".")
        k += L
    return " ".join(out)


def make_rows(n_groups: int = 60, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    gens = {"gen_a": (1.9, 400), "gen_b": (1.7, 700)}
    for g in range(n_groups):
        grp, dom = f"g{g}", ["news", "qa"][g % 2]
        n = int(rng.integers(90, 160))
        rows.append(dict(id=f"h{g}", text=synth_text(rng, 1.3, 1500, n), label=0, generator="human",
                         domain=dom, attack="none", temperature=None, group=grp, source="local"))
        for gen, (a, v) in gens.items():
            for t in (0.7, 1.2):
                a_eff = a - 0.25 * (t - 0.7)  # hotter sampling -> flatter distribution
                txt = synth_text(rng, a_eff, v, n)
                rows.append(dict(id=f"{gen}-{t}-{g}", text=txt, label=1, generator=gen, domain=dom,
                                 attack="none", temperature=t, group=grp, source="local"))
                if t == 0.7 and g % 2 == 0:  # "paraphrase": shuffle sentences + swap in rare words
                    w = txt.split()
                    idx = rng.choice(len(w), size=len(w) // 5, replace=False)
                    for i in idx:
                        w[i] = WORDS[int(rng.integers(0, 1500))]
                    rows.append(dict(id=f"para-{gen}-{g}", text=" ".join(w), label=1, generator=gen,
                                     domain=dom, attack="paraphrase", temperature=t, group=grp, source="local"))
    for g in range(15):  # held-out "frontier" model
        rows.append(dict(id=f"fr{g}", text=synth_text(rng, 1.8, 500, 120), label=1, generator="frontier_x",
                         domain="qa", attack="none", temperature=None, group=f"fr{g}", source="frontier"))
    return rows


def build(root: Path) -> dict:
    """Write tokenizer + models under ``root``; return feature-config overrides."""
    import torch
    from tokenizers import ByteLevelBPETokenizer
    from tokenizers.processors import RobertaProcessing
    from transformers import (GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast, RobertaConfig,
                              RobertaForSequenceClassification, RobertaModel)

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    corpus = [synth_text(rng, 1.3, 1500, 200) for _ in range(300)]
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(corpus, vocab_size=1200, min_frequency=1,
                            special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
    tk = bpe._tokenizer
    tk.post_processor = RobertaProcessing(("</s>", tk.token_to_id("</s>")), ("<s>", tk.token_to_id("<s>")))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<s>", eos_token="</s>", unk_token="<unk>",
                                  pad_token="<pad>", mask_token="<mask>")
    V = len(tok)
    assert V > 500, "tokenizer failed to load its vocabulary"

    torch.manual_seed(0)
    enc_cfg = RobertaConfig(vocab_size=V, hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                            intermediate_size=128, max_position_embeddings=600, pad_token_id=tok.pad_token_id,
                            bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id)
    for name, cls in [("encoder", RobertaModel), ("clf", RobertaForSequenceClassification)]:
        d = root / name
        cls(enc_cfg).save_pretrained(d)
        tok.save_pretrained(d)
    for name in ("lm", "lm_instruct"):
        d = root / name
        GPT2LMHeadModel(GPT2Config(vocab_size=V, n_positions=600, n_embd=64, n_layer=2, n_head=4,
                                   bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id)).save_pretrained(d)
        tok.save_pretrained(d)
    return dict(encoder=str(root / "encoder"), scorer=str(root / "lm"), bino_observer=str(root / "lm"),
                bino_performer=str(root / "lm_instruct"), clf=str(root / "clf"))


def write_jsonl(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
