"""Thin wrappers around Hugging Face models.

* ``TextEncoder``   -- masked-LM encoder (RoBERTa/BERT): token embeddings + attention maps
* ``CausalScorer``  -- causal LM (Qwen/Llama/GPT-2): per-token log-probs, ranks, entropies

Both can be built from already-loaded components (``from_components``), which is
how the unit tests run them on tiny randomly-initialised models without any
download. Everything returned is numpy on CPU.

Sized for a 6 GB laptop GPU (RTX 4050): base encoders in fp16 use < 1 GB; 0.5B-1.5B
causal LMs fit in fp16; 7-8B models need ``load_in_4bit=True`` (bitsandbytes).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .text_utils import clean_token, is_punct  # noqa: F401  (re-exported)


def pick_device(device: str | None = None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _from_pretrained(cls, name: str, dtype, **kw):
    """Handle the torch_dtype -> dtype rename across transformers versions."""
    try:
        return cls.from_pretrained(name, dtype=dtype, **kw)
    except TypeError:
        return cls.from_pretrained(name, torch_dtype=dtype, **kw)


@dataclass
class Encoded:
    tokens: list[str]               # all tokens incl. special tokens
    content_mask: np.ndarray        # True for non-special tokens
    hidden: dict[int, np.ndarray]   # layer -> (n_content_tokens, hidden) float32
    attentions: np.ndarray | None   # (layers, heads, T, T) float32, incl. specials


class TextEncoder:
    """Masked-LM encoder used for PHD / embedding topology / attention topology."""

    def __init__(self, model_name: str = "roberta-base", device: str | None = None,
                 fp16: bool = True, need_attentions: bool = True):
        from transformers import AutoModel, AutoTokenizer

        self.device = pick_device(device)
        dtype = torch.float16 if (fp16 and self.device.startswith("cuda")) else torch.float32
        kw = {"attn_implementation": "eager"} if need_attentions else {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = _from_pretrained(AutoModel, model_name, dtype, **kw).to(self.device).eval()
        self.name = model_name

    @classmethod
    def from_components(cls, model, tokenizer=None, device: str = "cpu"):
        self = cls.__new__(cls)
        self.model, self.tokenizer, self.device = model.to(device).eval(), tokenizer, device
        self.name = getattr(model.config, "_name_or_path", "custom")
        return self

    @torch.no_grad()
    def encode_ids(self, input_ids: torch.Tensor, special_mask: np.ndarray | None = None,
                   layers=(-1,), attentions: bool = False, tokens: list[str] | None = None) -> Encoded:
        input_ids = input_ids.to(self.device)
        if input_ids.dim() == 1:
            input_ids = input_ids[None]
        out = self.model(input_ids=input_ids, output_hidden_states=True, output_attentions=attentions)
        T = input_ids.shape[1]
        if special_mask is None:  # assume first and last are specials (<s> ... </s>)
            special_mask = np.zeros(T, dtype=bool)
            special_mask[[0, -1]] = True
        content = ~np.asarray(special_mask, dtype=bool)
        hs = out.hidden_states
        hidden = {int(l): hs[l][0].float().cpu().numpy()[content] for l in layers}
        att = None
        if attentions:
            att = torch.stack([a[0] for a in out.attentions]).float().cpu().numpy()
        if tokens is None:
            tokens = [str(i) for i in input_ids[0].tolist()]
        return Encoded(tokens=tokens, content_mask=content, hidden=hidden, attentions=att)

    def encode(self, text: str, max_length: int = 512, layers=(-1,), attentions: bool = False) -> Encoded:
        enc = self.tokenizer(text, truncation=True, max_length=max_length,
                             return_tensors="pt", return_special_tokens_mask=True)
        ids = enc["input_ids"]
        smask = enc["special_tokens_mask"][0].numpy().astype(bool)
        toks = self.tokenizer.convert_ids_to_tokens(ids[0].tolist())
        return self.encode_ids(ids, smask, layers=layers, attentions=attentions, tokens=toks)


class CausalScorer:
    """Causal LM giving per-token statistics for perplexity-style detectors."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B", device: str | None = None,
                 fp16: bool = True, load_in_4bit: bool = False):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = pick_device(device)
        dtype = torch.float16 if (fp16 and self.device.startswith("cuda")) else torch.float32
        kw = {}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
            kw["device_map"] = {"": 0}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = _from_pretrained(AutoModelForCausalLM, model_name, dtype, **kw)
        if not load_in_4bit:
            self.model = self.model.to(self.device)
        self.model.eval()
        self.name = model_name

    @classmethod
    def from_components(cls, model, tokenizer=None, device: str = "cpu"):
        self = cls.__new__(cls)
        self.model, self.tokenizer, self.device = model.to(device).eval(), tokenizer, device
        self.name = getattr(model.config, "_name_or_path", "custom")
        return self

    def tokenize(self, text: str, max_length: int = 512) -> torch.Tensor:
        ids = self.tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")["input_ids"]
        bos = self.tokenizer.bos_token_id
        if bos is not None and ids[0, 0].item() != bos:  # give the first real token a context
            ids = torch.cat([torch.tensor([[bos]]), ids], dim=1)
        return ids

    @torch.no_grad()
    def logits_for_ids(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (targets[T-1], logits[T-1, V]) in float32 on the model device."""
        if input_ids.dim() == 1:
            input_ids = input_ids[None]
        input_ids = input_ids.to(self.model.device if hasattr(self.model, "device") else self.device)
        logits = self.model(input_ids=input_ids).logits[0, :-1].float()
        return input_ids[0, 1:], logits
