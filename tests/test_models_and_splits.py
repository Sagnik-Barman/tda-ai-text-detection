"""Model-wrapper, baseline and split tests on tiny random models (no downloads)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from topodetect.encoders import CausalScorer, TextEncoder  # noqa: E402
from topodetect.features.perplexity import binoculars, fast_detectgpt, perplexity_features  # noqa: E402
from topodetect.splits import make_splits  # noqa: E402
from tiny_universe import make_rows  # noqa: E402


@pytest.fixture(scope="module")
def tiny_models():
    from transformers import GPT2Config, GPT2LMHeadModel, RobertaConfig, RobertaModel

    torch.manual_seed(0)
    enc = RobertaModel(RobertaConfig(vocab_size=300, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                                     intermediate_size=64, max_position_embeddings=300,
                                     attn_implementation="eager"))
    lm = GPT2LMHeadModel(GPT2Config(vocab_size=300, n_positions=300, n_embd=32, n_layer=1, n_head=2,
                                    bos_token_id=0, eos_token_id=0))
    return TextEncoder.from_components(enc), CausalScorer.from_components(lm)


def test_encoder_outputs(tiny_models):
    enc, _ = tiny_models
    ids = torch.randint(5, 300, (70,))
    e = enc.encode_ids(ids, attentions=True)
    assert e.hidden[-1].shape == (68, 32)  # specials stripped
    assert e.attentions.shape == (2, 2, 70, 70)
    assert np.allclose(e.attentions.sum(-1), 1, atol=1e-4)


def test_perplexity_and_baselines(tiny_models):
    _, sc = tiny_models
    tg, lg = sc.logits_for_ids(torch.randint(1, 300, (60,)))
    f = perplexity_features(tg, lg)
    assert f["ppl_mean_ll"] < 0 and 0 <= f["ppl_top1"] <= 1
    assert np.isfinite(fast_detectgpt(tg, lg))
    assert binoculars(tg, lg, lg) == pytest.approx(
        torch.nn.functional.cross_entropy(lg, tg).item() / (-(lg.softmax(-1) * lg.log_softmax(-1)).sum(-1).mean().item()))


def test_fast_detectgpt_detects_greedy_text(tiny_models):
    """Text sampled greedily from the scorer itself should score higher than random tokens."""
    _, sc = tiny_models
    ids = torch.tensor([[0]])
    with torch.no_grad():
        for _ in range(60):
            nxt = sc.model(input_ids=ids).logits[0, -1].argmax()
            ids = torch.cat([ids, nxt.view(1, 1)], dim=1)
    greedy = fast_detectgpt(*sc.logits_for_ids(ids[0]))
    rnd = fast_detectgpt(*sc.logits_for_ids(torch.randint(1, 300, (61,))))
    assert greedy > rnd


def test_splits_are_group_disjoint():
    df = pd.DataFrame(make_rows(40))
    splits = make_splits(df, min_per_class=5, min_per_generator=10)
    names = {s.name for s in splits}
    assert {"in_domain", "loo_gen:gen_a", "loo_gen:gen_b", "attack:paraphrase", "heldout:frontier"} <= names
    assert any(n.startswith("temp:") for n in names) and any(n.startswith("loo_dom:") for n in names)
    for s in splits:
        tr, te = set(df.loc[s.train, "group"]), set(df.loc[s.test, "group"])
        assert not (tr & te), s.name
        if s.name.startswith("loo_gen:"):
            g = s.name.split(":")[1]
            assert g not in set(df.loc[s.train, "generator"])
            assert set(df.loc[s.test, "generator"]) == {"human", g}
        if s.family == "attack":
            assert (df.loc[s.train, "attack"] == "none").all()


def test_attack_split_holds_out_attacker():
    df = pd.DataFrame(make_rows(40))
    sp = {s.name: s for s in make_splits(df, min_per_class=5, min_per_generator=10,
                                         attacker_generators=("gen_b",))}["attack:paraphrase"]
    assert "gen_b" not in set(df.loc[sp.train, "generator"])
    assert (df.loc[sp.test, "attack"] == "paraphrase").any()
