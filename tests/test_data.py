"""Dataset loaders with the HF hub mocked (formats as documented by HC3 / MAGE / RAID)."""

import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from topodetect import data as D  # noqa: E402

LONG = "This is a fairly long answer sentence , with a spaced comma . " * 8


def test_hc3_loader(tmp_path, monkeypatch):
    for dom in D.HC3_DOMAINS:
        with open(tmp_path / f"{dom}.jsonl", "w") as f:
            for i in range(5):
                f.write(json.dumps(dict(question=f"q{i}", human_answers=[LONG + str(i)],
                                        chatgpt_answers=[LONG + "gpt" + str(i), "too short"])) + "\n")
    fake = types.ModuleType("huggingface_hub")
    fake.hf_hub_download = lambda repo, fn, repo_type=None: str(tmp_path / fn)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    rows = D.load_hc3(max_per_domain=3, min_words=20)
    assert len(rows) == 5 * 3 * 2
    assert all(" ," not in r["text"] for r in rows)                  # HC3 spacing artefacts cleaned
    by_group = {}
    for r in rows:
        by_group.setdefault(r["group"], set()).add(r["label"])
    assert any(v == {0, 1} for v in by_group.values())               # human + AI share a group


@pytest.mark.parametrize("src,expected", [
    ("xsum_human", ("xsum", "human")),
    ("xsum_machine_continuation_gpt-3.5-turbo", ("xsum", "gpt-3.5-turbo")),
    ("sci_gen_machine_topical_flan_t5_xxl", ("sci", "xxl")),
])
def test_mage_src_parsing(src, expected):
    dom, gen = D._parse_mage_src(src)
    assert dom == expected[0]
    assert (gen == "human") == (expected[1] == "human")


def test_raid_loader_streams_and_caps(monkeypatch):
    rows = []
    for i in range(400):
        m = ["human", "gpt4", "llama-chat"][i % 3]
        rows.append(dict(id=f"r{i}", source_id=f"s{i // 3}", adv_source_id=None, model=m, decoding="sampling",
                         repetition_penalty="no", attack=["none", "paraphrase"][i % 2], domain="news",
                         title="t", prompt="p", generation=LONG))
    fake = types.ModuleType("datasets")
    fake.load_dataset = lambda *a, **k: iter(rows)
    monkeypatch.setitem(sys.modules, "datasets", fake)
    out = D.load_raid(attacks=("none",), n_per_cell=20, min_words=20)
    assert {r["attack"] for r in out} == {"none"}
    assert {r["generator"] for r in out} == {"human", "gpt4", "llama-chat"}
    assert max(sum(r["generator"] == g for r in out) for g in ("human", "gpt4")) <= 20
    assert all((r["label"] == 1) == (r["generator"] != "human") for r in out)


def test_raid_schema_change_is_loud(monkeypatch):
    fake = types.ModuleType("datasets")
    fake.load_dataset = lambda *a, **k: iter([dict(text="x", label=1)])
    monkeypatch.setitem(sys.modules, "datasets", fake)
    with pytest.raises(KeyError, match="schema changed"):
        D.load_raid()


@pytest.mark.parametrize("raw,clean", [
    ("Here's the rewritten text: The sky is blue.", "The sky is blue."),
    ("Sure! Here is a paraphrased version of the text:\n\nThe sky is blue.", "The sky is blue."),
    ("Okay, let's break this down. The sky is blue.", "Okay, let's break this down. The sky is blue."),
    ("The **sky** is blue. ## Why It scatters.", "The sky is blue. Why It scatters."),
    ("Here we go again, said the human.", "Here we go again, said the human."),
])
def test_strip_generation_artifacts(raw, clean):
    assert D.strip_generation_artifacts(raw) == clean
