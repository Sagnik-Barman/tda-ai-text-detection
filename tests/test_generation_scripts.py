"""Generation scripts with Ollama / the HF hub mocked out (no network needed)."""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from topodetect import data as D  # noqa: E402
from topodetect import genutils as G  # noqa: E402


def load_script(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), ROOT / "scripts" / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fake_questions(domains, per_domain, min_words, seed):
    return [dict(domain=d, question=f"Why is the sky blue {i} {d}?", human="word " * 90,
                 group=f"hc3-{d}-{i}") for d in domains for i in range(per_domain)]


def fake_generate(url, model, prompt, temperature, max_tokens, seed=0, timeout=0):
    return f"<think>hmm</think> {model} at {temperature} says " + "blue light scatters more. " * 20


def test_generate_paraphrase_and_frontier(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "hc3_questions", fake_questions)
    monkeypatch.setattr(G, "ollama_generate", fake_generate)
    out = tmp_path / "ollama.jsonl"

    gen = load_script("01b_generate_ollama.py")
    monkeypatch.setattr(sys, "argv", ["x", "--per-domain", "2", "--out", str(out)])
    gen.main()
    rows = D.read_jsonl(out)
    ai = [r for r in rows if r["label"] == 1]
    assert len([r for r in rows if r["label"] == 0]) == 10          # 5 domains x 2 questions
    assert len(ai) == 10 * 7                                         # 4 models @0.7 + 3 sweep temps
    assert all("<think>" not in r["text"] for r in ai)
    assert {r["temperature"] for r in ai} == {0.2, 0.7, 1.0, 1.4}
    gen.main()                                                       # resumable: no duplicates
    assert len(D.read_jsonl(out)) == len(rows)

    para = load_script("01c_paraphrase.py")
    pout = tmp_path / "para.jsonl"
    monkeypatch.setattr(sys, "argv", ["x", "--inputs", str(out), "--n", "5", "--out", str(pout)])
    para.main()
    prow = D.read_jsonl(pout)
    assert {r["attack"] for r in prow} == {"paraphrase", "humanize"} and len(prow) == 10
    assert all(r["generator"] != "llama3.1:8b" for r in prow)       # attacker never rewrites itself

    fr = load_script("01d_collect_frontier.py")
    monkeypatch.setattr(fr, "OUT", tmp_path / "frontier.jsonl")
    csv_path = tmp_path / "manual.csv"
    csv_path.write_text('text,generator,domain,question\n"' + "An answer. " * 30 + '",gpt-5,open_qa,What is X?\n',
                        encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["x", "csv", str(csv_path)])
    fr.main()
    f = D.read_jsonl(tmp_path / "frontier.jsonl")
    assert f[0]["source"] == "frontier" and f[0]["label"] == 1 and D.validate(f)
