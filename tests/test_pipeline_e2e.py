"""End-to-end: run every pipeline script on the tiny offline universe (~1-2 min on CPU).

    pytest -q tests/test_pipeline_e2e.py
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from tiny_universe import build, make_rows, write_jsonl  # noqa: E402

NAME = "_e2e_test"


def run(*args):
    r = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, f"{args}\nSTDOUT:\n{r.stdout[-3000:]}\nSTDERR:\n{r.stderr[-3000:]}"
    return r.stdout


@pytest.fixture(scope="module")
def cfg_path(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("universe")
    m = build(tmp / "models")
    write_jsonl(make_rows(40), tmp / "rows.jsonl")
    cfg = yaml.safe_load(open(ROOT / "configs" / "default.yaml"))
    cfg["dataset_name"] = NAME
    cfg["data"]["hc3"]["enabled"] = False
    cfg["data"]["local_files"] = [str(tmp / "rows.jsonl")]
    cfg["features"].update(encoder=m["encoder"], scorer=m["scorer"], bino_observer=m["bino_observer"],
                           bino_performer=m["bino_performer"], chunk_size=60, workers=2)
    cfg["splits"].update(min_per_class=5, min_per_generator=10)
    cfg["roberta"].update(model=m["clf"], epochs=1, batch_size=8, max_length=128)
    p = tmp / "cfg.yaml"
    yaml.safe_dump(cfg, open(p, "w"))
    yield p
    for d in (ROOT / "data" / "processed" / f"{NAME}.jsonl", ROOT / "features" / NAME,
              ROOT / "results" / NAME, ROOT / "models" / NAME):
        shutil.rmtree(d, ignore_errors=True) if d.is_dir() else d.unlink(missing_ok=True)


def test_full_pipeline(cfg_path):
    run("scripts/01_build_dataset.py", "--config", cfg_path)
    run("scripts/02_extract_features.py", "--config", cfg_path)
    run("scripts/05_finetune_roberta.py", "--config", cfg_path, "--families", "in_domain", "loo_gen")
    out = run("scripts/03_run_grid.py", "--config", cfg_path)
    assert "Robustness summary" in out
    run("scripts/04_report.py", "--config", cfg_path)

    res = pd.read_csv(ROOT / "results" / NAME / "grid.csv")
    fams = set(res["family"])
    assert {"in_domain", "loo_gen", "loo_dom", "attack", "temp", "heldout"} <= fams
    dets = set(res["detector"])
    assert {"PHD (zero-shot)", "Attention topology [LR]", "Hybrid topo+likelihood [GBM]",
            "Fast-DetectGPT", "Binoculars", "RoBERTa fine-tuned (supervised)"} <= dets
    assert res["auroc"].notna().all()
    ind = res[res["split"] == "in_domain"].set_index("detector")["auroc"]
    # The tiny models have random weights, so zero-shot scores are ~chance here; trained
    # detectors must still pick up the planted difference (AI text is more repetitive).
    assert ind["Embedding topology [LR]"] > 0.8
    assert ind["Hybrid topo+likelihood [GBM]"] > 0.8
    figs = ROOT / "results" / NAME / "figures"
    for f in ("auroc_heatmap.png", "robustness_gap.png", "phd_by_generator.png",
              "phd_vs_temperature.png", "phd_by_attack.png"):
        assert (figs / f).exists(), f
    assert (ROOT / "models" / NAME / "demo_model.joblib").exists()
