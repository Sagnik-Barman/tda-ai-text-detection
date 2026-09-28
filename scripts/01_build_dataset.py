"""Assemble the evaluation dataset from HC3 / RAID / MAGE / local JSONL files.

    python scripts/01_build_dataset.py                 # uses configs/default.yaml
    python scripts/01_build_dataset.py --config my.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from topodetect import data as D  # noqa: E402
from topodetect.config import load_config, paths  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    dc, seed = cfg["data"], cfg["seed"]
    rows: list[dict] = []

    if dc["hc3"]["enabled"]:
        print("loading HC3 ...")
        rows += D.load_hc3(dc["hc3"]["domains"], dc["hc3"]["max_per_domain"], dc["min_words"], seed)
    if dc["raid"]["enabled"]:
        r = dc["raid"]
        print("streaming RAID (this can take a while) ...")
        rows += D.load_raid(r["split"], r["models"], r["domains"], r["attacks"], r["decodings"],
                            r["n_per_cell"], dc["min_words"], r["max_scan"], seed)
    if dc["mage"]["enabled"]:
        print("loading MAGE ...")
        rows += D.load_mage(dc["mage"]["split"], dc["mage"]["n"], dc["min_words"], seed)
    for f in dc.get("local_files") or []:
        p = ROOT / f
        if p.exists():
            local = D.validate(D.read_jsonl(p))
            n_fixed = 0
            for r in local:
                if r["label"] == 1:
                    t = D.strip_generation_artifacts(r["text"])
                    n_fixed += t != r["text"]
                    r["text"] = t
            local = [r for r in local if D.n_words(r["text"]) >= dc["min_words"]]
            if n_fixed:
                print(f"  stripped meta-preambles / markdown from {n_fixed} generated texts")
            print(f"loaded {len(local)} rows from {f}")
            rows += local

    rows = D.dedupe(rows)
    out = paths(cfg)["dataset"]
    D.write_jsonl(rows, out)
    print(D.summarize(rows))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
