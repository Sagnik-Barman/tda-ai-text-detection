from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "configs" / "default.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def paths(cfg: dict) -> dict[str, Path]:
    name = cfg.get("dataset_name", "main")
    return dict(
        dataset=ROOT / "data" / "processed" / f"{name}.jsonl",
        features=ROOT / "features" / name,
        results=ROOT / "results" / name,
        models=ROOT / "models" / name,
    )
