"""Chunked, resumable feature storage.

Features for each group are written as ``features/<dataset>/<group>/part_00000.npz``
(ids, names, X). Re-running extraction skips finished parts, so a long run on a
laptop can be interrupted and resumed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def part_path(root: Path, group: str, k: int) -> Path:
    return Path(root) / group / f"part_{k:05d}.npz"


def save_part(path: Path, ids: list[str], names: list[str], X: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, ids=np.asarray(ids), names=np.asarray(names), X=np.asarray(X, dtype=np.float32))
    tmp.replace(path)


def load_group(root: Path, group: str) -> pd.DataFrame | None:
    parts = sorted((Path(root) / group).glob("part_*.npz"))
    parts = [p for p in parts if not p.name.endswith(".tmp.npz")]
    if not parts:
        return None
    frames = []
    for p in parts:
        z = np.load(p, allow_pickle=False)
        frames.append(pd.DataFrame(z["X"], index=z["ids"].astype(str), columns=z["names"].astype(str)))
    df = pd.concat(frames)
    return df[~df.index.duplicated(keep="last")]


def load_features(root: Path, groups=("cloud", "attn", "ppl", "bino")) -> pd.DataFrame:
    frames = [f for g in groups if (f := load_group(root, g)) is not None]
    if not frames:
        raise FileNotFoundError(f"no features under {root}")
    return pd.concat(frames, axis=1, join="outer")
