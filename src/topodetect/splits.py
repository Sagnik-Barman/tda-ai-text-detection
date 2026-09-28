"""Train/test splits for the robustness grid.

All splits are made at the *group* level (a group = one prompt / source document),
so a human answer and the AI answers to the same question never straddle train
and test. Split families:

  in_domain          random 70/30 over groups, clean (attack == "none") data
  loo_gen:<g>        train on every AI generator except g, test on g   (unseen model)
  loo_dom:<d>        train on every domain except d, test on d         (unseen domain)
  attack:<a>         train clean, test on texts perturbed by attack a  (adversarial). The model that
                     *performed* the rewrite (``attacker_generators``) is removed from training, so a
                     trained detector can't pass by recognising the attacker's own writing style
  temp:<t>           train at the reference temperature(s), test at t  (decoding shift)
  heldout:<source>   train without that source, test on it  (e.g. frontier-model texts)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Split:
    name: str
    family: str
    train: np.ndarray  # integer row positions into the dataframe
    test: np.ndarray


def _group_partition(groups: pd.Series, test_size: float, seed: int) -> np.ndarray:
    """Boolean mask: True where the row's group is assigned to test."""
    uniq = np.array(sorted(groups.unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    test_groups = set(uniq[: int(round(len(uniq) * test_size))])
    return groups.isin(test_groups).to_numpy()


def _ok(df: pd.DataFrame, tr: np.ndarray, te: np.ndarray, min_per_class: int) -> bool:
    ytr, yte = df["label"].to_numpy()[tr], df["label"].to_numpy()[te]
    return all(((y == c).sum() >= min_per_class) for y in (ytr, yte) for c in (0, 1))


def make_splits(df: pd.DataFrame, test_size: float = 0.3, seed: int = 0, min_per_class: int = 20,
                min_per_generator: int = 50, reference_temps=(0.7,), heldout_sources=("frontier",),
                families=("in_domain", "loo_gen", "loo_dom", "attack", "temp", "heldout"),
                attacker_generators=()) -> list[Split]:
    df = df.reset_index(drop=True)
    is_test = _group_partition(df["group"], test_size, seed)
    clean = (df["attack"] == "none").to_numpy()
    human = (df["label"] == 0).to_numpy()
    held = df["source"].isin(heldout_sources).to_numpy()
    base = clean & ~held
    splits: list[Split] = []

    def add(name, family, tr_mask, te_mask):
        tr, te = np.flatnonzero(tr_mask), np.flatnonzero(te_mask)
        if _ok(df, tr, te, min_per_class):
            splits.append(Split(name, family, tr, te))

    if "in_domain" in families:
        add("in_domain", "in_domain", base & ~is_test, base & is_test)

    if "loo_gen" in families:
        gens = df.loc[base & ~human, "generator"].value_counts()
        for g in gens[gens >= min_per_generator].index:
            if len(gens) < 2:
                break
            gm = (df["generator"] == g).to_numpy()
            add(f"loo_gen:{g}", "loo_gen", base & ~is_test & (human | ~gm), base & is_test & (human | gm))

    if "loo_dom" in families:
        doms = df.loc[base, "domain"].unique()
        if len(doms) >= 2:
            for d in sorted(doms):
                dm = (df["domain"] == d).to_numpy()
                add(f"loo_dom:{d}", "loo_dom", base & ~is_test & ~dm, base & is_test & dm)

    if "attack" in families:
        for a in sorted(set(df["attack"]) - {"none"}):
            am = (df["attack"] == a).to_numpy() & ~held
            has_attacked_humans = (am & human).any()
            te = is_test & (am | (human & clean & ~held if not has_attacked_humans else False))
            not_attacker = ~df["generator"].isin(attacker_generators).to_numpy()
            add(f"attack:{a}", "attack", base & ~is_test & not_attacker, te)

    if "temp" in families and df["temperature"].notna().any():
        temps = sorted(df.loc[base & ~human & df["temperature"].notna(), "temperature"].unique())
        if len(temps) >= 2:
            ref = df["temperature"].isin(reference_temps).to_numpy()
            for t in temps:
                tm = (df["temperature"] == t).to_numpy()
                add(f"temp:{t:g}", "temp", base & ~is_test & (human | ref), base & is_test & (human | tm))

    if "heldout" in families:
        for s in heldout_sources:
            sm = (df["source"] == s).to_numpy()
            if not sm.any():
                continue
            has_humans = (sm & human).any()
            te = sm | ((base & is_test & human) if not has_humans else False)
            add(f"heldout:{s}", "heldout", base & ~is_test, te)

    return splits
