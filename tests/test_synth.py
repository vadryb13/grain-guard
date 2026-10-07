"""Тесты синтетического генератора: детерминизм и физическая валидность."""

from __future__ import annotations

import hashlib

import pandas as pd

from grain_guard.synth.generator import SynthConfig, generate, validate


def _frame_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes()
    ).hexdigest()


def test_determinism_same_seed() -> None:
    a = generate(SynthConfig(n_silos=5, seed=7))
    b = generate(SynthConfig(n_silos=5, seed=7))
    assert _frame_hash(a) == _frame_hash(b), (
        "Одинаковый seed должен давать бит-в-бит тот же датасет"
    )


def test_different_seed_differs() -> None:
    a = generate(SynthConfig(n_silos=5, seed=1))
    b = generate(SynthConfig(n_silos=5, seed=2))
    assert _frame_hash(a) != _frame_hash(b)


def test_schema_and_ranges() -> None:
    df = generate(SynthConfig(n_silos=8, seed=3))
    validate(df)  # не должно падать

    layers = df[[c for c in df.columns if c.startswith("T_layer_")]]
    assert layers.min().min() >= -5.0
    assert layers.max().max() <= 60.0
    assert df["W_moisture"].between(9, 25).all()
    assert set(df["grain_type"].unique()) <= {"wheat", "barley", "corn"}
    assert df["silo_id"].nunique() == 8


def test_no_gaps_in_timestamps() -> None:
    df = generate(SynthConfig(n_silos=6, seed=5))
    for silo, g in df.groupby("silo_id"):
        ts = g["timestamp"].sort_values()
        assert ts.is_unique
        assert (ts.diff().dropna() == pd.Timedelta(hours=1)).all(), f"Пропуск часов у {silo}"


def test_incident_share_in_range() -> None:
    df = generate(SynthConfig(n_silos=30, seed=11))
    share = df.groupby("silo_id")["incident"].first().mean()
    assert 0.2 <= share <= 0.5


def test_operator_detect_not_before_incident_start() -> None:
    df = generate(SynthConfig(n_silos=30, seed=13))
    bad = df[
        df["incident"].eq(1)
        & df["operator_detect"].notna()
        & (df["operator_detect"] < df["incident_start"])
    ]
    assert bad.empty
