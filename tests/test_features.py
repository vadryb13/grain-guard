"""Тесты фич: отсутствие заглядывания в будущее и корректность таргета."""

from __future__ import annotations

import numpy as np

from grain_guard.features.build import build_features
from grain_guard.synth.generator import SynthConfig, generate


def test_target_window_and_keep() -> None:
    df = generate(SynthConfig(n_silos=15, seed=21))
    X, y, meta = build_features(df)

    assert len(X) == len(y) == len(meta)

    # y=1 только в окне (0, 72] часов до incident_start.
    y1 = meta[y.eq(1)]
    hours = (
        (y1["incident_start"] - y1["timestamp"]) / np.timedelta64(1, "h")
    )
    assert (hours > 0).all() and (hours <= 72).all()

    # Постинцидентные строки исключены из обучения.
    post = (meta["incident"] == 1) & meta["incident_start"].notna() & (
        meta["timestamp"] > meta["incident_start"]
    )
    assert not meta.loc[post, "keep"].any()


def test_delta_features_use_only_past() -> None:
    df = generate(SynthConfig(n_silos=5, seed=33))
    X, _, meta = build_features(df)
    cols = [c for c in X.columns if c.startswith("dTmax_")]
    # Первая строка каждого силоса: дельта неизвестна -> 0 (не значение из будущего).
    first_rows = ~meta["silo_id"].duplicated()
    assert X.loc[first_rows.to_numpy(), cols].eq(0.0).all().all()
