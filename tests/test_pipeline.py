"""Смоук-тест end-to-end конвейера и фабрики бэкендов."""

from __future__ import annotations

import pandas as pd

from grain_guard.models import get_backend
from grain_guard.pipeline import run


def test_backend_default_is_local() -> None:
    backend = get_backend()
    assert hasattr(backend, "fit_baselines")
    assert not hasattr(backend, "health"), (
        "Без GRAIN_GUARD_API_URL должен выбираться локальный бэкенд"
    )


def test_pipeline_smoke() -> None:
    result = run(n_silos=12, seed=42, write_reports=False)
    assert not result.benchmark.empty
    systems = set(result.benchmark["system"])
    assert "physics" in systems
    assert any(s.startswith("ml_") for s in systems)


def test_pipeline_on_real_data_without_labels() -> None:
    """Этап 5: прогон на ingested-данных без разметки — физика работает, ML пропущен."""
    from grain_guard.synth.generator import SynthConfig, generate

    df = generate(SynthConfig(n_silos=8, seed=5))
    df.loc[df["incident"] == 1, ["incident", "incident_start", "operator_detect"]] = (
        0,
        pd.NaT,
        pd.NaT,
    )
    result = run(df=df, write_reports=False)
    assert list(result.benchmark["system"]) == ["physics"]
    assert result.bundles == {}
    assert result.ml_name is not None and "пропущен" in result.ml_name
    assert "net_benefit_rub" in result.econ
