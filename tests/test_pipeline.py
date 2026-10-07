"""Смоук-тест end-to-end конвейера и фабрики бэкендов."""

from __future__ import annotations

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
    assert "net_benefit_rub" in result.econ
    best = result.bundles["best"]
    assert "roc_auc" in best.metrics
