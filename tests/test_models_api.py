"""Тесты HTTP-сервиса моделей: fit/predict roundtrip, утечка, консистентность.

Проверяется и автотест на утечку (Этап 3): перемешанный таргет -> AUC ~ 0.5.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from grain_guard.features.build import build_features
from grain_guard.models.server import create_app
from grain_guard.models.train import deserialize_bundles, fit_baselines, split_by_group
from grain_guard.synth.generator import SynthConfig, generate

N_SILOS = 20


@pytest.fixture(scope="module")
def data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    df = generate(SynthConfig(n_silos=N_SILOS, seed=42))
    X, y, meta = build_features(df)
    tr, te = split_by_group(X, y, meta, test_size=0.3, seed=42)
    return df, X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te]


def _parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


def _fit_files(
    X_tr: pd.DataFrame, y_tr: pd.Series, X_te: pd.DataFrame, y_te: pd.Series
) -> dict[str, tuple[str, bytes, str]]:
    pq = "application/octet-stream"
    return {
        "X_train": ("X.parquet", _parquet_bytes(X_tr), pq),
        "y_train": ("y.parquet", _parquet_bytes(y_tr.to_frame("y")), pq),
        "X_test": ("X.parquet", _parquet_bytes(X_te), pq),
        "y_test": ("y.parquet", _parquet_bytes(y_te.to_frame("y")), pq),
    }


def test_fit_predict_roundtrip(data) -> None:
    _, X_tr, y_tr, X_te, y_te = data
    client = TestClient(create_app())

    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"

    r = client.post("/v1/fit", files=_fit_files(X_tr, y_tr, X_te, y_te), params={"seed": 42})
    assert r.status_code == 200, r.text
    model_bytes = r.content

    r = client.post(
        "/v1/predict",
        files={
            "model": ("model.joblib", model_bytes, "application/octet-stream"),
            "X": ("X.parquet", _parquet_bytes(X_te), "application/octet-stream"),
        },
        params={"bundle_name": "best"},
    )
    assert r.status_code == 200, r.text
    proba = np.array(r.json()["proba"])
    assert len(proba) == len(X_te)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_remote_fit_matches_local(data) -> None:
    _, X_tr, y_tr, X_te, y_te = data
    local = fit_baselines(X_tr, y_tr, X_te, y_te, seed=42)
    remote_client = TestClient(create_app())
    r = remote_client.post(
        "/v1/fit",
        files=_fit_files(X_tr, y_tr, X_te, y_te),
        params={"seed": 42},
    )
    remote_bundles = deserialize_bundles(r.content)
    local_proba = local["best"].predict_proba(X_te)
    remote_proba = remote_bundles["best"].predict_proba(X_te)
    np.testing.assert_allclose(local_proba, remote_proba, atol=1e-9)


def test_auth_token_rejected() -> None:
    import os

    os.environ["GRAIN_GUARD_API_TOKEN"] = "secret"
    try:
        client = TestClient(create_app())
        r = client.post("/v1/fit", files={})
        assert r.status_code == 401
    finally:
        del os.environ["GRAIN_GUARD_API_TOKEN"]


def test_no_leakage_shuffled_target(data) -> None:
    """Перемешанный таргет не должен обучаться: AUC ~ 0.5 (автотест на утечку, Этап 3)."""
    _, X_tr, y_tr, X_te, y_te = data
    rng = np.random.default_rng(0)
    y_shuffled = pd.Series(rng.permutation(y_tr.to_numpy()), index=y_tr.index)
    bundles = fit_baselines(X_tr, y_shuffled, X_te, y_te, seed=42)
    auc = bundles["best"].metrics["roc_auc"]
    assert auc < 0.65, f"AUC на случайном таргете {auc} — признак утечки"
