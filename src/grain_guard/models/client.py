"""HTTP-клиент к сервису моделей: тот же интерфейс, что и локальный fit_baselines."""

from __future__ import annotations

import io
import os

import pandas as pd

from grain_guard.models.train import ModelBundle, deserialize_bundles

DEFAULT_TIMEOUT_S = 1800.0


def _df_parquet(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


def _series_parquet(s: pd.Series) -> bytes:
    return _df_parquet(s.to_frame(name="y"))


class RemoteBackend:
    """Вызов обучения и инференса на удалённом сервисе (VPS)."""

    def __init__(self, base_url: str | None = None, token: str | None = None,
                 timeout: float = DEFAULT_TIMEOUT_S) -> None:
        self.base_url = (base_url or os.environ.get("GRAIN_GUARD_API_URL", "")).rstrip("/")
        self.token = token or os.environ.get("GRAIN_GUARD_API_TOKEN")
        self.timeout = timeout
        if not self.base_url:
            raise ValueError(
                "Не задан адрес сервиса: установите GRAIN_GUARD_API_URL или configs/remote.yaml"
            )

    def _headers(self) -> dict[str, str]:
        return {"X-API-Token": self.token} if self.token else {}

    def health(self) -> dict[str, str]:
        import httpx

        r = httpx.get(f"{self.base_url}/health", timeout=10.0)
        r.raise_for_status()
        return r.json()

    def fit_baselines(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        seed: int = 42,
    ) -> dict[str, ModelBundle]:
        import httpx

        files = {
            "X_train": ("X_train.parquet", _df_parquet(X_train), "application/octet-stream"),
            "y_train": ("y_train.parquet", _series_parquet(y_train), "application/octet-stream"),
            "X_test": ("X_test.parquet", _df_parquet(X_test), "application/octet-stream"),
            "y_test": ("y_test.parquet", _series_parquet(y_test), "application/octet-stream"),
        }
        r = httpx.post(
            f"{self.base_url}/v1/fit",
            files=files,
            params={"seed": seed},
            headers=self._headers(),
            timeout=self.timeout,
        )
        if r.status_code != 200:
            raise RuntimeError(f"Сервис обучения вернул {r.status_code}: {r.text[:500]}")
        return deserialize_bundles(r.content)

    def predict_proba(self, model_bytes: bytes, X: pd.DataFrame, name: str = "best") -> list[float]:
        import httpx

        files = {
            "model": ("model.joblib", model_bytes, "application/octet-stream"),
            "X": ("X.parquet", _df_parquet(X), "application/octet-stream"),
        }
        r = httpx.post(
            f"{self.base_url}/v1/predict",
            files=files,
            params={"bundle_name": name},
            headers=self._headers(),
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["proba"]
