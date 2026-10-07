"""HTTP-сервис обучения/инференса моделей (запускается на VPS)."""

from __future__ import annotations

import io
import os

import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, Response, Security, UploadFile
from fastapi.security import APIKeyHeader

from grain_guard import __version__
from grain_guard.models.train import (
    deserialize_bundles,
    fit_baselines,
    serialize_bundles,
)

TOKEN_HEADER = "X-API-Token"
_app_state: dict[str, bytes] = {}


def _check_token(
    token: str | None = Security(APIKeyHeader(name=TOKEN_HEADER, auto_error=False)),
) -> None:
    expected = os.environ.get("GRAIN_GUARD_API_TOKEN")
    if expected and token != expected:
        raise HTTPException(status_code=401, detail="Неверный токен")


def create_app() -> FastAPI:
    app = FastAPI(title="grain-guard model service", version=__version__)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.post("/v1/fit", dependencies=[Depends(_check_token)])
    def fit(
        X_train: UploadFile = File(...),
        y_train: UploadFile = File(...),
        X_test: UploadFile = File(...),
        y_test: UploadFile = File(...),
        seed: int = 42,
    ) -> Response:
        try:
            bundles = fit_baselines(
                pd.read_parquet(io.BytesIO(X_train.file.read())),
                pd.read_parquet(io.BytesIO(y_train.file.read())).iloc[:, 0],
                pd.read_parquet(io.BytesIO(X_test.file.read())),
                pd.read_parquet(io.BytesIO(y_test.file.read())).iloc[:, 0],
                seed=seed,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        best = bundles["best"]
        headers = {"X-Metrics": str(best.metrics)}
        return Response(
            content=serialize_bundles(bundles),
            media_type="application/octet-stream",
            headers=headers,
        )

    @app.post("/v1/predict", dependencies=[Depends(_check_token)])
    def predict(
        model: UploadFile = File(...),
        X: UploadFile = File(...),
        bundle_name: str = "best",
    ) -> dict[str, list[float]]:
        data = pd.read_parquet(io.BytesIO(X.file.read()))
        bundles = deserialize_bundles(model.file.read())
        if bundle_name not in bundles:
            raise HTTPException(status_code=404, detail=f"Нет бандла '{bundle_name}'")
        proba = bundles[bundle_name].predict_proba(data)
        return {"proba": [round(float(p), 6) for p in proba]}

    @app.post("/v1/models", dependencies=[Depends(_check_token)])
    def store_model(model: UploadFile = File(...)) -> dict[str, str]:
        _app_state["current"] = model.file.read()
        return {"status": "stored"}

    @app.get("/v1/models", dependencies=[Depends(_check_token)])
    def fetch_model() -> Response:
        if "current" not in _app_state:
            raise HTTPException(status_code=404, detail="Модель не сохранена")
        return Response(content=_app_state["current"], media_type="application/octet-stream")

    return app


def main() -> None:  # pragma: no cover - ручной запуск
    import uvicorn

    uvicorn.run(
        "grain_guard.models.server:create_app",
        factory=True,
        host=os.environ.get("GRAIN_GUARD_HOST", "127.0.0.1"),
        port=int(os.environ.get("GRAIN_GUARD_PORT", "8000")),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
