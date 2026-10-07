"""Пакет моделей: локальное обучение и HTTP-бэкенд на VPS."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol

from grain_guard.models.train import (  # noqa: F401
    ModelBundle,
    deserialize_bundles,
    serialize_bundles,
)

REMOTE_URL_ENV = "GRAIN_GUARD_API_URL"
REMOTE_TOKEN_ENV = "GRAIN_GUARD_API_TOKEN"
REMOTE_CONFIG_PATHS = ("configs/remote.yaml", "configs/remote.yml")


class TrainingBackend(Protocol):
    def fit_baselines(self, *args: Any, **kwargs: Any) -> dict[str, ModelBundle]: ...


def _remote_settings_from_config() -> tuple[str | None, str | None]:
    for path in REMOTE_CONFIG_PATHS:
        p = Path(path)
        if p.exists():
            import yaml

            cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            return cfg.get("base_url"), cfg.get("token")
    return None, None


def get_backend() -> TrainingBackend:
    """Локальный бэкенд по умолчанию; HTTP — если задан GRAIN_GUARD_API_URL.

    Адрес можно задать также в configs/remote.yaml (base_url, token).
    """
    url = os.environ.get(REMOTE_URL_ENV)
    token = os.environ.get(REMOTE_TOKEN_ENV)
    cfg_url, cfg_token = _remote_settings_from_config()
    url = url or cfg_url
    token = token or cfg_token
    if url:
        from grain_guard.models.client import RemoteBackend

        return RemoteBackend(base_url=url, token=token)

    from grain_guard.models import train

    return train
