"""Загрузка и валидация реальных данных элеватора."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from grain_guard.config import GRAIN_TYPES, LAYER_COLUMNS

REQUIRED = ["silo_id", "timestamp", "T_air", "W_moisture", *LAYER_COLUMNS]


@dataclass
class IngestResult:
    data: pd.DataFrame | None
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def validate(df: pd.DataFrame) -> list[str]:
    """Вернуть список человекочитаемых проблем в данных."""
    errors: list[str] = []
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        errors.append(f"Отсутствуют обязательные столбцы: {', '.join(missing)}")
        return errors

    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    bad_ts = int(ts.isna().sum())
    if bad_ts:
        errors.append(f"Некорректный timestamp в {bad_ts} строках")

    if df["silo_id"].isna().any():
        errors.append("Пустые silo_id")

    joined = df[LAYER_COLUMNS].apply(pd.to_numeric, errors="coerce")
    n_nan = int(joined.isna().sum().sum())
    if n_nan:
        errors.append(f"Пропуски/нечисловые значения в слоях датчиков: {n_nan}")
    tmin, tmax = float(np.nanmin(joined.to_numpy())), float(np.nanmax(joined.to_numpy()))
    if tmin < -5 or tmax > 60:
        errors.append(f"Температура вне физического диапазона -5..60 °C (найдено {tmin:.1f}..{tmax:.1f})")

    w = pd.to_numeric(df["W_moisture"], errors="coerce")
    if w.isna().any():
        errors.append("Некорректная W_moisture")
    elif w.min() < 9 or w.max() > 25:
        errors.append(f"Влажность вне диапазона 9..25 % (найдено {w.min():.1f}..{w.max():.1f})")

    unknown = set(df["grain_type"].dropna().unique()) - set(GRAIN_TYPES)
    if unknown:
        errors.append(f"Неизвестные типы зерна: {', '.join(map(str, unknown))}")

    if not errors:
        dup = df.assign(_ts=ts).groupby("silo_id")["_ts"].apply(lambda s: s.duplicated().any())
        bad = dup[dup].index.tolist()
        if bad:
            errors.append(f"Дублирующиеся timestamp в силосах: {', '.join(map(str, bad[:5]))}")

    return errors


def ingest_csv(path: str | Path, config_path: str | Path | None = None) -> IngestResult:
    """Загрузить CSV, применить маппинг каналов из конфига и провалидировать."""
    raw = pd.read_csv(path)
    cfg = load_config(config_path) if config_path else {}
    mapping = cfg.get("column_map", {})
    if mapping:
        raw = raw.rename(columns=mapping)

    errors = validate(raw)
    if errors:
        return IngestResult(data=None, errors=errors)

    data = raw.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values(["silo_id", "timestamp"]).reset_index(drop=True)
    if "storage_day" not in data.columns:
        data["storage_day"] = data.groupby("silo_id").cumcount() // 24
    if "incident" not in data.columns:
        data["incident"] = 0
    for c in ("incident_start", "operator_detect"):
        if c not in data.columns:
            data[c] = pd.NaT
    return IngestResult(data=data, errors=[])
