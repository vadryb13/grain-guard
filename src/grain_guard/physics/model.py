"""Физическая модель самосогревания (baseline №1)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from grain_guard.config import HORIZON_HOURS, LAYER_COLUMNS, PhysicsParams, critical_temp


def _tcrit_series(df: pd.DataFrame) -> np.ndarray:
    pairs = df[["grain_type", "W_moisture"]].drop_duplicates()
    mapping = {
        (r.grain_type, float(r.W_moisture)): critical_temp(r.grain_type, float(r.W_moisture))
        for r in pairs.itertuples()
    }
    return np.array(
        [
            mapping[(g, float(w))]
            for g, w in zip(df["grain_type"], df["W_moisture"], strict=True)
        ]
    )


def forecast_temperature(
    temps: np.ndarray, slopes: np.ndarray, params: PhysicsParams, horizon: int = HORIZON_HOURS
) -> np.ndarray:
    """Экстраполировать температуру на `horizon` часов по Q10-кинетике дыхания зерна."""
    fc = temps.astype(np.float64).copy()
    anchor = temps.astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(horizon):
            rate = slopes * params.q10 ** ((fc - anchor) / 10.0)
            rate = np.clip(rate, 0.0, 50.0)
            fc = np.clip(fc + rate, -50.0, 200.0)
    return fc


def compute_alerts(df: pd.DataFrame, params: PhysicsParams | None = None) -> pd.DataFrame:
    """Вернуть по каждой строке прогноз и флаг тревоги физической модели."""
    params = params or PhysicsParams()
    data = df.sort_values(["silo_id", "timestamp"]).reset_index(drop=True)
    w = params.detection_slope_window

    cols = LAYER_COLUMNS
    temps = data[cols].to_numpy(dtype=np.float64)

    smoothed = (
        data.groupby("silo_id", sort=False)[cols]
        .rolling(w, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .sort_index()[cols]
        .to_numpy(dtype=np.float64)
    )
    shifted = (
        data.groupby("silo_id", sort=False)[cols]
        .shift(w)
        .to_numpy(dtype=np.float64)
    )
    shifted = np.where(np.isnan(shifted), smoothed, shifted)
    slope = (smoothed - shifted) / float(w)
    slope = np.where(slope > params.min_slope, slope, 0.0)

    forecast = forecast_temperature(temps, slope, params)
    forecast_max = forecast.max(axis=1)
    current_max = temps.max(axis=1)
    tcrit = _tcrit_series(data)

    out = pd.DataFrame(
        {
            "silo_id": data["silo_id"].to_numpy(),
            "timestamp": data["timestamp"].to_numpy(),
            "T_max": current_max,
            "forecast_T_max": forecast_max,
            "tcrit": tcrit,
        }
    )
    out["alert"] = (forecast_max >= tcrit) | (current_max >= tcrit)
    return out
