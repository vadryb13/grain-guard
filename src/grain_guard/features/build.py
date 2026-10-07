"""Построение признаков и целевой переменной."""

from __future__ import annotations

import numpy as np
import pandas as pd

from grain_guard.config import HORIZON_HOURS, LAYER_COLUMNS, critical_temp

DELTA_WINDOWS = (6, 12, 24, 48)
ROLL_WINDOWS = (24, 72)


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Вернуть (X, y, meta).

    meta содержит silo_id, timestamp, incident_start, operator_detect и keep-флаг.
    Признаки считаются только по прошлым наблюдениям (без утечки из будущего).
    """
    data = df.sort_values(["silo_id", "timestamp"]).reset_index(drop=True).copy()

    feat = pd.DataFrame(index=data.index)
    for c in LAYER_COLUMNS:
        feat[c] = data[c]

    temps = data[LAYER_COLUMNS].to_numpy(dtype=np.float64)
    feat["T_max"] = temps.max(axis=1)
    feat["T_mean"] = temps.mean(axis=1)
    feat["T_min"] = temps.min(axis=1)
    feat["T_grad"] = feat["T_max"] - feat["T_min"]
    feat["hot_layer"] = temps.argmax(axis=1)

    # Дельта самого горячего слоя для разных окон (сдвиг — только прошлые наблюдения).
    hot = feat["T_max"]
    for k in DELTA_WINDOWS:
        shifted = hot.groupby(data["silo_id"], sort=False).shift(k)
        feat[f"dTmax_{k}h"] = hot - shifted

    for w in ROLL_WINDOWS:
        feat[f"Tmax_mean_{w}h"] = (
            hot.groupby(data["silo_id"], sort=False)
            .rolling(w, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
            .sort_index()
        )
        feat[f"Tmax_std_{w}h"] = (
            hot.groupby(data["silo_id"], sort=False)
            .rolling(w, min_periods=1)
            .std()
            .reset_index(level=0, drop=True)
            .sort_index()
            .fillna(0.0)
        )

    feat["W_moisture"] = data["W_moisture"].to_numpy()
    feat["T_air"] = data["T_air"].to_numpy()
    feat["storage_day"] = data["storage_day"].to_numpy()
    feat["tcrit"] = [
        critical_temp(g, float(w))
        for g, w in zip(data["grain_type"], data["W_moisture"])
    ]

    dummies = pd.get_dummies(data["grain_type"], prefix="grain")
    feat = pd.concat([feat, dummies], axis=1)
    feat = feat.fillna(0.0)

    hours_to_start = (
        data["incident_start"] - data["timestamp"]
    ).dt.total_seconds() / 3600.0
    y = (
        data["incident"].eq(1)
        & hours_to_start.gt(0)
        & hours_to_start.le(HORIZON_HOURS)
    ).astype(int)

    # Не обучаемся на постинцидентных строках (после начала порчи).
    keep = ~(data["incident"].eq(1) & hours_to_start.lt(0))

    meta = pd.DataFrame(
        {
            "silo_id": data["silo_id"].to_numpy(),
            "timestamp": data["timestamp"].to_numpy(),
            "incident": data["incident"].to_numpy(),
            "incident_start": data["incident_start"].to_numpy(),
            "operator_detect": data["operator_detect"].to_numpy(),
            "keep": keep.to_numpy(),
        }
    )
    return feat, y, meta
