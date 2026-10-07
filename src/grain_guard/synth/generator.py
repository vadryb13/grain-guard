"""Генерация синтетических траекторий хранения зерна."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from grain_guard.config import (
    GRAIN_TYPES,
    HOURS_PER_DAY,
    LAYER_COLUMNS,
    LAYERS,
    SCHEMA_COLUMNS,
    W_SAFE,
    PhysicsParams,
    critical_temp,
)


@dataclass(frozen=True)
class SynthConfig:
    n_silos: int = 1000
    seed: int = 42
    incident_fraction: float = 0.35
    min_days: int = 60
    max_days: int = 120
    hour_step: int = 1
    physics: PhysicsParams = PhysicsParams()


def _air_temperature(hours: np.ndarray, phase: float, base: float) -> np.ndarray:
    seasonal = 8.0 * np.sin(2 * np.pi * (hours / (365.0 * HOURS_PER_DAY)) + phase)
    diurnal = 5.0 * np.sin(2 * np.pi * (hours / HOURS_PER_DAY) - np.pi / 2)
    return base + seasonal + diurnal


def generate(cfg: SynthConfig | None = None) -> pd.DataFrame:
    """Сгенерировать датафрейм датчиков по схеме из config.SCHEMA_COLUMNS."""
    cfg = cfg or SynthConfig()
    rng = np.random.default_rng(cfg.seed)
    p = cfg.physics

    n = cfg.n_silos
    max_hours = cfg.max_days * HOURS_PER_DAY

    days = rng.integers(cfg.min_days, cfg.max_days + 1, size=n)
    valid_len = days * HOURS_PER_DAY
    phase = rng.uniform(0, 2 * np.pi, size=n)
    base_air = rng.uniform(7.0, 16.0, size=n)

    n_in = int(round(n * cfg.incident_fraction))
    incident = np.zeros(n, dtype=bool)
    incident[rng.permutation(n)[:n_in]] = True

    grain_idx = rng.integers(0, len(GRAIN_TYPES), size=n)
    grain_types = np.array(GRAIN_TYPES)[grain_idx]

    # Влажность привязана к безопасной границе хранения (docs/sources.md п. 1.1):
    # инциденты — у партий выше границы, норма — у сухих.
    moisture = np.where(
        incident,
        [rng.uniform(W_SAFE[gt] + 1.5, W_SAFE[gt] + 6.5) for gt in grain_types],
        [rng.uniform(W_SAFE[gt] - 2.5, W_SAFE[gt] + 1.0) for gt in grain_types],
    )

    # Старт инцидента: не раньше 10-го дня и не позже len-10 дней.
    start_hour = np.full(n, -1, dtype=np.int64)
    for i in np.where(incident)[0]:
        low = 10 * HOURS_PER_DAY
        high = max(low + 1, valid_len[i] - 10 * HOURS_PER_DAY)
        start_hour[i] = rng.integers(low, high)

    hotspot = rng.integers(0, LAYERS, size=n)

    # Множитель влажности: выше безопасной влажности -> быстрее самонагрев.
    moist_factor = np.ones(n)
    for i in range(n):
        gt = grain_types[i]
        moist_factor[i] = 1.0 + max(0.0, moisture[i] - W_SAFE[gt]) * p.moisture_gain

    q0 = p.q0 * moist_factor

    # Начальные температуры слоёв: вертикальный градиент по столбу зерна
    # (нижние слои холоднее/теплее верхних, направление случайно) + шум.
    offset = rng.uniform(2.0, 6.0, size=n)
    grad = rng.choice([-1.0, 1.0], size=n)[:, None] * np.linspace(-1.5, 1.5, LAYERS)[None, :]
    t_layers = np.empty((n, LAYERS), dtype=np.float64)
    for i in range(n):
        base = _air_temperature(np.arange(48), phase[i], base_air[i]).mean()
        t_layers[i] = base + offset[i] + grad[i] + rng.normal(0, 0.2, size=LAYERS)

    out = np.full((max_hours, n, LAYERS), np.nan, dtype=np.float32)
    air_all = np.full((max_hours, n), np.nan, dtype=np.float32)

    rel = p.relax_rate
    diff = p.diffusion
    for h in range(max_hours):
        air = _air_temperature(h, phase, base_air)
        air_all[h] = air.astype(np.float32)

        relax = (air[:, None] - t_layers) * rel
        heat = np.zeros_like(t_layers)
        active = incident & (h >= start_hour)
        if active.any():
            q = q0[:, None] * p.q10 ** ((t_layers - p.t_ref) / 10.0)
            idx = np.arange(n)
            heat[idx, hotspot] += np.where(active, q[idx, hotspot], 0.0)
        # Перенос тепла между соседними слоями.
        lap = np.zeros_like(t_layers)
        lap[:, 1:] += t_layers[:, :-1] - t_layers[:, 1:]
        lap[:, :-1] += t_layers[:, 1:] - t_layers[:, :-1]
        t_layers = t_layers + relax + heat + diff * lap + rng.normal(0, 0.02, size=(n, LAYERS))
        t_layers = np.clip(t_layers, p.t_min, p.t_max)
        out[h] = t_layers.astype(np.float32)

    # Дата загрузки силоса: случайный день года (TASKS 1.1: data_zagruzki).
    load_day = rng.integers(0, 365, size=n)
    load_ts = np.datetime64("2020-01-01T00:00") + load_day.astype("timedelta64[D]")
    start_ts_base = load_ts.astype("datetime64[h]")

    incident_start = np.full(n, np.datetime64("NaT"), dtype="datetime64[h]")
    for i in np.where(incident)[0]:
        incident_start[i] = start_ts_base[i] + np.timedelta64(int(start_hour[i]), "h")

    operator_detect = np.full(n, np.datetime64("NaT"), dtype="datetime64[h]")
    for i in range(n):
        seg = out[: valid_len[i], i, :]
        hot = seg.max(axis=1)
        over = np.where(hot >= 32.0)[0]
        if over.size:
            operator_detect[i] = start_ts_base[i] + np.timedelta64(int(over[0]), "h")

    frames = []
    for i in range(n):
        length = int(valid_len[i])
        tt = out[:length, i, :]
        df = pd.DataFrame(tt, columns=LAYER_COLUMNS)
        df.insert(0, "storage_day", np.arange(length) // HOURS_PER_DAY)
        df.insert(0, "timestamp", pd.date_range(load_ts[i], periods=length, freq="h"))
        df.insert(0, "T_air", air_all[:length, i])
        # Влажность: базовый уровень + сезонный дрейф + рост в очаге после
        # начала инцидента (TASKS 1.3), не выходя за 9..25 %.
        hours_i = np.arange(length)
        w = moisture[i] + 0.6 * np.sin(2 * np.pi * hours_i / 2160.0 + phase[i])
        if incident[i]:
            rise = np.clip((hours_i - start_hour[i]) / 24.0 * 0.08, 0.0, 1.5)
            w = w + rise
        df["W_moisture"] = np.clip(w, 9.5, 24.5)
        df["grain_type"] = grain_types[i]
        df["incident"] = int(incident[i])
        df["incident_start"] = incident_start[i]
        df["operator_detect"] = operator_detect[i]
        df["silo_id"] = f"silo_{i:05d}"
        frames.append(df[SCHEMA_COLUMNS])

    result = pd.concat(frames, ignore_index=True)
    result["timestamp"] = pd.to_datetime(result["timestamp"])
    result["incident_start"] = pd.to_datetime(result["incident_start"])
    result["operator_detect"] = pd.to_datetime(result["operator_detect"])

    # Убедимся, что оператор замечает только после начала самонагрева.
    mask = (
        result["incident"].eq(1)
        & result["operator_detect"].notna()
        & (result["operator_detect"] < result["incident_start"])
    )
    result.loc[mask, "operator_detect"] = result.loc[mask, "incident_start"]
    return result


def validate(df: pd.DataFrame) -> None:
    """Проверить физическую валидность сгенерированного датасета."""
    assert list(df.columns) == SCHEMA_COLUMNS, "Схема столбцов не совпадает"
    joined = df[LAYER_COLUMNS].to_numpy()
    assert np.nanmin(joined) >= -5.0 and np.nanmax(joined) <= 60.0, "T выходит за -5..60"
    w = df["W_moisture"]
    assert w.min() >= 9.0 and w.max() <= 25.0, "Влажность выходит за 9..25"
    for silo, g in df.groupby("silo_id"):
        ts = g["timestamp"]
        assert ts.is_monotonic_increasing and ts.is_unique, f"Плохой timestamp у {silo}"


def critical_for_row(grain_type: str, moisture: float) -> float:
    return critical_temp(grain_type, moisture)
