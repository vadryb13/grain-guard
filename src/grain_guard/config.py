"""Доменные константы и параметры моделей grain-guard."""

from __future__ import annotations

from dataclasses import dataclass

LAYERS = 6
HOURS_PER_DAY = 24
HORIZON_HOURS = 72

# Температуры безопасного/критического хранения и влажность (справочные данные).
TCRIT_BASE: dict[str, float] = {
    "wheat": 30.0,
    "barley": 28.0,
    "corn": 27.0,
}
W_SAFE: dict[str, float] = {
    "wheat": 14.0,
    "barley": 13.5,
    "corn": 14.5,
}
GRAIN_TYPES = tuple(TCRIT_BASE)
T_OPERATOR_WARN = 32.0


def critical_temp(grain_type: str, moisture: float) -> float:
    """Критическая температура самосогревания как функция влажности.

    Чем выше влажность, тем ниже порог. Результат ограничен диапазоном 15..40 °C.
    """
    base = TCRIT_BASE[grain_type]
    safe = W_SAFE[grain_type]
    t = base - 1.5 * (moisture - safe)
    return max(15.0, min(40.0, t))


@dataclass(frozen=True)
class PhysicsParams:
    """Параметры тепловой модели зерновой массы."""

    q10: float = 2.5
    t_ref: float = 20.0
    relax_rate: float = 0.0015  # 1/ч, теплообмен с воздухом
    diffusion: float = 0.02  # доля переноса между соседними слоями
    q0: float = 0.07  # базовый прирост °C/ч при t_ref
    moisture_gain: float = 0.2  # множитель роста при влажности выше безопасной
    detection_slope_window: int = 24
    min_slope: float = 0.005  # °C/ч — ниже считаем шумом
    t_min: float = -5.0
    t_max: float = 55.0


SCHEMA_COLUMNS = [
    "silo_id",
    "timestamp",
    "storage_day",
    "T_air",
    "W_moisture",
    "T_layer_1",
    "T_layer_2",
    "T_layer_3",
    "T_layer_4",
    "T_layer_5",
    "T_layer_6",
    "grain_type",
    "incident",
    "incident_start",
    "operator_detect",
]

LAYER_COLUMNS = [f"T_layer_{i}" for i in range(1, LAYERS + 1)]
