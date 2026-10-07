"""Доменные константы и параметры моделей grain-guard."""

from __future__ import annotations

from dataclasses import dataclass

LAYERS = 6
HOURS_PER_DAY = 24
HORIZON_HOURS = 72

# Температуры безопасного/критического хранения и влажность.
# Источники: docs/sources.md (п. 1.1 CGC, п. 1.2 ASABE D535 / Purdue).

# Граница порчи за 5-6 месяцев хранения (CGC Safe Storage Guidelines):
# (T °C, W %) на концах кривой; выше кривой — порча в течение сезона.
SAFE_BOUNDARY: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "wheat": ((0.0, 18.0), (28.0, 10.0)),
    "barley": ((5.0, 18.0), (20.0, 10.0)),
    # Кукурузы нет в CGC; граница при AST=150 сут из ASABE D535 (docs/sources.md п.1.2).
    "corn": ((1.7, 22.0), (15.6, 16.0)),
}

W_SAFE: dict[str, float] = {
    "wheat": 14.0,
    "barley": 13.5,
    "corn": 14.5,
}
GRAIN_TYPES = tuple(SAFE_BOUNDARY)
T_OPERATOR_WARN = 32.0

# ASABE Standard D535: allowable storage time (сутки) зерновой кукурузы
# до 0.5 % потерь сухого вещества. Строки — T °C, столбцы — W % w.b.
# Опубликовано Purdue Extension (docs/sources.md п. 1.2); первоисточник
# Steele, Saul & Hukill (1969), Trans. ASAE 12(5):685-689.
_CORN_T_GRID = (1.7, 4.4, 7.2, 10.0, 12.8, 15.6, 18.3, 21.1, 23.9)
_CORN_W_GRID = (16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0)
CORN_AST_DAYS: tuple[tuple[float, ...], ...] = (
    (1144, 437, 216, 128, 86, 63, 50, 41),
    (763, 291, 144, 85, 57, 42, 33, 27),
    (509, 194, 96, 57, 38, 28, 22, 18),
    (339, 130, 64, 38, 26, 19, 15, 12),
    (226, 86, 43, 25, 17, 13, 10, 8),
    (151, 58, 29, 17, 11, 8, 7, 5),
    (113, 43, 22, 13, 9, 7, 5, 4),
    (85, 32, 16, 10, 7, 5, 4, 4),
    (63, 24, 12, 8, 5, 4, 3, 3),
)

# Допущение (docs/sources.md п. 1.2): AST пшеницы/ячменя = AST кукурузы,
# растянутая по температуре так, чтобы наклон границы AST=150 сут совпал с CGC.
# Наклоны, °C на 1 % влажности: wheat 3.5, barley 1.875, corn (ASABE) ≈ 2.85.
AST_SCALE: dict[str, float] = {
    "corn": 1.0,
    "wheat": 3.5 / 2.85,  # ≈ 1.23
    "barley": 1.875 / 2.85,  # ≈ 0.66
}


def _interp1d(xs: tuple[float, ...], ys: tuple[float, ...], x: float) -> float:
    """Линейная интерполяция с зажимом по краям."""
    if x <= xs[0]:
        return float(ys[0])
    if x >= xs[-1]:
        return float(ys[-1])
    for i in range(1, len(xs)):
        if x <= xs[i]:
            k = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return float(ys[i - 1] + k * (ys[i] - ys[i - 1]))
    return float(ys[-1])


def ast_days(grain_type: str, temp_c: float, moisture: float) -> float:
    """Допустимое время хранения до 0.5 % потерь сухого вещества, сутки.

    Кукуруза: билинейная интерполяция таблицы ASABE D535. За краем сетки по T —
    лог-линейная экстраполяция «AST ×2 на каждые −5.6 °C» (NDSU/Hellevang:
    охлаждение зерна на 10 °F удваивает допустимое время; docs/sources.md п. 1.2).
    Пшеница/ячмень: температура кукурузной сетки масштабируется на
    AST_SCALE[grain] (допущение, docs/sources.md п. 1.2).
    """
    t = temp_c / AST_SCALE[grain_type]
    w = max(_CORN_W_GRID[0], min(_CORN_W_GRID[-1], moisture))
    row = tuple(
        _interp1d(_CORN_W_GRID, CORN_AST_DAYS[i], w) for i in range(len(_CORN_T_GRID))
    )
    if t <= _CORN_T_GRID[0]:
        # Ниже сетки: AST растёт вдвое на каждые 5.6 °C охлаждения.
        return row[0] * 2.0 ** ((_CORN_T_GRID[0] - t) / 5.6)
    if t >= _CORN_T_GRID[-1]:
        # Выше сетки: AST падает вдвое на каждые 5.6 °C нагрева.
        return row[-1] / 2.0 ** ((t - _CORN_T_GRID[-1]) / 5.6)
    return _interp1d(_CORN_T_GRID, row, t)


def safe_boundary_temp(grain_type: str, moisture: float) -> float:
    """T (°C), ниже которой партия вне зоны порчи 5-6 мес (CGC)."""
    (t1, w1), (t2, w2) = SAFE_BOUNDARY[grain_type]
    k = (moisture - w1) / (w2 - w1)
    return t1 + k * (t2 - t1)


def critical_temp(grain_type: str, moisture: float, horizon_days: float = 30.0) -> float:
    """Критическая T (°C): при ней партия достигает 0.5 % DML за `horizon_days`.

    Решается бисекцией из ast_days(); физический смысл — порог тревоги
    самосогревания. Горизонт 30 сут (подобран на синтетике как баланс
    recall/lead time при FP ≤ 2/силосо-месяц). Результат ограничен 5..45 °C.
    """
    lo, hi = -5.0, 60.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if ast_days(grain_type, mid, moisture) > horizon_days:
            lo = mid
        else:
            hi = mid
    return max(5.0, min(45.0, 0.5 * (lo + hi)))


@dataclass(frozen=True)
class PhysicsParams:
    """Параметры тепловой модели зерновой массы."""

    q10: float = 2.5
    t_ref: float = 20.0
    relax_rate: float = 0.0015  # 1/ч, теплообмен с воздухом
    diffusion: float = 0.02  # доля переноса между соседними слоями
    q0: float = 0.07  # базовый прирост °C/ч при t_ref
    moisture_gain: float = 0.35  # множитель роста при влажности выше безопасной
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
