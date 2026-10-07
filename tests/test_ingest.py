"""Тесты адаптера реальных данных: человекочитаемые ошибки на «грязном» CSV (Этап 5)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from grain_guard.adapters.ingest import ingest_csv
from grain_guard.config import LAYER_COLUMNS

_DIRTY_ROWS = """silo_1,2024-01-01 00:00,5,13.0,wheat,12.0,11.5,11.0,10.5,10.0,9.5
silo_1,2024-01-01 00:00,5,13.0,wheat,12.1,11.6,11.1,10.6,10.1,9.6
silo_1,2024-01-01 01:00,5,13.1,wheat,12.2,11.7,11.2,10.7,10.2,9.7
silo_2,not-a-date,6,14.0,barley,13.0,12.5,12.0,11.5,11.0,10.5
silo_3,2024-01-01 00:00,7,30.0,rye,15.0,14.0,13.0,12.0,11.0,10.0
silo_4,2024-01-01 00:00,7,12.0,corn,999.0,10.0,9.0,8.0,7.0,6.0
"""

DIRTY_CSV = ",".join(
    ["silo_id", "timestamp", "T_air", "W_moisture", "grain_type", *LAYER_COLUMNS]
) + "\n" + _DIRTY_ROWS


@pytest.fixture(scope="module")
def dirty_csv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    p = tmp_path_factory.mktemp("raw") / "dirty.csv"
    p.write_text(DIRTY_CSV, encoding="utf-8")
    return p


def test_dirty_fixture_reports_every_problem(dirty_csv: Path) -> None:
    res = ingest_csv(dirty_csv)
    assert not res.ok
    text = "\n".join(res.errors)
    assert "Дублирующиеся timestamp" in text
    assert "Некорректный timestamp" in text
    assert "9..25" in text
    assert "rye" in text
    assert "-5..60" in text


def test_clean_fixture_passes(tmp_path: Path) -> None:
    p = tmp_path / "clean.csv"
    df = pd.DataFrame(
        {
            "silo_id": ["silo_1"] * 5,
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="h"),
            "T_air": [5.0] * 5,
            "W_moisture": [13.0] * 5,
            **{c: [10.0 + i * 0.1] * 5 for i, c in enumerate(LAYER_COLUMNS)},
            "grain_type": ["wheat"] * 5,
        }
    )
    df.to_csv(p, index=False)
    res = ingest_csv(p)
    assert res.ok, res.errors
    assert res.data is not None
    assert res.data["storage_day"].tolist() == [0, 0, 0, 0, 0]


def test_column_mapping_via_config(tmp_path: Path) -> None:
    p = tmp_path / "mapped.csv"
    pd.DataFrame(
        {
            "id": ["s1", "s1"],
            "time": pd.date_range("2024-01-01", periods=2, freq="h"),
            "T_vozduha": [5.0, 5.1],
            "W": [13.0, 13.0],
            "T_sl_1": [10.0, 10.1],
            "T_sl_2": [9.5, 9.6],
            "T_sl_3": [9.0, 9.1],
            "T_sl_4": [8.5, 8.6],
            "T_sl_5": [8.0, 8.1],
            "T_sl_6": [7.5, 7.6],
            "tip": ["wheat", "wheat"],
        }
    ).to_csv(p, index=False)
    cfg = tmp_path / "pilot.yaml"
    cfg.write_text(
        "column_map:\n"
        "  id: silo_id\n"
        "  time: timestamp\n"
        "  T_vozduha: T_air\n"
        "  W: W_moisture\n"
        "  tip: grain_type\n"
        + "".join(f"  T_sl_{i}: T_layer_{i}\n" for i in range(1, 7)),
        encoding="utf-8",
    )
    res = ingest_csv(p, cfg)
    assert res.ok, res.errors
    assert res.data is not None
    assert list(res.data["silo_id"].unique()) == ["s1"]
