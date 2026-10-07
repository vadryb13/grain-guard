"""Метрики и бенчмарк «система vs оператор»."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DETECTION_WINDOW_H = 168.0


@dataclass
class SystemScore:
    name: str
    n_incidents: int
    detected: int
    recall: float
    median_lead_h: float
    mean_lead_h: float
    earlier_than_operator: float
    fp_per_silo_month: float
    operator_missed: int = 0

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "system": self.name,
            "incidents": self.n_incidents,
            "detected": self.detected,
            "recall": round(self.recall, 3),
            "median_lead_h": round(self.median_lead_h, 1),
            "mean_lead_h": round(self.mean_lead_h, 1),
            "earlier_than_operator": round(self.earlier_than_operator, 3),
            "fp_per_silo_month": round(self.fp_per_silo_month, 2),
            "operator_missed": self.operator_missed,
        }


def _serialize(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for c in ("incident_start", "operator_detect"):
        if not np.issubdtype(d[c].dtype, np.datetime64):
            d[c] = pd.to_datetime(d[c])
    return d


def score_system(
    df: pd.DataFrame,
    alert: np.ndarray,
    name: str,
    window_h: float = DETECTION_WINDOW_H,
) -> SystemScore:
    """Оценить систему по посигнальным тревогам (выровнены с df по строкам)."""
    data = _serialize(df).reset_index(drop=True)
    data["_alert"] = np.asarray(alert, dtype=bool)

    incidents = data[data["incident"] == 1].drop_duplicates("silo_id")
    leads: list[float] = []
    earlier = 0
    detected = 0
    operator_missed = 0

    for row in incidents.itertuples():
        start = row.incident_start
        sub = data[data["silo_id"] == row.silo_id]
        window = sub[
            (sub["timestamp"] >= start - pd.Timedelta(hours=window_h))
            & (sub["timestamp"] <= start)
        ]
        hits = window[window["_alert"]]
        if hits.empty:
            continue
        detected += 1
        det = hits["timestamp"].iloc[0]
        op = row.operator_detect
        if pd.isna(op):
            operator_missed += 1
            continue
        leads.append((op - det).total_seconds() / 3600.0)
        if det < op:
            earlier += 1

    n_inc = int(incidents.shape[0])
    recall = detected / n_inc if n_inc else 0.0
    median_lead = float(np.median(leads)) if leads else 0.0
    mean_lead = float(np.mean(leads)) if leads else 0.0
    earlier_frac = earlier / len(leads) if leads else 0.0

    fp = _false_positive_rate(data, name)
    return SystemScore(
        name=name,
        n_incidents=n_inc,
        detected=detected,
        recall=recall,
        median_lead_h=median_lead,
        mean_lead_h=mean_lead,
        earlier_than_operator=earlier_frac,
        fp_per_silo_month=fp,
        operator_missed=operator_missed,
    )


def _false_positive_rate(data: pd.DataFrame, _name: str) -> float:
    normal = data[data["incident"] == 0].sort_values(["silo_id", "timestamp"])
    total_hours = 0
    episodes = 0
    for _silo, g in normal.groupby("silo_id", sort=False):
        a = g["_alert"].to_numpy(dtype=bool)
        total_hours += len(a)
        if a.any():
            onsets = np.sum(a[1:] & ~a[:-1]) + int(a[0])
            episodes += int(onsets)
    months = total_hours / (24 * 30)
    return episodes / months if months else 0.0


def compare(scores: list[SystemScore]) -> pd.DataFrame:
    return pd.DataFrame([s.as_dict() for s in scores])
