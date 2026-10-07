"""Экономическая модель: предотвращённые потери в тоннах и рублях."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from grain_guard.evaluation.metrics import SystemScore


@dataclass
class EconomicsConfig:
    tonnes_per_silo: float = 5000.0
    price_per_tonne: float = 15000.0
    loss_fraction_operator: float = 0.12
    loss_fraction_system: float = 0.03
    cost_per_false_alarm: float = 3000.0
    monitoring_cost_per_silo_season: float = 2000.0
    season_months: float = 6.0


def compute(
    score: SystemScore,
    cfg: EconomicsConfig | None = None,
) -> dict[str, float]:
    """Оценить экономический эффект по результатам бенчмарка."""
    cfg = cfg or EconomicsConfig()

    n_prevented = score.n_incidents * score.recall * score.earlier_than_operator
    saved_per_incident = (
        cfg.tonnes_per_silo
        * cfg.price_per_tonne
        * (cfg.loss_fraction_operator - cfg.loss_fraction_system)
    )
    gross_saved = n_prevented * saved_per_incident
    tonnes_saved = n_prevented * cfg.tonnes_per_silo * (
        cfg.loss_fraction_operator - cfg.loss_fraction_system
    )

    n_silos = max(score.n_incidents / max(score.recall, 1e-9), 1e-9) if score.recall else 1.0
    false_alarms = score.fp_per_silo_month * n_silos * cfg.season_months
    false_alarm_cost = false_alarms * cfg.cost_per_false_alarm
    monitoring_cost = n_silos * cfg.monitoring_cost_per_silo_season

    net = gross_saved - false_alarm_cost - monitoring_cost
    return {
        "incidents_prevented": round(n_prevented, 2),
        "tonnes_saved": round(tonnes_saved, 1),
        "gross_saved_rub": round(gross_saved, 0),
        "false_alarm_cost_rub": round(false_alarm_cost, 0),
        "monitoring_cost_rub": round(monitoring_cost, 0),
        "net_benefit_rub": round(net, 0),
        **{k: v for k, v in asdict(cfg).items()},
    }


def sensitivity(
    score: SystemScore, cfg: EconomicsConfig | None = None
) -> pd.DataFrame:
    """Чувствительность net_benefit к цене зерна и потерям оператора."""
    base = cfg or EconomicsConfig()
    rows = []
    for price in (10000, 15000, 20000, 25000):
        for loss_op in (0.08, 0.12, 0.18):
            c = EconomicsConfig(**{**asdict(base), "price_per_tonne": price,
                                   "loss_fraction_operator": loss_op})
            res = compute(score, c)
            rows.append(
                {
                    "price_per_tonne": price,
                    "loss_fraction_operator": loss_op,
                    "net_benefit_rub": res["net_benefit_rub"],
                }
            )
    return pd.DataFrame(rows)


def to_markdown(result: dict[str, float], sens: pd.DataFrame) -> str:
    lines = [
        "# Экономика: предотвращённые потери",
        "",
        "Все значения рассчитаны из конфига; исходные допущения приведены ниже.",
        "",
        "## Итог",
        "",
        f"- Предотвращено инцидентов: **{result['incidents_prevented']}**",
        f"- Предотвращено потерь: **{result['tonnes_saved']} т**",
        f"- Эффект до затрат: **{result['gross_saved_rub']:,.0f} ₽**",
        f"- Стоимость ложных тревог: {result['false_alarm_cost_rub']:,.0f} ₽",
        f"- Стоимость мониторинга: {result['monitoring_cost_rub']:,.0f} ₽",
        f"- **Чистый эффект: {result['net_benefit_rub']:,.0f} ₽**",
        "",
        "## Допущения",
        "",
        "| Параметр | Значение |",
        "| --- | --- |",
        f"| Тонн на силос | {result['tonnes_per_silo']:,.0f} |",
        f"| Цена зерна, ₽/т | {result['price_per_tonne']:,.0f} |",
        f"| Потери при позднем обнаружении (оператор) | {result['loss_fraction_operator']:.0%} |",
        f"| Потери при раннем обнаружении (система) | {result['loss_fraction_system']:.0%} |",
        f"| Стоимость одной ложной тревоги, ₽ | {result['cost_per_false_alarm']:,.0f} |",
        f"| Стоимость мониторинга на силос/сезон, ₽ | {result['monitoring_cost_per_silo_season']:,.0f} |",
        f"| Длительность сезона, мес | {result['season_months']} |",
        "",
        "## Чувствительность (net benefit, ₽)",
        "",
        sens.to_markdown(index=False),
        "",
    ]
    return "\n".join(lines)
