"""End-to-end конвейер: синтетика → физика → ML → бенчмарк → экономика."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC
from pathlib import Path

import pandas as pd

from grain_guard.evaluation.metrics import SystemScore, compare, score_system
from grain_guard.features.build import build_features
from grain_guard.models import ModelBundle, get_backend
from grain_guard.models.train import split_by_group
from grain_guard.physics.model import compute_alerts
from grain_guard.reports import economics
from grain_guard.synth.generator import SynthConfig, generate, validate


@dataclass
class PipelineResult:
    df: pd.DataFrame
    benchmark: pd.DataFrame
    bundles: dict[str, ModelBundle]
    scores: list[SystemScore] = field(default_factory=list)
    econ: dict[str, float] = field(default_factory=dict)
    sensitivity: pd.DataFrame = field(default_factory=pd.DataFrame)
    ml_name: str | None = None


def run(
    n_silos: int = 1000,
    seed: int = 42,
    test_size: float = 0.3,
    out_dir: str | Path | None = None,
    write_reports: bool = True,
    df: pd.DataFrame | None = None,
) -> PipelineResult:
    """Полный прогон. Если передан df (данные реального объекта), синтетика
    пропускается; при отсутствии положительной разметки ML не обучается
    и отчёт формируется по физической модели."""
    if df is None:
        df = generate(SynthConfig(n_silos=n_silos, seed=seed))
        validate(df)
    df = df.sort_values(["silo_id", "timestamp"]).reset_index(drop=True)

    phys = compute_alerts(df)
    X, y, meta = build_features(df)

    has_labels = int(y.sum()) > 0
    if has_labels:
        tr, te = split_by_group(X, y, meta, test_size=test_size, seed=seed)
        bundles = get_backend().fit_baselines(
            X.iloc[tr].reset_index(drop=True),
            y.iloc[tr].reset_index(drop=True),
            X.iloc[te].reset_index(drop=True),
            y.iloc[te].reset_index(drop=True),
            seed=seed,
        )

        best = bundles["best"]
        proba = best.predict_proba(X.iloc[te].reset_index(drop=True))
        ml_alert = proba >= best.threshold

        df_test = df.iloc[te].reset_index(drop=True)
        phys_alert = phys["alert"].to_numpy()[te]

        scores = [
            score_system(df_test, phys_alert, "physics"),
            score_system(df_test, ml_alert, f"ml_{best.metrics.get('selected', 'best')}"),
        ]
        ml_name = None
    else:
        bundles = {}
        scores = [score_system(df, phys["alert"].to_numpy(), "physics")]
        ml_name = "ml_пропущен (нет размеченных инцидентов)"

    benchmark = compare(scores)

    econ = economics.compute(scores[-1])
    sens = economics.sensitivity(scores[-1])

    result = PipelineResult(
        df=df,
        benchmark=benchmark,
        bundles=bundles,
        scores=scores,
        econ=econ,
        sensitivity=sens,
        ml_name=ml_name,
    )
    if write_reports and out_dir is not None:
        _write_reports(result, out_dir)
    return result


def _git_commit() -> str:
    """Короткий hash коммита, если репозиторий доступен; иначе 'unknown'."""
    import subprocess

    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            ).stdout.strip()
            or "unknown"
        )
    except Exception:
        return "unknown"


def _write_reports(result: PipelineResult, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    from datetime import datetime

    from grain_guard import __version__

    header = (
        f"Сгенерировано: {datetime.now(UTC):%Y-%m-%d %H:%M UTC} | "
        f"grain-guard {__version__} (commit: {_git_commit()})"
    )

    lines = [
        "# Бенчмарк: система vs оператор",
        "",
        header,
        "",
        f"Силосов в прогоне: {result.df['silo_id'].nunique()}",
        f"Инцидентов: {int(result.df.loc[result.df['incident'] == 1, 'silo_id'].nunique())}",
        "",
        result.benchmark.to_markdown(index=False),
        "",
        "Метрики: детекция — тревога в окне [incident_start − 72 ч; момент обнаружения",
        "оператором (порог 32 °C) либо конец траектории]; lead time — опережение оператора;",
        "fp_per_silo_month — ложные срабатывания на силос в месяц на нормальных партиях.",
        "",
    ]
    (out / "benchmark.md").write_text("\n".join(lines), encoding="utf-8")
    if result.ml_name:
        with (out / "benchmark.md").open("a", encoding="utf-8") as f:
            f.write(f"\nML: {result.ml_name}\n")
    econ_md = economics.to_markdown(result.econ, result.sensitivity)
    econ_title = "# Экономика: предотвращённые потери\n"
    econ_md = econ_md.replace(econ_title, econ_title + "\n" + header + "\n", 1)
    (out / "economics.md").write_text(econ_md, encoding="utf-8")

    best = result.bundles.get("best")
    if best is not None and best.importance is not None:
        (out / "feature_importance.md").write_text(
            "# Важность признаков (лучшая модель)\n\n" + best.importance.to_markdown(index=False),
            encoding="utf-8",
        )
