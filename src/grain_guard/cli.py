"""CLI grain-guard."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from grain_guard import __version__
from grain_guard.adapters.ingest import ingest_csv, load_config
from grain_guard.pipeline import run as run_pipeline
from grain_guard.synth.generator import SynthConfig, generate, validate


def _cmd_synth(args: argparse.Namespace) -> int:
    df = generate(SynthConfig(n_silos=args.n, seed=args.seed))
    validate(df)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"synthetic_n{args.n}_seed{args.seed}.csv"
    df.to_csv(path, index=False)
    print(f"Сгенерировано {len(df):,} строк, {df['silo_id'].nunique()} силосов -> {path}")
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    result = run_pipeline(
        n_silos=args.n, seed=args.seed, out_dir=args.out, write_reports=True
    )
    print(result.benchmark.to_string(index=False))
    print()
    print(f"Чистый экономический эффект: {result.econ['net_benefit_rub']:,.0f} ₽")
    print(f"Отчёты в: {args.out}")
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    res = ingest_csv(args.source, args.config)
    if not res.ok:
        print("Данные не прошли валидацию:", file=sys.stderr)
        for e in res.errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    assert res.data is not None
    print(f"ОК: {len(res.data):,} строк, {res.data['silo_id'].nunique()} силосов")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        res.data.to_csv(out, index=False)
        print(f"Сохранено -> {out}")
    return 0


def _cmd_economics(args: argparse.Namespace) -> int:
    from grain_guard.reports import economics

    cfg_data = load_config(args.config) if args.config else {}
    cfg = economics.EconomicsConfig(**{k: v for k, v in cfg_data.items()})
    print("Экономика считается по результатам benchmark. Запустите 'grain-guard benchmark'.")
    print(cfg)
    return 0


def _cmd_demo(_args: argparse.Namespace) -> int:
    out = Path("reports")
    result = run_pipeline(n_silos=1000, seed=42, out_dir=out, write_reports=True)
    print(result.benchmark.to_string(index=False))
    print()
    print(f"Net benefit: {result.econ['net_benefit_rub']:,.0f} ₽")
    print(f"Отчёты: {out}/benchmark.md, {out}/economics.md, {out}/feature_importance.md")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="grain-guard", description="Прогноз самосогревания зерна")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("synth", help="Сгенерировать синтетические данные")
    s.add_argument("--n", type=int, default=1000)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--out", default="data/synthetic")
    s.set_defaults(func=_cmd_synth)

    b = sub.add_parser("benchmark", help="Полный прогон и бенчмарк vs оператор")
    b.add_argument("--n", type=int, default=1000)
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--out", default="reports")
    b.set_defaults(func=_cmd_benchmark)

    i = sub.add_parser("ingest", help="Загрузить CSV реального элеватора")
    i.add_argument("--source", required=True)
    i.add_argument("--config", default=None)
    i.add_argument("--out", default=None)
    i.set_defaults(func=_cmd_ingest)

    e = sub.add_parser("economics", help="Показать/проверить параметры экономики")
    e.add_argument("--config", default=None)
    e.set_defaults(func=_cmd_economics)

    d = sub.add_parser("demo", help="End-to-end демо с отчётами")
    d.set_defaults(func=_cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
