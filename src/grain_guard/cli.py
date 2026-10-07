"""CLI grain-guard."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from grain_guard import __version__
from grain_guard.adapters.ingest import ingest_csv, load_config
from grain_guard.pipeline import run as run_pipeline


def _cmd_benchmark(args: argparse.Namespace) -> int:
    """Прогон на реальных данных: ingest -> валидация -> физика (+ML при разметке)."""
    if not args.source:
        print(
            "Укажите --source <csv реального объекта> (+ --config при маппинге). "
            "Работа без синтетических данных.",
            file=sys.stderr,
        )
        return 2
    res = ingest_csv(args.source, args.config)
    if not res.ok:
        print("Данные не прошли валидацию:", file=sys.stderr)
        for e in res.errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    assert res.data is not None
    df = res.data
    print(f"Ingest: ОК, {len(df):,} строк, {df['silo_id'].nunique()} силосов из {args.source}")
    result = run_pipeline(df=df, out_dir=args.out, write_reports=True)
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


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("Сервер не установлен: uv sync --extra service", file=sys.stderr)
        return 1

    from grain_guard.models import get_backend

    backend = get_backend()
    if not hasattr(backend, "health"):
        print(
            "Внимание: GRAIN_GUARD_API_URL не задан — сервис всё равно запустится.",
            file=sys.stderr,
        )
    uvicorn.run(
        "grain_guard.models.server:create_app",
        factory=True,
        host=args.host,
        port=args.port,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="grain-guard", description="Прогноз самосогревания зерна")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("benchmark", help="Полный прогон и бенчмарк vs оператор")
    b.add_argument("--out", default="reports")
    b.add_argument("--source", default=None, help="CSV реального объекта (ingest + валидация)")
    b.add_argument("--config", default=None, help="YAML с column_map для --source")
    b.set_defaults(func=_cmd_benchmark)

    i = sub.add_parser("ingest", help="Загрузить CSV реального элеватора")
    i.add_argument("--source", required=True)
    i.add_argument("--config", default=None)
    i.add_argument("--out", default=None)
    i.set_defaults(func=_cmd_ingest)

    e = sub.add_parser("economics", help="Показать/проверить параметры экономики")
    e.add_argument("--config", default=None)
    e.set_defaults(func=_cmd_economics)

    sv = sub.add_parser("serve", help="HTTP-сервис моделей (для VPS)")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.set_defaults(func=_cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
