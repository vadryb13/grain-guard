"""Адаптеры источников данных."""

from grain_guard.adapters.ingest import IngestResult, ingest_csv, load_config, validate

__all__ = ["IngestResult", "ingest_csv", "load_config", "validate"]
