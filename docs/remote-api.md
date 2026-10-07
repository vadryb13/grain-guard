# Удалённый сервис моделей (VPS)

Тяжёлые операции (обучение, длинные прогоны) выполняются на VPS, лёгкий код
(синтез, фичи, физика, валидация, отчёты) — локально. См. `~/AGENTS.md`.

## Схема

```
локально: pipeline/CLI ──HTTP──> VPS: FastAPI (grain-guard serve)
          parquet по wire       2 CPU / 3.8 ГиБ RAM / uv sync --extra service
```

Локальный код и сервер используют одну и ту же версию пакета `grain-guard`,
поэтому артефакт моделей (joblib) совместим в обе стороны.

## Запуск сервиса на VPS

```bash
~/projects/sync-code-to-vps.sh grain-guard        # пробный прогон, проверить список
~/projects/sync-code-to-vps.sh grain-guard --apply
ssh vps -t 'tmux new -A -s gg-api'
cd ~/projects/grain-guard
uv sync --extra service
uv run grain-guard serve --host 127.0.0.1 --port 8000
# отсоединение: Ctrl+B, D
```

Сервис слушает `127.0.0.1` — наружу порт не открывается (UFW не трогаем).
Доступ локально — через SSH-туннель:

```bash
ssh -f -N -L 8000:127.0.0.1:8000 vps
```

При смене сервера/ресурсов меняется только адрес — код не трогается.

## Подключение локального кода

Любой из вариантов (приоритет: env → конфиг → локальный бэкенд):

```bash
export GRAIN_GUARD_API_URL=http://127.0.0.1:8000          # + опционально
export GRAIN_GUARD_API_TOKEN=<секрет с VPS .env>          # GRAIN_GUARD_API_TOKEN
```

или `configs/remote.yaml` (в git не коммитить токен):

```yaml
base_url: http://127.0.0.1:8000
token: secret
```

Без этих настроек `grain-guard benchmark` обучает модели локально, как раньше.

## API

- `GET /health` — статус и версия.
- `POST /v1/fit` — multipart: `X_train`, `y_train`, `X_test`, `y_test` (parquet), `?seed=` → joblib-артефакт всех бандлов; метрики лучшей модели в заголовке `X-Metrics`.
- `POST /v1/predict` — multipart: `model` (joblib), `X` (parquet), `?bundle_name=best` → `{"proba": [...]}`.
- `POST /v1/models` / `GET /v1/models` — сохранить/забрать артефакт на сервере.

Проверка: `curl http://127.0.0.1:8000/health`.
