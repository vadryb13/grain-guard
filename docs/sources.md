# Источники данных и параметров

Все параметры моделей grain-guard обязаны ссылаться на источник из этого файла
либо иметь пометку «подобрано на синтетике» в docstring. Дата доступа ко всем
URL: 2026-10-07.

## 1. Критические кривые T ↔ W (Этап 2.2)

### 1.1 Canadian Grain Commission — Safe Storage Guidelines

- URL: https://www.grainscanada.gc.ca/en/grain-quality/manage/manage-storage-prevent-infestations/prevent-spoilage.html
- Что взято: опорные точки границы порчи за 5–6 месяцев хранения:
  - пшеница: 0–28 °C ↔ 18–10 % (без порчи до 6 мес)
  - ячмень: 5–20 °C ↔ 18–10 %
  - овёс: 0–21 °C ↔ 17–8 %
- Использование: `SAFE_BOUNDARY` в `src/grain_guard/config.py`; граница «партия
  выше кривой → вероятен инцидент в течение сезона» в генераторе.
- Кредит: фак-лист Cereal Research Centre, Agriculture and Agri-Food Canada
  (Dr. Vincent Hervet), воспроизведён Canadian Grain Commission с разрешения.

### 1.2 ASABE Standard D535 — Allowable Storage Time for Shelled Corn (0.5 % DML)

- Вторичная публикация (таблица целиком): Purdue University Extension,
  «Shelled Corn — Maximum Allowable Storage Time»,
  https://extension.entm.purdue.edu/grainlab/content/pdf/ShelledCorn.pdf
- Первоисточник данных: Steele, J.L., Saul, R.A., Hukill, W.V. (1969).
  Deterioration of shelled corn as measured by carbon dioxide production.
  Trans. ASAE 12(5):685–689. Модель потерь сухого вещества (DML) от T и W.
- Что взято: полная сетка AST (дни) для T = 1.7…23.9 °C (35…75 °F) и
  W = 16…30 % → `CORN_AST_DAYS` в `config.py`; функция `ast_days()`.
- Допущение: для пшеницы/ячменя AST-таблица кукурузы масштабируется
  коэффициентом `AST_SCALE` из отношения CGC-границ к corn-границе (п. 1.1);
  коэффициенты вычислены и помечены в коде как оценка, а не измерение.

## 2. Реальные сенсорные данные (Этапы 1, 5)

### 2.1 intellidry-public-datasets

- URL: https://github.com/intellidry/intellidry-public-datasets
- Что это: сырые логи сушки зерна (овёс ×2, гречиха, горох, 2023): 6 температурных
  датчиков в массе зерна, 2 датчика влажности воздуха, ручные замеры влажности
  зерна каждые 10 мин; long-format CSV (~1 строка/сек, сессии ~14 ч, T 17.5–84.6 °C).
- Использование: калибровка шума датчиков и межслойной диффузии генератора;
  «грязная» фикстура адаптера реальных данных (Этап 5.4).
- Ограничение: сушка, а не хранение; не годится для обучения детектора
  самосогревания.

### 2.2 Просо в металлических силосах (послойная телеметрия)

- DOI: 10.5281/zenodo.3263761 — «Research of the grain temperature of millet
  stored in metal silos» (2019). Послойные температурные кривые в силосе.
- Использование: референс послойной динамики для валидации генератора
  (оцифровка кривых из PDF — в бэклоге).

## 3. Метеоданные (Этап 1.2)

- ERA5-Land hourly, Copernicus Climate Data Service:
  https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land
- Использование: реальная сезонная T воздуха для генератора (в бэклоге;
  сейчас синусоида с параметрами, помеченными как допущение).

## 4. Теплофизика зерновой массы (Этап 2.1)

- ASAE D243 «Thermal Properties of Grain and Grain Products» — таблицы
  теплопроводности/теплоёмкости по культурам (в бэклоге; текущие
  `relax_rate`/`diffusion` — помечены «подобрано на синтетике»).

## 5. Поисковая сводка (для протокола)

- Открытая телеметрия хранения зерна с размеченными инцидентами
  самосогревания не опубликована нигде (Zenodo, DataCite, Dryad, Kaggle,
  GitHub, IEEE DataPort — проверено 2026-10-07). Единственный путь к таким
  данным — только закрытая телеметрия реального объекта.
- Ближайшие соседи по выдаче: почвенное дыхание (Q10-методологии, напр.
  10.5281/zenodo.7106267), симуляция хранения пшеницы в Тунисе
  (doi:10.54536/ajaset.v7i1.1056, DML ≈ 0.15 %/мес при 24 °C / 12.2 % —
  калибровочная точка), модель вентиляции (doi:10.15587/1729-4061.2022.253038).
