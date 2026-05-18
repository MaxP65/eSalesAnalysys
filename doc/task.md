# Описание задачи

## Цель проекта

Проект реализует MVP MLOps-конвейера для потоковых табличных данных:

- получение новых данных батчами;
- проверка качества и очистка;
- обучение и валидация модели;
- выбор лучшей версии модели;
- inference на внешнем CSV;
- summary-отчет и сохранение артефактов.

Основные команды:

```bash
python3 run.py -mode update
python3 run.py -mode inference -file ./path/to/file.csv
python3 run.py -mode summary
```

## Данные и target

Используется датасет Olist. Из исходных таблиц формируется order-level датасет:

```text
data/processed/olist_ml_dataset.csv
```

Временная колонка:

```text
order_purchase_timestamp
```

Она используется для сортировки данных и эмуляции потока.

Прогнозируется длительная доставка заказа:

```text
delivery_time_days = order_delivered_customer_date - order_purchase_timestamp
is_long_delivery = 1, если delivery_time_days > Q75
```

Задача: бинарная классификация `is_long_delivery`.

## Pipeline

Основной сценарий `update`:

1. Получить следующий батч данных.
2. Сохранить сырой батч в `artifacts/raw_storage`.
3. Рассчитать metadata батча в `artifacts/batch_metadata`.
4. Проверить data quality.
5. Очистить данные и сохранить батч в `artifacts/cleaned_storage`.
6. Обновить накопленный датасет `data/processed/accumulated_training_data.csv`.
7. Выполнить валидацию и подбор гиперпараметров.
8. Обучить модель и обновить `model_registry.json`.
9. Сохранить состояние потока и истории мониторинга.

## Data Quality

Проверка качества реализована в:

```text
src/data_analysis/quality_checker.py
```

Проверяются:

- обязательные колонки;
- пропуски;
- дубли;
- корректность временной колонки;
- допустимые значения target.

Параметры задаются в `configs/config.yaml`:

```yaml
max_missing_ratio: 0.35
max_duplicate_ratio: 0.05
drop_columns_missing_ratio: 0.60
allowed_target_values: [0, 1]
```

Отчеты сохраняются в:

```text
artifacts/batch_metadata
artifacts/history/data_quality_history.jsonl
```

Data drift сохраняется в:

```text
artifacts/history/drift_history.jsonl
```

## Признаки и предобработка

Предобработка реализована в:

```text
src/data_preparation/preprocessor.py
```

Используется `ColumnTransformer`:

- числовые признаки: `SimpleImputer`;
- категориальные признаки: `SimpleImputer` + `OneHotEncoder`.

Из обучения исключаются leakage и служебные признаки:

- `order_id`;
- `customer_id`;
- `order_status`;
- `order_purchase_timestamp`;
- `delivery_time_days`.

Препроцессор сохраняется в:

```text
artifacts/models/preprocessor_latest.joblib
```

## Модели и гиперпараметры

Рассматриваются две модели:

- `DecisionTreeClassifier`;
- `LogisticRegression`.

Основная метрика:

```text
f1
```

Гиперпараметры задаются в `configs/config.yaml`.

Для дерева решений:

```yaml
max_depth: [6, 8, 12]
min_samples_split: [2, 10]
min_samples_leaf: [5, 10]
class_weight: [null, balanced]
```

Для логистической регрессии:

```yaml
C: [0.5, 1.0, 2.0]
class_weight: [null, balanced]
max_iter: [1000]
solver: [liblinear]
```

## Валидация и хранение моделей

Валидация реализована через `TimeSeriesSplit`:

```yaml
n_splits: 4
```

История валидации:

```text
artifacts/history/validation_history.jsonl
```

Модели и registry:

```text
artifacts/models/model_latest.joblib
artifacts/models/preprocessor_latest.joblib
artifacts/models/model_registry.json
artifacts/models/training_metadata.json
```

Для интерпретации сохраняются `feature_importance_*.csv`.

Model drift:

```text
artifacts/history/model_drift_history.jsonl
```

## Inference и summary

Inference выбирает лучшую модель из registry, применяет ее к CSV и сохраняет файл с колонкой `predict`:

```text
artifacts/reports/predictions
```

Summary формируется командой:

```bash
python3 run.py -mode summary
```

Результат:

```text
artifacts/reports/summary/latest_summary.md
```

## CI/CD и артефакты

Workflow:

```text
.github/workflows/ml-pipeline.yml
```

Он запускает `update`, строит `summary`, выполняет smoke-тесты и сохраняет GitHub Actions artifacts:

- `training-logs`;
- `runnable-model-bundle`;
- `data-collector-state`;
- `monitoring-dashboard`.

Smoke-тесты:

```bash
PYTHONPATH=. python3 -m unittest discover -s tests
```
