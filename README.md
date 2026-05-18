# eSalesAnalysys
Репозиторий для задания по Практикуму на ЭВМ 2026

MVP MLOps-конвейера на датасете Olist для предсказания длительной доставки заказа.

Target:
- `delivery_time_days = order_delivered_customer_date - order_purchase_timestamp`
- `is_long_delivery = 1`, если `delivery_time_days > Q75`

Что реализовано:
- потоковая эмуляция батчей;
- quality checks, cleaning и EDA;
- подготовка признаков;
- обучение, валидация и выбор лучшей модели;
- versioned model storage;
- `update`, `inference`, `summary`.

## Документация

Подробное описание проекта и распределение баллов находятся в директории [`doc`](doc):
- [`doc/task.md`](doc/task.md) - постановка задачи, данные, pipeline, модель и артефакты;
- [`doc/grade.md`](doc/grade.md) - ожидаемые баллы по участникам.

## Запуск

```bash
pip install -r requirements.txt
PYTHONPATH=. python3 src/prepare_initial_data.py
PYTHONPATH=. python3 src/run_eda.py
python3 run.py -mode update
python3 run.py -mode inference -file ./path/to/file.csv
python3 run.py -mode summary
```

## CI/CD

Запускается при:
- `push`/`pull_request` в `main`;
- ежедневном `cron`;
- ручном запуске через `workflow_dispatch`.

Что делает:
- запускает `python run.py -mode update`;
- строит `summary`;
- запускает smoke-тесты;
- сохраняет артефакты.

Actions сохраняет:
- `training-logs` - лог обучения;
- `runnable-model-bundle` - модель, препроцессор, registry и metadata;
- `data-collector-state` - состояние потока и накопленный датасет;
- `monitoring-dashboard` - summary и истории качества/обучения/drift.

Для продолжения инкрементального обучения между запусками используется cache.
инкрементальный подход: каждый запуск обрабатывает следующий батч данных, обновляет модель и сохраняет состояние для следующих запусков.
