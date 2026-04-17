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

## Запуск

```bash
pip install -r requirements.txt
PYTHONPATH=. python3 src/prepare_initial_data.py
PYTHONPATH=. python3 src/run_eda.py
python3 run.py -mode update
python3 run.py -mode inference -file ./path/to/file.csv
python3 run.py -mode summary
```
