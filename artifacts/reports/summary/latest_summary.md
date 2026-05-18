# Monitoring Summary

Generated at: 2026-05-17T20:03:25

## Data Quality

- Processed batches: 8
- Latest batch: `batch_0008`
- Latest quality acceptable: `True`

## Model Validation

- Validation runs: 5
- Latest batch: `batch_0008`
- Best params: `{'C': 0.5, 'class_weight': 'balanced', 'max_iter': 1000, 'solver': 'liblinear'}`
- Best F1: `0.5422`

## Training

- Training runs: 6
- Latest accumulated rows: `30000`
- Encoded feature count: `134`
- Training F1: `0.5671`

## Model Registry

- Stored model versions: 5
- Current best model: `v005`
- Best model type: `logistic_regression`
- Best params: `{'C': 0.5, 'class_weight': 'balanced', 'max_iter': 1000, 'solver': 'liblinear'}`
- Best validation F1: `0.5422`

## Performance

- Performance records: 17
- Latest stage: `update`
- Duration seconds: `33.5891`

## Drift

- Drift records: 5
- Latest batch: `batch_0008`
- Target drift: `0.2172`
- Numeric feature drift:
  - `price_sum` standardized mean shift: `0.0484`
  - `freight_value_sum` standardized mean shift: `0.0186`
  - `items_count` standardized mean shift: `0.0216`
  - `payment_value_sum` standardized mean shift: `0.0490`

## Model Drift

- Model drift records: 4
- Latest version: `v005`
- Metric drop: `-0.0270`
- Drift flag: `False`
