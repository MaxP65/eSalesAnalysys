from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from statistics import mean

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit


@dataclass
class ModelValidator:
    n_splits: int = 4

    def build_splitter(self) -> TimeSeriesSplit:
        return TimeSeriesSplit(n_splits=self.n_splits)

    def calculate_metrics(self, y_true, y_pred) -> dict[str, float]:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }

    @staticmethod
    def iter_param_grid(param_grid: dict) -> list[dict]:
        keys = list(param_grid.keys())
        values = [param_grid[key] for key in keys]
        return [dict(zip(keys, combination)) for combination in product(*values)]

    def evaluate_param_candidates(self, data, preprocessor, trainer, model_candidates: dict, time_column: str) -> dict:
        ordered_data = data.sort_values(time_column).reset_index(drop=True)
        splitter = self.build_splitter()
        best_result: dict | None = None

        for model_type, candidate_config in model_candidates.items():
            for params in self.iter_param_grid(candidate_config["param_grid"]):
                fold_metrics: list[dict[str, float]] = []
                for train_index, test_index in splitter.split(ordered_data):
                    train_df = ordered_data.iloc[train_index].copy()
                    test_df = ordered_data.iloc[test_index].copy()

                    train_features, train_target, transformer, _ = preprocessor.fit_transform(train_df)
                    test_features, test_target = preprocessor.transform(test_df, transformer)

                    model = trainer.train(train_features, train_target, model_type=model_type, **params)
                    predictions = model.predict(test_features)
                    fold_metrics.append(self.calculate_metrics(test_target, predictions))

                avg_metrics = {
                    metric_name: float(mean(metric[metric_name] for metric in fold_metrics))
                    for metric_name in fold_metrics[0]
                }
                candidate_result = {
                    "model_type": model_type,
                    "params": params,
                    "fold_metrics": fold_metrics,
                    "avg_metrics": avg_metrics,
                }

                if best_result is None or avg_metrics["f1"] > best_result["avg_metrics"]["f1"]:
                    best_result = candidate_result

        if best_result is None:
            raise ValueError("No validation result was produced.")
        return best_result
