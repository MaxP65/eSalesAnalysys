from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class DataQualityChecker:
    required_columns: list[str]
    max_missing_ratio: float
    max_duplicate_ratio: float
    time_column: str | None = None
    target_column: str | None = None
    allowed_target_values: list[int] | None = None

    def evaluate(self, data: pd.DataFrame) -> dict[str, Any]:
        missing_columns = [column for column in self.required_columns if column not in data.columns]
        missing_ratio = data.isna().mean().sort_values(ascending=False).to_dict() if not data.empty else {}
        duplicate_ratio = float(data.duplicated().mean()) if not data.empty else 0.0
        high_missing_columns = [
            column for column, ratio in missing_ratio.items() if ratio > self.max_missing_ratio
        ]
        type_profile = {column: str(dtype) for column, dtype in data.dtypes.items()}
        invalid_time_rows = 0
        invalid_target_count = 0
        target_distribution: dict[str, float] = {}

        if self.target_column and self.target_column in data.columns and not data.empty:
            target_series = data[self.target_column]
            target_distribution = {
                str(key): float(value)
                for key, value in target_series.value_counts(normalize=True, dropna=False).sort_index().items()
            }
            if self.allowed_target_values is not None:
                invalid_mask = target_series.notna() & ~target_series.isin(self.allowed_target_values)
                invalid_target_count = int(invalid_mask.sum())

        if self.time_column and self.time_column in data.columns and not data.empty:
            parsed_timestamps = pd.to_datetime(data[self.time_column], errors="coerce")
            invalid_time_rows = int(parsed_timestamps.isna().sum())

        is_acceptable = (
            not missing_columns
            and duplicate_ratio <= self.max_duplicate_ratio
            and not high_missing_columns
            and not data.empty
            and invalid_time_rows == 0
            and invalid_target_count == 0
        )

        return {
            "rows": int(len(data)),
            "columns": list(data.columns),
            "missing_columns": missing_columns,
            "missing_ratio": missing_ratio,
            "high_missing_columns": high_missing_columns,
            "duplicate_ratio": duplicate_ratio,
            "type_profile": type_profile,
            "invalid_time_rows": invalid_time_rows,
            "invalid_target_count": invalid_target_count,
            "target_distribution": target_distribution,
            "is_acceptable": is_acceptable,
        }

    def clean(
        self,
        data: pd.DataFrame,
        categorical_fill_value: str = "unknown",
        drop_columns_missing_ratio: float | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        cleaned = data.drop_duplicates().copy()
        dropped_columns: list[str] = []
        if drop_columns_missing_ratio is not None and not cleaned.empty:
            dropped_columns = [
                column
                for column, ratio in cleaned.isna().mean().items()
                if ratio > drop_columns_missing_ratio and column not in self.required_columns
            ]
            if dropped_columns:
                cleaned = cleaned.drop(columns=dropped_columns)

        filled_numeric_columns: list[str] = []
        filled_categorical_columns: list[str] = []
        for column in cleaned.columns:
            if pd.api.types.is_object_dtype(cleaned[column]) or pd.api.types.is_categorical_dtype(cleaned[column]):
                cleaned[column] = cleaned[column].fillna(categorical_fill_value)
                filled_categorical_columns.append(column)
            elif pd.api.types.is_datetime64_any_dtype(cleaned[column]):
                cleaned[column] = pd.to_datetime(cleaned[column], errors="coerce")
            else:
                median_value = cleaned[column].median()
                if pd.isna(median_value):
                    median_value = 0
                cleaned[column] = cleaned[column].fillna(median_value)
                filled_numeric_columns.append(column)

        cleaning_report = {
            "rows_before": int(len(data)),
            "rows_after": int(len(cleaned)),
            "duplicates_removed": int(len(data) - len(data.drop_duplicates())),
            "dropped_columns": dropped_columns,
            "filled_numeric_columns": filled_numeric_columns,
            "filled_categorical_columns": filled_categorical_columns,
            "remaining_missing_total": int(cleaned.isna().sum().sum()) if not cleaned.empty else 0,
        }
        return cleaned, cleaning_report
