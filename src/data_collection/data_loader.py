from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.io import ensure_directory, write_json


@dataclass
class DataStreamEmulator:
    dataset_path: Path
    time_column: str
    batch_size: int
    current_position: int = 0

    def load(self) -> pd.DataFrame:
        data = pd.read_csv(self.dataset_path, parse_dates=[self.time_column])
        return data.sort_values(self.time_column).reset_index(drop=True)

    def get_next_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        start = self.current_position
        end = start + self.batch_size
        batch = data.iloc[start:end].copy()
        self.current_position = min(end, len(data))
        return batch

    def has_more_data(self, data: pd.DataFrame) -> bool:
        return self.current_position < len(data)


def save_raw_batch(batch_df: pd.DataFrame, output_dir: str | Path, batch_id: str) -> Path:
    output_directory = ensure_directory(output_dir)
    output_path = output_directory / f"{batch_id}.csv"
    batch_df.to_csv(output_path, index=False)
    return output_path


def load_stream_state(state_path: str | Path) -> dict[str, Any]:
    path = Path(state_path)
    if not path.exists():
        return {"current_position": 0, "batch_number": 0}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_stream_state(state: dict[str, Any], state_path: str | Path) -> Path:
    path = Path(state_path)
    ensure_directory(path.parent)
    write_json(state, path)
    return path


def build_batch_metadata(batch_df: pd.DataFrame, time_column: str, batch_id: str) -> dict[str, Any]:
    numeric_summary = batch_df.describe(include=["number"]).to_dict() if not batch_df.empty else {}
    categorical_columns = batch_df.select_dtypes(include=["object", "category"]).columns.tolist()
    categorical_cardinality = {
        column: int(batch_df[column].nunique(dropna=True)) for column in categorical_columns
    }

    time_min = None
    time_max = None
    if time_column in batch_df.columns and not batch_df.empty:
        time_min = str(batch_df[time_column].min())
        time_max = str(batch_df[time_column].max())

    return {
        "batch_id": batch_id,
        "rows": int(len(batch_df)),
        "columns": list(batch_df.columns),
        "time_column": time_column,
        "time_min": time_min,
        "time_max": time_max,
        "missing_ratio": batch_df.isna().mean().to_dict() if not batch_df.empty else {},
        "duplicate_ratio": float(batch_df.duplicated().mean()) if not batch_df.empty else 0.0,
        "numeric_summary": numeric_summary,
        "categorical_cardinality": categorical_cardinality,
    }
