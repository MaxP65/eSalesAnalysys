from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


@dataclass
class ModelServing:
    model_path: Path
    preprocessor_path: Path

    def load_model(self) -> Any:
        return joblib.load(self.model_path)

    def load_preprocessor(self) -> Any:
        return joblib.load(self.preprocessor_path)

    def predict(self, model: Any, transformed_data):
        return model.predict(transformed_data)
