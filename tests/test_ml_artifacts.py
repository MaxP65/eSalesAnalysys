from __future__ import annotations

import json
import unittest
from pathlib import Path

import joblib
import pandas as pd

from src.data_preparation.preprocessor import DataPreprocessor
from src.utils.config import load_config


class MLArtifactSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config("configs/config.yaml")
        self.models_dir = Path(self.config["paths"]["models_dir"])

    def test_dataset_config_is_usable(self) -> None:
        dataset_path = Path(self.config["data"]["prepared_dataset_path"])

        self.assertTrue(dataset_path.exists())
        self.assertIn("time_column", self.config["data"])
        self.assertIn("target_column", self.config["data"])
        self.assertGreater(self.config["data"]["batch_size"], 0)

    def test_model_registry_has_best_model(self) -> None:
        registry_path = self.models_dir / "model_registry.json"

        self.assertTrue(registry_path.exists())
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        self.assertTrue(registry.get("best_model"))
        self.assertGreater(len(registry.get("models", [])), 0)

    def test_latest_model_and_preprocessor_can_predict(self) -> None:
        model_path = self.models_dir / "model_latest.joblib"
        preprocessor_path = self.models_dir / "preprocessor_latest.joblib"
        metadata_path = self.models_dir / "training_metadata.json"
        dataset_path = Path(self.config["data"]["prepared_dataset_path"])

        self.assertTrue(model_path.exists())
        self.assertTrue(preprocessor_path.exists())
        self.assertTrue(metadata_path.exists())

        model = joblib.load(model_path)
        transformer = joblib.load(preprocessor_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        sample = pd.read_csv(dataset_path).head(10)
        preprocessor = DataPreprocessor(
            target_column=self.config["data"]["target_column"],
            drop_columns=metadata["dropped_feature_columns"],
        )
        features, _ = preprocessor.transform(sample, transformer)
        predictions = model.predict(features)

        self.assertEqual(len(predictions), len(sample))


if __name__ == "__main__":
    unittest.main()
