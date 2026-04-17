from __future__ import annotations

from dataclasses import dataclass

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier


@dataclass
class ModelTrainer:
    random_state: int = 42

    def build_model(self, model_type: str, **params):
        if model_type == "decision_tree":
            return DecisionTreeClassifier(random_state=self.random_state, **params)
        if model_type == "logistic_regression":
            return LogisticRegression(random_state=self.random_state, **params)
        raise ValueError(f"Unsupported model type: {model_type}")

    def train(self, features, target, model_type: str, **params):
        model = self.build_model(model_type=model_type, **params)
        model.fit(features, target)
        return model

    @staticmethod
    def calculate_training_metrics(model, features, target) -> dict[str, float]:
        predictions = model.predict(features)
        return {
            "accuracy": float(accuracy_score(target, predictions)),
            "precision": float(precision_score(target, predictions, zero_division=0)),
            "recall": float(recall_score(target, predictions, zero_division=0)),
            "f1": float(f1_score(target, predictions, zero_division=0)),
        }

    @staticmethod
    def save_model(model, output_path: str) -> None:
        joblib.dump(model, output_path)
