from __future__ import annotations

from dataclasses import dataclass

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass
class DataPreprocessor:
    target_column: str
    drop_columns: list[str] | None = None

    def _removable_columns(self) -> list[str]:
        removable_columns = [self.target_column]
        if self.drop_columns:
            removable_columns.extend(self.drop_columns)
        return removable_columns

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        existing_columns = [column for column in self._removable_columns() if column in data.columns]
        return data.drop(columns=existing_columns)

    def split_features_target(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        features = self.prepare_features(data)
        if self.target_column not in data.columns:
            raise KeyError(f"Target column is missing: {self.target_column}")
        target = data[self.target_column]
        return features, target

    def split_features_target_optional(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
        features = self.prepare_features(data)
        target = data[self.target_column] if self.target_column in data.columns else None
        return features, target

    def build_transformer(self, features: pd.DataFrame) -> ColumnTransformer:
        categorical_columns = features.select_dtypes(include=["object", "category"]).columns.tolist()
        numeric_columns = [column for column in features.columns if column not in categorical_columns]

        numeric_pipeline = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median"))]
        )
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_columns),
                ("cat", categorical_pipeline, categorical_columns),
            ]
        )

    def fit_transform(self, data: pd.DataFrame):
        features, target = self.split_features_target(data)
        transformer = self.build_transformer(features)
        transformed_features = transformer.fit_transform(features)
        feature_names = transformer.get_feature_names_out().tolist()
        return transformed_features, target, transformer, feature_names

    def transform(self, data: pd.DataFrame, transformer: ColumnTransformer):
        features, target = self.split_features_target_optional(data)
        transformed_features = transformer.transform(features)
        return transformed_features, target

    @staticmethod
    def save_transformer(transformer: ColumnTransformer, output_path: str) -> None:
        joblib.dump(transformer, output_path)
