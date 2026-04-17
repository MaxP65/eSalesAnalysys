from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.data_analysis.quality_checker import DataQualityChecker
from src.data_collection.data_loader import (
    DataStreamEmulator,
    build_batch_metadata,
    load_stream_state,
    save_raw_batch,
    save_stream_state,
)
from src.data_preparation.preprocessor import DataPreprocessor
from src.model_serving.serving import ModelServing
from src.model_training.trainer import ModelTrainer
from src.model_validation.validator import ModelValidator
from src.reporting.summary import build_summary_report
from src.utils.config import load_config
from src.utils.io import ensure_directory
from src.utils.io import append_jsonl, write_json
from src.utils.logging_utils import setup_logging


LOGGER = logging.getLogger(__name__)


def record_performance(history_dir: Path, stage: str, started_at: float, extra: dict | None = None) -> None:
    duration_seconds = time.perf_counter() - started_at
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "stage": stage,
        "duration_seconds": duration_seconds,
    }
    if extra:
        record.update(extra)
    append_jsonl(record, history_dir / "performance_history.jsonl")


def update_accumulated_dataset(
    cleaned_batch_df,
    accumulated_dataset_path: Path,
    time_column: str,
) -> Path:
    if accumulated_dataset_path.exists():
        accumulated_df = pd.read_csv(accumulated_dataset_path, parse_dates=[time_column])
        combined_df = pd.concat([accumulated_df, cleaned_batch_df], ignore_index=True)
    else:
        combined_df = cleaned_batch_df.copy()

    if "order_id" in combined_df.columns:
        combined_df = combined_df.drop_duplicates(subset=["order_id"], keep="last")
    combined_df = combined_df.sort_values(time_column).reset_index(drop=True)
    ensure_directory(accumulated_dataset_path.parent)
    combined_df.to_csv(accumulated_dataset_path, index=False)
    return accumulated_dataset_path


def run_validation_training_and_registry(config: dict, batch_id: str, history_dir: Path) -> dict:
    accumulated_dataset_path = Path(config["data"]["accumulated_dataset_path"])
    time_column = config["data"]["time_column"]
    target_column = config["data"]["target_column"]
    models_dir = ensure_directory(config["paths"]["models_dir"])
    primary_metric = config["model"]["primary_metric"]
    validation_start = time.perf_counter()

    accumulated_df = pd.read_csv(accumulated_dataset_path, parse_dates=[time_column])
    preprocessor = DataPreprocessor(
        target_column=target_column,
        drop_columns=["order_id", "customer_id", "order_status", time_column, "delivery_time_days"],
    )
    trainer = ModelTrainer(random_state=config["project"]["random_state"])
    validator = ModelValidator(n_splits=config["validation"]["n_splits"])
    validation_result = validator.evaluate_param_candidates(
        data=accumulated_df,
        preprocessor=preprocessor,
        trainer=trainer,
        model_candidates=config["model"]["candidates"],
        time_column=time_column,
    )
    append_jsonl(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "batch_id": batch_id,
            "best_model_type": validation_result["model_type"],
            "best_params": validation_result["params"],
            "avg_metrics": validation_result["avg_metrics"],
            "fold_metrics": validation_result["fold_metrics"],
        },
        history_dir / "validation_history.jsonl",
    )
    record_performance(
        history_dir,
        stage="validation",
        started_at=validation_start,
        extra={"batch_id": batch_id, "best_f1": validation_result["avg_metrics"]["f1"], "best_model_type": validation_result["model_type"]},
    )

    training_start = time.perf_counter()
    training_features, training_target, transformer, feature_names = preprocessor.fit_transform(accumulated_df)
    model_params = validation_result["params"]
    model_type = validation_result["model_type"]
    model = trainer.train(training_features, training_target, model_type=model_type, **model_params)
    training_metrics = trainer.calculate_training_metrics(model, training_features, training_target)

    registry_path = Path(models_dir) / "model_registry.json"
    registry = {"models": []}
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    next_version = len(registry.get("models", [])) + 1

    model_path = Path(models_dir) / f"model_v{next_version:03d}.joblib"
    preprocessor_path = Path(models_dir) / f"preprocessor_v{next_version:03d}.joblib"
    training_metadata_path = Path(models_dir) / "training_metadata.json"

    trainer.save_model(model, str(model_path))
    DataPreprocessor.save_transformer(transformer, str(preprocessor_path))
    trainer.save_model(model, str(Path(models_dir) / "model_latest.joblib"))
    DataPreprocessor.save_transformer(transformer, str(Path(models_dir) / "preprocessor_latest.joblib"))

    class_distribution = {
        str(key): float(value)
        for key, value in accumulated_df[target_column].value_counts(normalize=True).sort_index().items()
    }

    feature_importance_path = Path(models_dir) / f"feature_importance_{next_version:03d}.csv"
    if hasattr(model, "feature_importances_"):
        feature_importance_df = pd.DataFrame(
            {
                "feature_name": feature_names,
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)
        feature_importance_df.to_csv(feature_importance_path, index=False)
        top_features = feature_importance_df.head(10).to_dict(orient="records")
    elif hasattr(model, "coef_"):
        feature_importance_df = pd.DataFrame(
            {
                "feature_name": feature_names,
                "importance": model.coef_[0],
                "abs_importance": abs(model.coef_[0]),
            }
        ).sort_values("abs_importance", ascending=False)
        feature_importance_df.to_csv(feature_importance_path, index=False)
        top_features = feature_importance_df.head(10).to_dict(orient="records")
    else:
        top_features = []

    training_metadata = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "batch_id": batch_id,
        "target_column": target_column,
        "rows": int(len(accumulated_df)),
        "feature_count_after_encoding": int(len(feature_names)),
        "model_type": model_type,
        "model_params": model_params,
        "validation_metrics": validation_result["avg_metrics"],
        "training_metrics": training_metrics,
        "class_distribution": class_distribution,
        "model_path": str(model_path),
        "preprocessor_path": str(preprocessor_path),
        "version": f"v{next_version:03d}",
        "dropped_feature_columns": ["order_id", "customer_id", "order_status", time_column, "delivery_time_days"],
        "top_feature_importance": top_features,
    }
    write_json(training_metadata, training_metadata_path)
    append_jsonl(training_metadata, history_dir / "training_history.jsonl")

    previous_best_name = registry.get("best_model")
    previous_best_score = None
    if previous_best_name:
        for item in registry.get("models", []):
            if item["version"] == previous_best_name:
                previous_best_score = item["validation_metrics"][primary_metric]
                break

    registry_entry = {
        "version": training_metadata["version"],
        "batch_id": batch_id,
        "model_type": model_type,
        "model_path": str(model_path),
        "preprocessor_path": str(preprocessor_path),
        "feature_importance_path": str(feature_importance_path),
        "model_params": model_params,
        "validation_metrics": validation_result["avg_metrics"],
        "training_metrics": training_metrics,
        "created_at": training_metadata["timestamp"],
    }
    registry.setdefault("models", []).append(registry_entry)
    if previous_best_score is None or validation_result["avg_metrics"][primary_metric] >= previous_best_score:
        registry["best_model"] = training_metadata["version"]
        registry["best_metric"] = primary_metric
        registry["best_metric_value"] = validation_result["avg_metrics"][primary_metric]
    write_json(registry, registry_path)

    model_drift_record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "batch_id": batch_id,
        "version": training_metadata["version"],
        "validation_metric_name": primary_metric,
        "validation_metric_value": validation_result["avg_metrics"][primary_metric],
        "reference_best_metric_value": previous_best_score if previous_best_score is not None else validation_result["avg_metrics"][primary_metric],
        "metric_drop": 0.0 if previous_best_score is None else float(previous_best_score - validation_result["avg_metrics"][primary_metric]),
        "is_model_drift": False if previous_best_score is None else bool(validation_result["avg_metrics"][primary_metric] < previous_best_score),
    }
    append_jsonl(model_drift_record, history_dir / "model_drift_history.jsonl")

    record_performance(
        history_dir,
        stage="training",
        started_at=training_start,
        extra={"batch_id": batch_id, "rows": len(accumulated_df), "feature_count": len(feature_names)},
    )
    return training_metadata


def calculate_simple_drift(batch_df: pd.DataFrame, accumulated_dataset_path: Path, target_column: str, batch_id: str) -> dict:
    current_ratio = float(batch_df[target_column].mean())
    numeric_columns_for_drift = [
        column
        for column in ["price_sum", "freight_value_sum", "items_count", "payment_value_sum"]
        if column in batch_df.columns
    ]
    drift_report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "batch_id": batch_id,
        "current_target_ratio": current_ratio,
        "reference_target_ratio": current_ratio,
        "target_ratio_change": 0.0,
        "feature_mean_shift": {},
    }
    if accumulated_dataset_path.exists():
        accumulated_df = pd.read_csv(accumulated_dataset_path)
        if not accumulated_df.empty:
            reference_ratio = float(accumulated_df[target_column].mean())
            drift_report["reference_target_ratio"] = reference_ratio
            drift_report["target_ratio_change"] = abs(current_ratio - reference_ratio)
            for column in numeric_columns_for_drift:
                if column not in accumulated_df.columns:
                    continue
                current_mean = float(batch_df[column].mean())
                reference_mean = float(accumulated_df[column].mean())
                reference_std = float(accumulated_df[column].std()) if pd.notna(accumulated_df[column].std()) else 0.0
                standardized_shift = 0.0 if reference_std == 0 else abs(current_mean - reference_mean) / reference_std
                drift_report["feature_mean_shift"][column] = {
                    "current_mean": current_mean,
                    "reference_mean": reference_mean,
                    "standardized_shift": standardized_shift,
                }
    return drift_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLOps MVP for long delivery prediction.")
    parser.add_argument("-mode", choices=["update", "inference", "summary"], required=True)
    parser.add_argument("-file", default=None, help="Path to inference CSV file.")
    parser.add_argument("-config", default="configs/config.yaml", help="Path to config file.")
    return parser.parse_args()


def run_update(config: dict) -> bool:
    update_start = time.perf_counter()
    LOGGER.info("Update mode started.")
    prepared_dataset = Path(config["data"]["prepared_dataset_path"])
    if not prepared_dataset.exists():
        LOGGER.warning("Prepared dataset is missing: %s", prepared_dataset)
        return False

    time_column = config["data"]["time_column"]
    batch_size = config["data"]["batch_size"]
    history_dir = Path(config["paths"]["history_dir"])
    state_path = history_dir / "stream_state.json"

    state = load_stream_state(state_path)
    emulator = DataStreamEmulator(
        dataset_path=prepared_dataset,
        time_column=time_column,
        batch_size=batch_size,
        current_position=int(state.get("current_position", 0)),
    )
    data = emulator.load()
    if not emulator.has_more_data(data):
        LOGGER.info("No more data available for update.")
        return True

    next_batch_number = int(state.get("batch_number", 0)) + 1
    batch_id = f"batch_{next_batch_number:04d}"
    batch_df = emulator.get_next_batch(data)

    raw_batch_path = save_raw_batch(batch_df, config["paths"]["raw_storage_dir"], batch_id)
    metadata = build_batch_metadata(batch_df, time_column=time_column, batch_id=batch_id)
    metadata_path = Path(config["paths"]["batch_metadata_dir"]) / f"{batch_id}_meta.json"
    write_json(metadata, metadata_path)

    checker = DataQualityChecker(
        required_columns=config["data"]["required_columns"],
        max_missing_ratio=config["quality"]["max_missing_ratio"],
        max_duplicate_ratio=config["quality"]["max_duplicate_ratio"],
        time_column=config["data"]["time_column"],
        target_column=config["data"]["target_column"],
        allowed_target_values=config["quality"]["allowed_target_values"],
    )
    quality_report = checker.evaluate(batch_df)
    quality_path = Path(config["paths"]["batch_metadata_dir"]) / f"{batch_id}_quality.json"
    write_json(quality_report, quality_path)

    cleaned_batch_df, cleaning_report = checker.clean(
        batch_df,
        categorical_fill_value=config["quality"]["categorical_fill_value"],
        drop_columns_missing_ratio=config["quality"]["drop_columns_missing_ratio"],
    )
    cleaned_storage_dir = ensure_directory(config["paths"]["cleaned_storage_dir"])
    cleaned_batch_path = cleaned_storage_dir / f"{batch_id}_cleaned.csv"
    cleaned_batch_df.to_csv(cleaned_batch_path, index=False)
    cleaning_path = Path(config["paths"]["batch_metadata_dir"]) / f"{batch_id}_cleaning.json"
    write_json(cleaning_report, cleaning_path)

    post_clean_quality_report = checker.evaluate(cleaned_batch_df)
    post_clean_quality_path = Path(config["paths"]["batch_metadata_dir"]) / f"{batch_id}_post_clean_quality.json"
    write_json(post_clean_quality_report, post_clean_quality_path)

    append_jsonl(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "batch_id": batch_id,
            "raw_batch_path": str(raw_batch_path),
            "cleaned_batch_path": str(cleaned_batch_path),
            "metadata_path": str(metadata_path),
            "quality_path": str(quality_path),
            "cleaning_path": str(cleaning_path),
            "post_clean_quality_path": str(post_clean_quality_path),
            "rows": int(len(batch_df)),
            "rows_after_cleaning": int(len(cleaned_batch_df)),
            "is_acceptable": quality_report["is_acceptable"],
            "is_acceptable_after_cleaning": post_clean_quality_report["is_acceptable"],
        },
        history_dir / "stream_history.jsonl",
    )

    append_jsonl(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "batch_id": batch_id,
            "quality_report": quality_report,
            "cleaning_report": cleaning_report,
            "post_clean_quality_report": post_clean_quality_report,
        },
        history_dir / "data_quality_history.jsonl",
    )

    drift_report = calculate_simple_drift(
        batch_df=cleaned_batch_df,
        accumulated_dataset_path=Path(config["data"]["accumulated_dataset_path"]),
        target_column=config["data"]["target_column"],
        batch_id=batch_id,
    )
    append_jsonl(drift_report, history_dir / "drift_history.jsonl")

    accumulated_dataset_path = update_accumulated_dataset(
        cleaned_batch_df=cleaned_batch_df,
        accumulated_dataset_path=Path(config["data"]["accumulated_dataset_path"]),
        time_column=time_column,
    )
    training_metadata = run_validation_training_and_registry(
        config=config,
        batch_id=batch_id,
        history_dir=history_dir,
    )

    save_stream_state(
        {
            "current_position": emulator.current_position,
            "batch_number": next_batch_number,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        },
        state_path,
    )

    LOGGER.info(
        "Processed %s with %s rows. Quality acceptable before cleaning: %s, after cleaning: %s. Accumulated rows: %s. Validation F1: %.4f. Training F1: %.4f",
        batch_id,
        len(batch_df),
        quality_report["is_acceptable"],
        post_clean_quality_report["is_acceptable"],
        training_metadata["rows"],
        training_metadata["validation_metrics"]["f1"],
        training_metadata["training_metrics"]["f1"],
    )
    record_performance(
        history_dir,
        stage="update",
        started_at=update_start,
        extra={"batch_id": batch_id, "rows_after_cleaning": len(cleaned_batch_df)},
    )
    return True


def run_inference(config: dict, file_path: str | None) -> Path:
    inference_start = time.perf_counter()
    if not file_path:
        raise ValueError("Inference mode requires the -file argument.")

    inference_path = Path(file_path)
    if not inference_path.exists():
        raise FileNotFoundError(f"Inference file not found: {inference_path}")

    models_dir = Path(config["paths"]["models_dir"])
    registry_path = models_dir / "model_registry.json"
    if not registry_path.exists():
        raise FileNotFoundError(f"Model registry not found: {registry_path}")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if not registry.get("best_model"):
        raise ValueError("Best model is not defined in model registry.")

    best_entry = None
    for item in registry.get("models", []):
        if item["version"] == registry["best_model"]:
            best_entry = item
            break
    if best_entry is None:
        raise ValueError("Best model entry is missing in registry.")

    serving = ModelServing(
        model_path=Path(best_entry["model_path"]),
        preprocessor_path=Path(best_entry["preprocessor_path"]),
    )
    model = serving.load_model()
    transformer = serving.load_preprocessor()

    header = pd.read_csv(inference_path, nrows=0)
    parse_dates = [config["data"]["time_column"]] if config["data"]["time_column"] in header.columns else None
    data = pd.read_csv(inference_path, parse_dates=parse_dates)
    preprocessor = DataPreprocessor(
        target_column=config["data"]["target_column"],
        drop_columns=json.loads((models_dir / "training_metadata.json").read_text(encoding="utf-8"))["dropped_feature_columns"],
    )
    features, _ = preprocessor.transform(data, transformer)
    predictions = serving.predict(model, features)

    output_dir = Path(config["paths"]["reports_dir"]) / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{inference_path.stem}_with_predictions.csv"
    result = data.copy()
    result["predict"] = predictions
    result.to_csv(output_path, index=False)

    record_performance(
        Path(config["paths"]["history_dir"]),
        stage="inference",
        started_at=inference_start,
        extra={"rows": len(result), "output_path": str(output_path)},
    )
    LOGGER.info("Inference completed, output path: %s", output_path)
    return output_path


def run_summary(config: dict) -> Path:
    LOGGER.info("Summary mode started.")
    return build_summary_report(
        Path(config["paths"]["reports_dir"]) / "summary",
        history_dir=Path(config["paths"]["history_dir"]),
        models_dir=Path(config["paths"]["models_dir"]),
    )


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    setup_logging(
        logs_dir=config["paths"]["logs_dir"],
        file_name=config["logging"]["file_name"],
        level=config["logging"]["level"],
    )

    if args.mode == "update":
        return 0 if run_update(config) else 1
    if args.mode == "inference":
        output_path = run_inference(config, args.file)
        print(output_path)
        return 0
    if args.mode == "summary":
        output_path = run_summary(config)
        print(output_path)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
