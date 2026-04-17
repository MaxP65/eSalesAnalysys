from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.utils.io import ensure_directory


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_summary_report(output_dir: str | Path, history_dir: str | Path, models_dir: str | Path) -> Path:
    report_dir = ensure_directory(output_dir)
    report_path = report_dir / "latest_summary.md"
    timestamp = datetime.now().isoformat(timespec="seconds")
    history_path = Path(history_dir)
    models_path = Path(models_dir)

    quality_history = _read_jsonl(history_path / "data_quality_history.jsonl")
    training_history = _read_jsonl(history_path / "training_history.jsonl")
    validation_history = _read_jsonl(history_path / "validation_history.jsonl")
    performance_history = _read_jsonl(history_path / "performance_history.jsonl")
    drift_history = _read_jsonl(history_path / "drift_history.jsonl")
    model_drift_history = _read_jsonl(history_path / "model_drift_history.jsonl")
    registry_path = models_path / "model_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {"models": []}

    lines = [
        "# Monitoring Summary",
        "",
        f"Generated at: {timestamp}",
        "",
        "## Data Quality",
        "",
        f"- Processed batches: {len(quality_history)}",
    ]

    if quality_history:
        latest_quality = quality_history[-1]
        lines.extend([
            f"- Latest batch: `{latest_quality['batch_id']}`",
            f"- Latest quality acceptable: `{latest_quality['post_clean_quality_report']['is_acceptable']}`",
        ])

    lines.extend(["", "## Model Validation", "", f"- Validation runs: {len(validation_history)}"])
    if validation_history:
        latest_validation = validation_history[-1]
        lines.extend([
            f"- Latest batch: `{latest_validation['batch_id']}`",
            f"- Best params: `{latest_validation['best_params']}`",
            f"- Best F1: `{latest_validation['avg_metrics']['f1']:.4f}`",
        ])

    lines.extend(["", "## Training", "", f"- Training runs: {len(training_history)}"])
    if training_history:
        latest_training = training_history[-1]
        lines.extend([
            f"- Latest accumulated rows: `{latest_training['rows']}`",
            f"- Encoded feature count: `{latest_training['feature_count_after_encoding']}`",
            f"- Training F1: `{latest_training['training_metrics']['f1']:.4f}`",
        ])

    lines.extend(["", "## Model Registry", "", f"- Stored model versions: {len(registry.get('models', []))}"])
    if registry.get("best_model"):
        lines.append(f"- Current best model: `{registry['best_model']}`")
        best_entry = None
        for item in registry.get("models", []):
            if item["version"] == registry["best_model"]:
                best_entry = item
                break
        if best_entry:
            lines.extend([
                f"- Best model type: `{best_entry.get('model_type', 'unknown')}`",
                f"- Best params: `{best_entry.get('model_params')}`",
                f"- Best validation F1: `{best_entry['validation_metrics']['f1']:.4f}`",
            ])

    lines.extend(["", "## Performance", "", f"- Performance records: {len(performance_history)}"])
    if performance_history:
        latest_perf = performance_history[-1]
        lines.extend([
            f"- Latest stage: `{latest_perf['stage']}`",
            f"- Duration seconds: `{latest_perf['duration_seconds']:.4f}`",
        ])

    lines.extend(["", "## Drift", "", f"- Drift records: {len(drift_history)}"])
    if drift_history:
        latest_drift = drift_history[-1]
        lines.extend([
            f"- Latest batch: `{latest_drift['batch_id']}`",
            f"- Target drift: `{latest_drift['target_ratio_change']:.4f}`",
        ])
        feature_shift = latest_drift.get("feature_mean_shift", {})
        if feature_shift:
            lines.append("- Numeric feature drift:")
            for feature_name, values in feature_shift.items():
                lines.append(
                    f"  - `{feature_name}` standardized mean shift: `{values['standardized_shift']:.4f}`"
                )

    lines.extend(["", "## Model Drift", "", f"- Model drift records: {len(model_drift_history)}"])
    if model_drift_history:
        latest_model_drift = model_drift_history[-1]
        lines.extend([
            f"- Latest version: `{latest_model_drift['version']}`",
            f"- Metric drop: `{latest_model_drift['metric_drop']:.4f}`",
            f"- Drift flag: `{latest_model_drift['is_model_drift']}`",
        ])

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
