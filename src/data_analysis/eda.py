from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils.io import ensure_directory


def _save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_eda_report(
    data: pd.DataFrame,
    output_dir: str | Path,
    time_column: str,
    target_column: str,
    top_categories_limit: int = 15,
    report_name: str = "initial_eda",
) -> Path:
    report_dir = ensure_directory(output_dir)
    figures_dir = ensure_directory(report_dir / "figures")
    report_path = report_dir / f"{report_name}.md"

    dataset = data.copy()
    if time_column in dataset.columns:
        dataset[time_column] = pd.to_datetime(dataset[time_column], errors="coerce")

    sns.set_theme(style="whitegrid")
    numeric_columns = dataset.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = dataset.select_dtypes(include=["object", "category"]).columns.tolist()

    figure_paths: list[Path] = []

    if time_column in dataset.columns:
        monthly_counts = (
            dataset.assign(order_month=dataset[time_column].dt.to_period("M").astype(str))
            .groupby("order_month")
            .size()
            .reset_index(name="orders")
        )
        if not monthly_counts.empty:
            plt.figure(figsize=(10, 4))
            sns.lineplot(data=monthly_counts, x="order_month", y="orders", marker="o")
            plt.xticks(rotation=45, ha="right")
            plt.title("Orders Over Time")
            figure_path = figures_dir / f"{report_name}_orders_over_time.png"
            _save_figure(figure_path)
            figure_paths.append(figure_path)

    if target_column in dataset.columns:
        plt.figure(figsize=(6, 4))
        target_plot = dataset[target_column].astype("string")
        sns.countplot(x=target_plot)
        plt.title("Target Distribution")
        plt.xlabel(target_column)
        figure_path = figures_dir / f"{report_name}_target_distribution.png"
        _save_figure(figure_path)
        figure_paths.append(figure_path)

    if "purchase_month" in dataset.columns and target_column in dataset.columns:
        monthly_long_delivery = (
            dataset.groupby("purchase_month")[target_column]
            .mean()
            .reset_index(name="long_delivery_ratio")
            .sort_values("purchase_month")
        )
        plt.figure(figsize=(8, 4))
        sns.barplot(data=monthly_long_delivery, x="purchase_month", y="long_delivery_ratio")
        plt.title("Long Delivery Ratio by Purchase Month")
        figure_path = figures_dir / f"{report_name}_long_delivery_by_month.png"
        _save_figure(figure_path)
        figure_paths.append(figure_path)

    for column in [value for value in ["customer_state", "seller_state", "product_category_name_english"] if value in dataset.columns]:
        if target_column not in dataset.columns:
            break
        grouped = (
            dataset.groupby(column)
            .agg(long_delivery_ratio=(target_column, "mean"), sample_size=(target_column, "size"))
            .reset_index()
            .query("sample_size >= 100")
            .sort_values("long_delivery_ratio", ascending=False)
            .head(top_categories_limit)
        )
        if grouped.empty:
            continue
        plt.figure(figsize=(10, 5))
        sns.barplot(data=grouped, x="long_delivery_ratio", y=column)
        plt.title(f"Long Delivery Ratio by {column}")
        figure_path = figures_dir / f"{report_name}_{column}_long_delivery_ratio.png"
        _save_figure(figure_path)
        figure_paths.append(figure_path)

    if "purchase_dayofweek" in dataset.columns and target_column in dataset.columns:
        weekday_ratio = (
            dataset.groupby("purchase_dayofweek")[target_column]
            .mean()
            .reset_index(name="long_delivery_ratio")
            .sort_values("purchase_dayofweek")
        )
        plt.figure(figsize=(8, 4))
        sns.barplot(data=weekday_ratio, x="purchase_dayofweek", y="long_delivery_ratio")
        plt.title("Long Delivery Ratio by Purchase Day of Week")
        figure_path = figures_dir / f"{report_name}_long_delivery_by_weekday.png"
        _save_figure(figure_path)
        figure_paths.append(figure_path)

    if "same_state_flag" in dataset.columns and target_column in dataset.columns:
        same_state_ratio = (
            dataset.groupby("same_state_flag")[target_column]
            .mean()
            .reset_index(name="long_delivery_ratio")
            .sort_values("same_state_flag")
        )
        plt.figure(figsize=(6, 4))
        sns.barplot(data=same_state_ratio, x="same_state_flag", y="long_delivery_ratio")
        plt.title("Long Delivery Ratio by Same State Flag")
        figure_path = figures_dir / f"{report_name}_long_delivery_by_same_state.png"
        _save_figure(figure_path)
        figure_paths.append(figure_path)

    for column in [value for value in ["price_sum", "freight_value_sum", "items_count", "delivery_time_days"] if value in numeric_columns]:
        plt.figure(figsize=(8, 4))
        sns.histplot(dataset[column].dropna(), bins=30, kde=True)
        plt.title(f"Distribution of {column}")
        figure_path = figures_dir / f"{report_name}_{column}_distribution.png"
        _save_figure(figure_path)
        figure_paths.append(figure_path)

    for column in [value for value in ["price_sum", "freight_value_sum"] if value in dataset.columns and target_column in dataset.columns]:
        sample = dataset[[column, target_column]].dropna()
        if sample.empty:
            continue
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=sample, x=target_column, y=column)
        plt.title(f"{column} by {target_column}")
        figure_path = figures_dir / f"{report_name}_{column}_by_target.png"
        _save_figure(figure_path)
        figure_paths.append(figure_path)

    if numeric_columns:
        corr = dataset[numeric_columns].corr(numeric_only=True)
        if not corr.empty:
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, cmap="Blues", center=0, square=False)
            plt.title("Numeric Feature Correlation")
            figure_path = figures_dir / f"{report_name}_correlation_heatmap.png"
            _save_figure(figure_path)
            figure_paths.append(figure_path)

    missing_ratio = dataset.isna().mean().sort_values(ascending=False)
    top_missing = missing_ratio[missing_ratio > 0].head(20)

    lines = [
        f"# EDA Report: {report_name}",
        "",
        f"Rows: {len(dataset)}",
        f"Columns: {len(dataset.columns)}",
        "",
        "## Dataset Overview",
        "",
        f"- Time column: `{time_column}`",
        f"- Target column: `{target_column}`",
        f"- Numeric columns: {len(numeric_columns)}",
        f"- Categorical columns: {len(categorical_columns)}",
        f"- Duplicate ratio: {float(dataset.duplicated().mean()) if not dataset.empty else 0.0:.4f}",
        "",
        "## Missing Values",
        "",
    ]

    if top_missing.empty:
        lines.append("- No missing values detected.")
    else:
        for column, ratio in top_missing.items():
            lines.append(f"- `{column}`: {ratio:.4f}")

    if target_column in dataset.columns:
        target_distribution = dataset[target_column].value_counts(normalize=True, dropna=False).sort_index()
        lines.extend(["", "## Target Distribution", ""])
        for value, ratio in target_distribution.items():
            lines.append(f"- `{value}`: {ratio:.4f}")

    lines.extend(["", "## Figures", ""])
    for figure_path in figure_paths:
        relative_path = figure_path.relative_to(report_dir)
        lines.append(f"- ![]({relative_path.as_posix()})")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
