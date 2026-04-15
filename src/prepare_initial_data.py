from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.io import ensure_directory, write_json
from src.utils.logging_utils import setup_logging


LOGGER = logging.getLogger(__name__)


EXPECTED_FILES = {
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "products": "olist_products_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Olist dataset for the MLOps MVP.")
    parser.add_argument("-config", default="configs/config.yaml", help="Path to config file.")
    parser.add_argument(
        "--missing-rate",
        type=float,
        default=0.03,
        help="Share of values to blank out in selected columns for quality-check demonstrations.",
    )
    return parser.parse_args()


def load_csv(raw_dir: Path, filename: str, parse_dates: list[str] | None = None) -> pd.DataFrame:
    path = raw_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Required source file is missing: {path}")
    return pd.read_csv(path, parse_dates=parse_dates)


def build_order_level_dataset(raw_dir: Path) -> pd.DataFrame:
    orders = load_csv(
        raw_dir,
        EXPECTED_FILES["orders"],
        parse_dates=[
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    )
    order_items = load_csv(raw_dir, EXPECTED_FILES["order_items"])
    products = load_csv(raw_dir, EXPECTED_FILES["products"])
    customers = load_csv(raw_dir, EXPECTED_FILES["customers"])
    sellers = load_csv(raw_dir, EXPECTED_FILES["sellers"])
    payments = load_csv(raw_dir, EXPECTED_FILES["payments"])
    category_translation = load_csv(raw_dir, EXPECTED_FILES["category_translation"])

    products = products.merge(category_translation, on="product_category_name", how="left")

    item_features = order_items.merge(
        products[["product_id", "product_category_name_english"]],
        on="product_id",
        how="left",
    )
    item_features = item_features.merge(
        sellers[["seller_id", "seller_state"]],
        on="seller_id",
        how="left",
    )

    item_agg = (
        item_features.groupby("order_id")
        .agg(
            items_count=("order_item_id", "count"),
            price_sum=("price", "sum"),
            freight_value_sum=("freight_value", "sum"),
            product_category_name_english=(
                "product_category_name_english",
                lambda values: values.mode().iloc[0] if not values.mode().empty else np.nan,
            ),
            seller_state=("seller_state", lambda values: values.mode().iloc[0] if not values.mode().empty else np.nan),
        )
        .reset_index()
    )

    payment_agg = (
        payments.groupby("order_id")
        .agg(
            payments_count=("payment_sequential", "count"),
            payment_value_sum=("payment_value", "sum"),
            payment_type_main=("payment_type", lambda values: values.mode().iloc[0] if not values.mode().empty else np.nan),
            payment_installments_max=("payment_installments", "max"),
        )
        .reset_index()
    )

    dataset = orders.merge(customers[["customer_id", "customer_state"]], on="customer_id", how="left")
    dataset = dataset.merge(item_agg, on="order_id", how="left")
    dataset = dataset.merge(payment_agg, on="order_id", how="left")

    dataset["delivery_time_days"] = (
        dataset["order_delivered_customer_date"] - dataset["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400.0
    dataset["same_state_flag"] = (dataset["customer_state"] == dataset["seller_state"]).astype("Int64")
    dataset["purchase_month"] = dataset["order_purchase_timestamp"].dt.month
    dataset["purchase_dayofweek"] = dataset["order_purchase_timestamp"].dt.dayofweek
    dataset["purchase_hour"] = dataset["order_purchase_timestamp"].dt.hour
    dataset["is_weekend"] = dataset["purchase_dayofweek"].isin([5, 6]).astype("Int64")

    dataset = dataset[dataset["order_delivered_customer_date"].notna()].copy()
    dataset = dataset[dataset["delivery_time_days"].notna()].copy()
    dataset = dataset[dataset["delivery_time_days"] >= 0].copy()

    long_delivery_threshold_q75 = float(dataset["delivery_time_days"].quantile(0.75))
    dataset["is_long_delivery"] = (dataset["delivery_time_days"] > long_delivery_threshold_q75).astype("Int64")
    dataset = dataset.sort_values("order_purchase_timestamp").reset_index(drop=True)

    LOGGER.info("Computed Q75 delivery time threshold: %.4f days", long_delivery_threshold_q75)

    selected_columns = [
        "order_id",
        "customer_id",
        "order_status",
        "order_purchase_timestamp",
        "customer_state",
        "seller_state",
        "product_category_name_english",
        "items_count",
        "price_sum",
        "freight_value_sum",
        "payments_count",
        "payment_value_sum",
        "payment_type_main",
        "payment_installments_max",
        "same_state_flag",
        "purchase_month",
        "purchase_dayofweek",
        "purchase_hour",
        "is_weekend",
        "delivery_time_days",
        "is_long_delivery",
    ]

    return dataset[selected_columns], long_delivery_threshold_q75


def inject_missing_values(data: pd.DataFrame, missing_rate: float, random_state: int) -> pd.DataFrame:
    if missing_rate <= 0:
        return data

    result = data.copy()
    rng = np.random.default_rng(random_state)
    target_columns = ["product_category_name_english", "payment_type_main", "freight_value_sum"]

    for column in target_columns:
        if column not in result.columns:
            continue
        sample_size = int(len(result) * missing_rate)
        if sample_size == 0:
            continue
        indices = rng.choice(result.index.to_numpy(), size=sample_size, replace=False)
        result.loc[indices, column] = np.nan

    return result


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    setup_logging(
        logs_dir=config["paths"]["logs_dir"],
        file_name=config["logging"]["file_name"],
        level=config["logging"]["level"],
    )

    raw_dir = Path(config["paths"]["raw_data_dir"])
    output_path = Path(config["data"]["prepared_dataset_path"])
    ensure_directory(output_path.parent)

    LOGGER.info("Preparing order-level dataset from raw Olist files in %s", raw_dir)
    dataset, long_delivery_threshold_q75 = build_order_level_dataset(raw_dir)
    dataset = inject_missing_values(
        dataset,
        missing_rate=args.missing_rate,
        random_state=config["project"]["random_state"],
    )

    dataset.to_csv(output_path, index=False)
    metadata_path = Path(config["paths"]["processed_data_dir"]) / "olist_ml_dataset_metadata.json"
    write_json(
        {
            "target_column": config["data"]["target_column"],
            "delivery_time_q75_days": long_delivery_threshold_q75,
            "rows": int(len(dataset)),
            "columns": list(dataset.columns),
        },
        metadata_path,
    )
    LOGGER.info("Prepared dataset saved to %s with %s rows and %s columns", output_path, len(dataset), len(dataset.columns))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
