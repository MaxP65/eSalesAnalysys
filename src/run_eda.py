from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.data_analysis.eda import generate_eda_report
from src.utils.config import load_config
from src.utils.logging_utils import setup_logging


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate EDA report for the prepared dataset.")
    parser.add_argument("-config", default="configs/config.yaml", help="Path to config file.")
    parser.add_argument("--report-name", default="initial_eda", help="Markdown report name without extension.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    setup_logging(
        logs_dir=config["paths"]["logs_dir"],
        file_name=config["logging"]["file_name"],
        level=config["logging"]["level"],
    )

    dataset_path = Path(config["data"]["prepared_dataset_path"])
    if not dataset_path.exists():
        raise FileNotFoundError(f"Prepared dataset not found: {dataset_path}")

    LOGGER.info("Loading prepared dataset for EDA: %s", dataset_path)
    data = pd.read_csv(dataset_path)
    report_path = generate_eda_report(
        data=data,
        output_dir=Path(config["paths"]["reports_dir"]) / "eda",
        time_column=config["data"]["time_column"],
        target_column=config["data"]["target_column"],
        top_categories_limit=config["eda"]["top_categories_limit"],
        report_name=args.report_name,
    )
    LOGGER.info("EDA report saved to %s", report_path)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
