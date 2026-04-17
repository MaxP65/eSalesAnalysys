from __future__ import annotations

import logging
from pathlib import Path

from src.utils.io import ensure_directory


def setup_logging(logs_dir: str | Path, file_name: str = "pipeline.log", level: str = "INFO") -> None:
    ensure_directory(logs_dir)
    log_path = Path(logs_dir) / file_name

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
