"""Shared helpers: config loading and SparkSession construction."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

# Project root = one level above /src
ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | os.PathLike = "config/config.yaml") -> dict:
    """Load the YAML config, resolving relative paths against the project root."""
    cfg_path = ROOT / path
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    # Resolve every path in the `paths` block to an absolute path.
    for key, value in cfg["paths"].items():
        cfg["paths"][key] = str(ROOT / value)
    return cfg


def get_spark(cfg: dict):
    """Build a local SparkSession from config. Imported lazily so that the
    modeling/EDA steps don't require Spark if run on their own."""
    from pyspark.sql import SparkSession

    spark_cfg = cfg["spark"]
    spark = (
        SparkSession.builder.appName(spark_cfg["app_name"])
        .master(spark_cfg["master"])
        .config("spark.driver.memory", spark_cfg["driver_memory"])
        .config("spark.sql.shuffle.partitions", spark_cfg["shuffle_partitions"])
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def ensure_dirs(cfg: dict) -> None:
    """Create output directories if they don't already exist."""
    for key in ("processed_dir", "figures_dir", "metrics_dir"):
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)
