"""Configuration module for SilentBreak.

Loads environment variables and defines core filesystem paths used across the pipeline.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from a .env file located in the project root
load_dotenv()

# Project root directory
ROOT: Path = Path(__file__).resolve().parent.parent

# Data paths
RAW_DIR: Path = ROOT / "data" / "raw"
BATCH_DIR: Path = ROOT / "data" / "batches"
BATCHES_FAULTY_DIR: Path = ROOT / "data" / "batches_faulty"
QUARANTINE_DIR: Path = ROOT / "data" / "quarantine"
GROUND_TRUTH_PATH: Path = ROOT / "data" / "ground_truth.csv"

# Database path
DB_PATH: Path = ROOT / "silentbreak.duckdb"

# Active batch source setting: "faulty" or "clean"
BATCH_SOURCE: str = os.getenv("BATCH_SOURCE", "faulty")


def active_batch_dir() -> Path:
    """Return the active batch directory based on the BATCH_SOURCE setting.

    Returns:
        BATCH_DIR if BATCH_SOURCE is 'clean', otherwise BATCHES_FAULTY_DIR.
    """
    source = os.getenv("BATCH_SOURCE", BATCH_SOURCE).strip().lower()
    if source == "clean":
        return BATCH_DIR
    return BATCHES_FAULTY_DIR
