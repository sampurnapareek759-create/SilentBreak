"""Configuration module for SilentBreak.

Loads environment variables and defines core filesystem paths used across the pipeline.
"""

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
