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
QUARANTINE_DIR: Path = ROOT / "data" / "quarantine"

# Database path
DB_PATH: Path = ROOT / "silentbreak.duckdb"
