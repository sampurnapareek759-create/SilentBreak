"""Database reset utility for SilentBreak.

Deletes the DuckDB database file and any associated WAL journal files to allow
a clean pipeline restart.
"""

from pathlib import Path
from src.config import DB_PATH


def reset_database(db_path: Path | None = None) -> bool:
    """Delete the specified DuckDB file and associated temporary files.

    Args:
        db_path: Optional DuckDB path (defaults to DB_PATH).

    Returns:
        True if an existing database was found and deleted, False otherwise.
    """
    target = db_path if db_path is not None else DB_PATH
    deleted = False

    # Remove primary database file
    if target.exists():
        target.unlink()
        deleted = True

    # Remove write-ahead log (WAL) if present
    wal_path = target.with_suffix(target.suffix + ".wal")
    if wal_path.exists():
        wal_path.unlink()

    return deleted


def main() -> None:
    """Execute database reset from command line."""
    was_deleted = reset_database(DB_PATH)
    if was_deleted:
        print(f"Database reset successful: deleted {DB_PATH}")
    else:
        print(f"No existing database found at {DB_PATH}. Ready for fresh start.")


if __name__ == "__main__":
    main()
