"""Data loading module for SilentBreak.

Loads daily Parquet batch files into a DuckDB analytical database with
idempotent transactions and audit logging.
"""

from pathlib import Path
import duckdb

from src.config import BATCH_DIR, DB_PATH


def init_db(con: duckdb.DuckDBPyConnection, sample_parquet: Path) -> None:
    """Initialize required database tables if they do not already exist.

    Args:
        con: Active DuckDB connection.
        sample_parquet: Path to a Parquet file used to infer the orders schema.
    """
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS load_log (
            batch_date VARCHAR PRIMARY KEY,
            rows_loaded BIGINT,
            loaded_at TIMESTAMP
        );
        """
    )

    parquet_str = str(sample_parquet.resolve())
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS orders AS
        SELECT *, ''::VARCHAR AS batch_date
        FROM read_parquet('{parquet_str}')
        WHERE 1=0;
        """
    )


def load_batch(
    batch_date: str,
    db_path: Path | None = None,
    batch_dir: Path | None = None,
) -> int:
    """Load a single daily batch Parquet file into the DuckDB orders table.

    The operation is idempotent: it deletes any existing records for the given
    batch_date before inserting new ones inside a single ACID transaction.

    Args:
        batch_date: Target batch date string in 'YYYY-MM-DD' format.
        db_path: Optional DuckDB database file path (defaults to DB_PATH).
        batch_dir: Optional batches directory path (defaults to BATCH_DIR).

    Returns:
        Number of rows loaded into the orders table.

    Raises:
        FileNotFoundError: If the batch Parquet file does not exist.
    """
    target_db = db_path if db_path is not None else DB_PATH
    target_batch_dir = batch_dir if batch_dir is not None else BATCH_DIR

    batch_file = target_batch_dir / f"{batch_date}.parquet"
    if not batch_file.exists():
        raise FileNotFoundError(f"Batch file not found: {batch_file}")

    target_db.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(target_db))

    try:
        init_db(con, batch_file)

        con.execute("BEGIN TRANSACTION")
        # Idempotent load: delete existing records for this batch date
        con.execute("DELETE FROM orders WHERE batch_date = ?", [batch_date])

        # Insert new records from the Parquet file with batch_date added
        parquet_str = str(batch_file.resolve())
        con.execute(
            f"""
            INSERT INTO orders
            SELECT *, ? AS batch_date
            FROM read_parquet('{parquet_str}')
            """,
            [batch_date],
        )

        rows_loaded = con.execute(
            "SELECT COUNT(*) FROM orders WHERE batch_date = ?", [batch_date]
        ).fetchone()[0]

        # Log the load in load_log
        con.execute("DELETE FROM load_log WHERE batch_date = ?", [batch_date])
        con.execute(
            """
            INSERT INTO load_log (batch_date, rows_loaded, loaded_at)
            VALUES (?, ?, current_timestamp)
            """,
            [batch_date, rows_loaded],
        )

        con.execute("COMMIT")
        return int(rows_loaded)
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.close()


def next_unloaded_date(
    db_path: Path | None = None,
    batch_dir: Path | None = None,
) -> str | None:
    """Identify the earliest available batch date that has not yet been loaded.

    Args:
        db_path: Optional DuckDB database file path (defaults to DB_PATH).
        batch_dir: Optional batches directory path (defaults to BATCH_DIR).

    Returns:
        The earliest unloaded batch date string ('YYYY-MM-DD'), or None if all are loaded.
    """
    target_db = db_path if db_path is not None else DB_PATH
    target_batch_dir = batch_dir if batch_dir is not None else BATCH_DIR

    if not target_batch_dir.exists():
        return None

    batch_files = sorted(target_batch_dir.glob("*.parquet"))
    if not batch_files:
        return None

    loaded_dates: set[str] = set()
    if target_db.exists():
        con = duckdb.connect(str(target_db))
        try:
            table_check = con.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'load_log'"
            ).fetchone()
            if table_check and table_check[0] > 0:
                rows = con.execute("SELECT batch_date FROM load_log").fetchall()
                loaded_dates = {row[0] for row in rows}
        finally:
            con.close()

    for batch_file in batch_files:
        date_str = batch_file.stem
        if date_str not in loaded_dates:
            return date_str

    return None
