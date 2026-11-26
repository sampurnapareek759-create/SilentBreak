"""Data loading module for SilentBreak.

Loads daily Parquet batch files into a DuckDB analytical database with
idempotent transactions, schema resilience (INSERT BY NAME), and audit logging.
"""

from pathlib import Path
import duckdb

from src.config import DB_PATH, active_batch_dir


def init_db(con: duckdb.DuckDBPyConnection) -> None:
    """Initialize canonical database tables if they do not already exist.

    Args:
        con: Active DuckDB connection.
    """
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS load_log (
            batch_date VARCHAR PRIMARY KEY,
            rows_loaded BIGINT,
            loaded_at TIMESTAMP,
            status VARCHAR,
            error VARCHAR
        );
        """
    )

    # Ensure status and error columns exist for backwards compatibility
    for col_name in ["status", "error"]:
        try:
            con.execute(f"ALTER TABLE load_log ADD COLUMN {col_name} VARCHAR")
        except Exception:
            pass

    # Canonical orders table schema
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS orders (
            order_id VARCHAR,
            order_item_id BIGINT,
            customer_id VARCHAR,
            customer_state VARCHAR,
            order_status VARCHAR,
            order_purchase_timestamp TIMESTAMP,
            product_id VARCHAR,
            seller_id VARCHAR,
            price DOUBLE,
            freight_value DOUBLE,
            payment_type VARCHAR,
            payment_installments BIGINT,
            payment_value DOUBLE,
            batch_date VARCHAR
        );
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
    Uses 'INSERT INTO orders BY NAME' so missing columns become NULL instead
    of shifting columns. If schema corruption causes the insert to fail,
    the transaction is rolled back and the batch is recorded as 'rejected'.

    Args:
        batch_date: Target batch date string in 'YYYY-MM-DD' format.
        db_path: Optional DuckDB database file path (defaults to DB_PATH).
        batch_dir: Optional batches directory path (defaults to active_batch_dir()).

    Returns:
        Number of rows loaded into the orders table (0 if rejected).

    Raises:
        FileNotFoundError: If the batch Parquet file does not exist.
    """
    target_db = db_path if db_path is not None else DB_PATH
    target_batch_dir = batch_dir if batch_dir is not None else active_batch_dir()

    batch_file = target_batch_dir / f"{batch_date}.parquet"
    if not batch_file.exists():
        raise FileNotFoundError(f"Batch file not found: {batch_file}")

    target_db.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(target_db))

    try:
        init_db(con)

        con.execute("BEGIN TRANSACTION")
        # Idempotent load: delete existing records for this batch date
        con.execute("DELETE FROM orders WHERE batch_date = ?", [batch_date])

        parquet_str = str(batch_file.resolve())
        try:
            # Insert by column name: missing columns get NULL, unknown columns raise error
            con.execute(
                f"""
                INSERT INTO orders BY NAME
                SELECT *, ? AS batch_date
                FROM read_parquet('{parquet_str}')
                """,
                [batch_date],
            )

            rows_loaded = con.execute(
                "SELECT COUNT(*) FROM orders WHERE batch_date = ?", [batch_date]
            ).fetchone()[0]

            # Log successful load in load_log
            con.execute("DELETE FROM load_log WHERE batch_date = ?", [batch_date])
            con.execute(
                """
                INSERT INTO load_log (batch_date, rows_loaded, loaded_at, status, error)
                VALUES (?, ?, current_timestamp, 'loaded', NULL)
                """,
                [batch_date, rows_loaded],
            )

            con.execute("COMMIT")
            return int(rows_loaded)
        except Exception as err:
            con.execute("ROLLBACK")
            short_error = str(err).splitlines()[0] if str(err) else "Schema violation"

            # Record rejection in a separate transaction so the batch is marked processed
            con.execute("BEGIN TRANSACTION")
            con.execute("DELETE FROM orders WHERE batch_date = ?", [batch_date])
            con.execute("DELETE FROM load_log WHERE batch_date = ?", [batch_date])
            con.execute(
                """
                INSERT INTO load_log (batch_date, rows_loaded, loaded_at, status, error)
                VALUES (?, 0, current_timestamp, 'rejected', ?)
                """,
                [batch_date, short_error],
            )
            con.execute("COMMIT")
            return 0
    finally:
        con.close()


def next_unloaded_date(
    db_path: Path | None = None,
    batch_dir: Path | None = None,
) -> str | None:
    """Identify the earliest available batch date that has not yet been processed.

    Both 'loaded' and 'rejected' batches in load_log are considered processed.

    Args:
        db_path: Optional DuckDB database file path (defaults to DB_PATH).
        batch_dir: Optional batches directory path (defaults to active_batch_dir()).

    Returns:
        The earliest unprocessed batch date string ('YYYY-MM-DD'), or None if all are processed.
    """
    target_db = db_path if db_path is not None else DB_PATH
    target_batch_dir = batch_dir if batch_dir is not None else active_batch_dir()

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
