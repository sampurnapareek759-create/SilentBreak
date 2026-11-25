"""Tests for DuckDB batch loading and idempotency."""

from pathlib import Path
import duckdb
import pandas as pd
import pytest

from src.ingest.load import load_batch, next_unloaded_date


@pytest.fixture
def test_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """Create temporary directories and sample batch parquet files.

    Returns:
        Tuple of (temp_db_path, temp_batch_dir).
    """
    db_path = tmp_path / "test_silentbreak.duckdb"
    batch_dir = tmp_path / "batches"
    batch_dir.mkdir(parents=True, exist_ok=True)

    columns = [
        "order_id",
        "order_item_id",
        "customer_id",
        "customer_state",
        "order_status",
        "order_purchase_timestamp",
        "product_id",
        "seller_id",
        "price",
        "freight_value",
        "payment_type",
        "payment_installments",
        "payment_value",
    ]

    # Create batch 1 (2020-01-01) with 3 rows
    df1 = pd.DataFrame(
        [
            [
                f"order_{i}",
                1,
                f"cust_{i}",
                "SP",
                "delivered",
                pd.Timestamp("2020-01-01 10:00:00"),
                f"prod_{i}",
                f"seller_{i}",
                100.0 + i,
                15.0,
                "credit_card",
                1,
                115.0 + i,
            ]
            for i in range(3)
        ],
        columns=columns,
    )
    df1.to_parquet(batch_dir / "2020-01-01.parquet", index=False)

    # Create batch 2 (2020-01-02) with 2 rows
    df2 = pd.DataFrame(
        [
            [
                f"order_b_{i}",
                1,
                f"cust_b_{i}",
                "RJ",
                "delivered",
                pd.Timestamp("2020-01-02 12:00:00"),
                f"prod_b_{i}",
                f"seller_b_{i}",
                50.0 + i,
                10.0,
                "boleto",
                1,
                60.0 + i,
            ]
            for i in range(2)
        ],
        columns=columns,
    )
    df2.to_parquet(batch_dir / "2020-01-02.parquet", index=False)

    return db_path, batch_dir


def test_load_batch_once(test_dirs: tuple[Path, Path]) -> None:
    """Verify loading a batch creates tables, inserts rows, and logs the load."""
    db_path, batch_dir = test_dirs

    loaded_count = load_batch("2020-01-01", db_path=db_path, batch_dir=batch_dir)
    assert loaded_count == 3

    con = duckdb.connect(str(db_path))
    try:
        # Check orders table
        orders_count = con.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        assert orders_count == 3

        # Check load_log table
        log_rows = con.execute("SELECT batch_date, rows_loaded FROM load_log").fetchall()
        assert len(log_rows) == 1
        assert log_rows[0] == ("2020-01-01", 3)
    finally:
        con.close()


def test_load_batch_idempotent(test_dirs: tuple[Path, Path]) -> None:
    """Verify loading the same date twice yields identical row counts without duplication."""
    db_path, batch_dir = test_dirs

    count_1 = load_batch("2020-01-01", db_path=db_path, batch_dir=batch_dir)
    count_2 = load_batch("2020-01-01", db_path=db_path, batch_dir=batch_dir)

    assert count_1 == count_2 == 3

    con = duckdb.connect(str(db_path))
    try:
        orders_count = con.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        assert orders_count == 3

        log_count = con.execute("SELECT COUNT(*) FROM load_log").fetchone()[0]
        assert log_count == 1
    finally:
        con.close()


def test_next_unloaded_date_moves_forward(test_dirs: tuple[Path, Path]) -> None:
    """Verify next_unloaded_date returns the earliest date and advances as batches are loaded."""
    db_path, batch_dir = test_dirs

    # Initially, neither batch is loaded; 2020-01-01 is earliest
    assert next_unloaded_date(db_path=db_path, batch_dir=batch_dir) == "2020-01-01"

    # Load 2020-01-01; next unloaded date should advance to 2020-01-02
    load_batch("2020-01-01", db_path=db_path, batch_dir=batch_dir)
    assert next_unloaded_date(db_path=db_path, batch_dir=batch_dir) == "2020-01-02"

    # Load 2020-01-02; all batches are loaded, so should return None
    load_batch("2020-01-02", db_path=db_path, batch_dir=batch_dir)
    assert next_unloaded_date(db_path=db_path, batch_dir=batch_dir) is None
