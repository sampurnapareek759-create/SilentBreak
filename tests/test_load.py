"""Tests for DuckDB batch loading, idempotency, and schema resilience."""

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
        log_rows = con.execute(
            "SELECT batch_date, rows_loaded, status, error FROM load_log"
        ).fetchall()
        assert len(log_rows) == 1
        assert log_rows[0] == ("2020-01-01", 3, "loaded", None)
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


def test_load_batch_dropped_column_loads_with_nulls(tmp_path: Path) -> None:
    """Verify a batch with a dropped column loads successfully and missing values become NULL."""
    db_path = tmp_path / "test_drop.duckdb"
    batch_dir = tmp_path / "drop_batches"
    batch_dir.mkdir()

    # Batch missing 'customer_state'
    df = pd.DataFrame(
        {
            "order_id": ["drop_ord_1", "drop_ord_2"],
            "order_item_id": [1, 2],
            "customer_id": ["c1", "c2"],
            "order_status": ["delivered", "delivered"],
            "order_purchase_timestamp": [pd.Timestamp("2020-02-01")] * 2,
            "product_id": ["p1", "p2"],
            "seller_id": ["s1", "s2"],
            "price": [50.0, 75.0],
            "freight_value": [10.0, 12.0],
            "payment_type": ["credit_card", "boleto"],
            "payment_installments": [1, 1],
            "payment_value": [60.0, 87.0],
        }
    )
    df.to_parquet(batch_dir / "2020-02-01.parquet", index=False)

    rows_loaded = load_batch("2020-02-01", db_path=db_path, batch_dir=batch_dir)
    assert rows_loaded == 2

    con = duckdb.connect(str(db_path))
    try:
        results = con.execute(
            "SELECT customer_state FROM orders WHERE batch_date = '2020-02-01'"
        ).fetchall()
        assert results == [(None,), (None,)]

        log_row = con.execute(
            "SELECT status, error FROM load_log WHERE batch_date = '2020-02-01'"
        ).fetchone()
        assert log_row == ("loaded", None)
    finally:
        con.close()


def test_load_batch_renamed_column_rejected(tmp_path: Path) -> None:
    """Verify a batch with an unknown/renamed column is rejected without corrupting database."""
    db_path = tmp_path / "test_rename.duckdb"
    batch_dir = tmp_path / "rename_batches"
    batch_dir.mkdir()

    # Batch with 'price' renamed to 'item_price'
    df = pd.DataFrame(
        {
            "order_id": ["renamed_ord_1"],
            "order_item_id": [1],
            "customer_id": ["c1"],
            "customer_state": ["SP"],
            "order_status": ["delivered"],
            "order_purchase_timestamp": [pd.Timestamp("2020-02-02")],
            "product_id": ["p1"],
            "seller_id": ["s1"],
            "item_price": [100.0],  # Unknown column name
            "freight_value": [15.0],
            "payment_type": ["credit_card"],
            "payment_installments": [1],
            "payment_value": [115.0],
        }
    )
    df.to_parquet(batch_dir / "2020-02-02.parquet", index=False)

    rows_loaded = load_batch("2020-02-02", db_path=db_path, batch_dir=batch_dir)
    assert rows_loaded == 0

    con = duckdb.connect(str(db_path))
    try:
        orders_count = con.execute(
            "SELECT COUNT(*) FROM orders WHERE batch_date = '2020-02-02'"
        ).fetchone()[0]
        assert orders_count == 0

        log_row = con.execute(
            "SELECT status, error, rows_loaded FROM load_log WHERE batch_date = '2020-02-02'"
        ).fetchone()
        assert log_row[0] == "rejected"
        assert "item_price" in log_row[1]
        assert log_row[2] == 0
    finally:
        con.close()


def test_flow_continues_after_rejection(tmp_path: Path) -> None:
    """Verify pipeline continues processing subsequent batches even when one is rejected."""
    db_path = tmp_path / "test_progression.duckdb"
    batch_dir = tmp_path / "progression_batches"
    batch_dir.mkdir()

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

    # Valid batch 1 (2020-03-01)
    df1 = pd.DataFrame(
        [
            [
                "ord_1",
                1,
                "c1",
                "SP",
                "delivered",
                pd.Timestamp("2020-03-01"),
                "p1",
                "s1",
                50.0,
                10.0,
                "credit_card",
                1,
                60.0,
            ]
        ],
        columns=columns,
    )
    df1.to_parquet(batch_dir / "2020-03-01.parquet", index=False)

    # Broken batch 2 (2020-03-02) with unknown column
    df2 = df1.rename(columns={"price": "unexpected_price"})
    df2.to_parquet(batch_dir / "2020-03-02.parquet", index=False)

    # Valid batch 3 (2020-03-03)
    df3 = pd.DataFrame(
        [
            [
                "ord_3",
                1,
                "c3",
                "RJ",
                "delivered",
                pd.Timestamp("2020-03-03"),
                "p3",
                "s3",
                80.0,
                12.0,
                "boleto",
                1,
                92.0,
            ]
        ],
        columns=columns,
    )
    df3.to_parquet(batch_dir / "2020-03-03.parquet", index=False)

    # Ingest Batch 1
    assert next_unloaded_date(db_path=db_path, batch_dir=batch_dir) == "2020-03-01"
    assert load_batch("2020-03-01", db_path=db_path, batch_dir=batch_dir) == 1

    # Ingest Batch 2 (rejected, but moves forward)
    assert next_unloaded_date(db_path=db_path, batch_dir=batch_dir) == "2020-03-02"
    assert load_batch("2020-03-02", db_path=db_path, batch_dir=batch_dir) == 0

    # Ingest Batch 3 (advances successfully past rejection)
    assert next_unloaded_date(db_path=db_path, batch_dir=batch_dir) == "2020-03-03"
    assert load_batch("2020-03-03", db_path=db_path, batch_dir=batch_dir) == 1

    # No more batches
    assert next_unloaded_date(db_path=db_path, batch_dir=batch_dir) is None

    # Total loaded rows in orders table should be exactly 2 (Batch 1 + Batch 3)
    con = duckdb.connect(str(db_path))
    try:
        total_orders = con.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        assert total_orders == 2
    finally:
        con.close()
