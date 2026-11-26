"""Unit tests for batch profiler, DuckDB storage, and idempotency."""

import json
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
import pytest

from src.profile.profiler import profile_batch


@pytest.fixture
def sample_profiler_df() -> pd.DataFrame:
    """Create a sample DataFrame with numeric, text, nulls, and duplicate rows."""
    base_df = pd.DataFrame(
        {
            "order_id": [f"ord_{i}" for i in range(95)],
            "price": [10.0 + i for i in range(85)] + [None] * 10,
            "payment_type": ["credit_card"] * 60 + ["boleto"] * 25 + ["voucher"] * 10,
            "customer_state": ["SP"] * 65 + ["RJ"] * 30,
        }
    )
    # Append 5 exact duplicate rows
    dupes = base_df.iloc[:5].copy()
    return pd.concat([base_df, dupes], ignore_index=True)


def test_profile_batch_metrics(tmp_path: Path, sample_profiler_df: pd.DataFrame) -> None:
    """Verify profile_batch calculates accurate batch and column metrics."""
    db_path = tmp_path / "test_profile.duckdb"
    res = profile_batch(sample_profiler_df, "2020-01-01", db_path=db_path)

    # Check batch metrics
    batch_res = res["batch"]
    assert batch_res["row_count"] == 100
    assert "order_id" in batch_res["columns"]
    assert batch_res["duplicate_row_share"] > 0

    # Check column metrics map
    cols_map = {c["column"]: c for c in res["columns"]}

    # Numeric column (price)
    price_col = cols_map["price"]
    assert price_col["kind"] == "numeric"
    assert np.isclose(price_col["null_share"], 0.10)
    assert price_col["min"] == 10.0
    assert price_col["p50"] > price_col["p05"]
    assert price_col["p95"] > price_col["p50"]

    # Text column (payment_type)
    pay_col = cols_map["payment_type"]
    assert pay_col["kind"] == "text"
    assert pay_col["null_share"] == 0.0
    assert pay_col["distinct_count"] == 3
    assert "credit_card" in pay_col["top_values"]
    assert np.isclose(pay_col["top_values"]["credit_card"], 0.65)


def test_profile_duckdb_storage(tmp_path: Path, sample_profiler_df: pd.DataFrame) -> None:
    """Verify DuckDB receives records in batch_profile, column_profile, and column_sample."""
    db_path = tmp_path / "test_profile.duckdb"
    profile_batch(sample_profiler_df, "2020-01-01", db_path=db_path)

    con = duckdb.connect(str(db_path))
    try:
        # Check batch_profile
        bp_row = con.execute("SELECT * FROM batch_profile WHERE batch_date = '2020-01-01'").fetchone()
        assert bp_row is not None
        assert bp_row[0] == "2020-01-01"
        assert bp_row[1] == 100  # row_count
        cols_parsed = json.loads(bp_row[2])
        assert "price" in cols_parsed

        # Check column_profile
        cp_rows = con.execute("SELECT * FROM column_profile WHERE batch_date = '2020-01-01'").fetchall()
        assert len(cp_rows) == 4  # 4 columns

        # Check column_sample (price has 90 non-null values, so exactly 90 sampled)
        cs_count = con.execute("SELECT COUNT(*) FROM column_sample WHERE batch_date = '2020-01-01'").fetchone()[0]
        assert cs_count == 90
    finally:
        con.close()


def test_profile_idempotent(tmp_path: Path, sample_profiler_df: pd.DataFrame) -> None:
    """Verify profiling the same date twice replaces records rather than duplicating them."""
    db_path = tmp_path / "test_profile.duckdb"

    # Profile once
    profile_batch(sample_profiler_df, "2020-01-01", db_path=db_path)

    # Profile same date again
    profile_batch(sample_profiler_df, "2020-01-01", db_path=db_path)

    con = duckdb.connect(str(db_path))
    try:
        bp_count = con.execute("SELECT COUNT(*) FROM batch_profile").fetchone()[0]
        assert bp_count == 1

        cp_count = con.execute("SELECT COUNT(*) FROM column_profile").fetchone()[0]
        assert cp_count == 4

        cs_count = con.execute("SELECT COUNT(*) FROM column_sample").fetchone()[0]
        assert cs_count == 90
    finally:
        con.close()
