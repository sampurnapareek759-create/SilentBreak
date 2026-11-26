"""Unit tests for Detection Layer 1 (Schema Checks) and findings persistence."""

import json
from pathlib import Path
import duckdb
import pytest

from src.detect.findings import Finding, save_findings
from src.detect.schema import check_schema, run_all_schema_checks


def seed_fake_profile(
    con: duckdb.DuckDBPyConnection,
    batch_date: str,
    columns: list[str],
    dtypes: dict[str, str],
    price_p50: float = 100.0,
    price_col_name: str = "price",
    state_vals: dict[str, float] | None = None,
) -> None:
    """Helper to seed fake batch_profile and column_profile records."""
    con.execute(
        """
        INSERT INTO batch_profile (
            batch_date, row_count, columns, dtypes, duplicate_row_share, profiled_at
        ) VALUES (?, ?, ?, ?, ?, current_timestamp)
        """,
        [batch_date, 100, json.dumps(columns), json.dumps(dtypes), 0.0],
    )

    for col in columns:
        if col == price_col_name:
            con.execute(
                """
                INSERT INTO column_profile (
                    batch_date, "column", kind, null_share, mean, std, min, max,
                    p05, p25, p50, p75, p95, distinct_count, top_values
                ) VALUES (?, ?, 'numeric', 0.0, 100.0, 15.0, 10.0, 200.0,
                          ?, ?, ?, ?, ?, NULL, NULL)
                """,
                [
                    batch_date,
                    col,
                    price_p50 * 0.2,
                    price_p50 * 0.5,
                    price_p50,
                    price_p50 * 1.5,
                    price_p50 * 1.9,
                ],
            )
        elif col in ("customer_state", "state"):
            top_v = state_vals or {"SP": 0.7, "RJ": 0.3}
            con.execute(
                """
                INSERT INTO column_profile (
                    batch_date, "column", kind, null_share, mean, std, min, max,
                    p05, p25, p50, p75, p95, distinct_count, top_values
                ) VALUES (?, ?, 'text', 0.0, NULL, NULL, NULL, NULL,
                          NULL, NULL, NULL, NULL, NULL, 2, ?)
                """,
                [batch_date, col, json.dumps(top_v)],
            )
        else:
            con.execute(
                """
                INSERT INTO column_profile (
                    batch_date, "column", kind, null_share, mean, std, min, max,
                    p05, p25, p50, p75, p95, distinct_count, top_values
                ) VALUES (?, ?, 'text', 0.0, NULL, NULL, NULL, NULL,
                          NULL, NULL, NULL, NULL, NULL, 10, NULL)
                """,
                [batch_date, col],
            )


@pytest.fixture
def test_db_con(tmp_path: Path) -> tuple[Path, duckdb.DuckDBPyConnection]:
    """Create a temporary DuckDB database initialized with profile tables."""
    db_path = tmp_path / "schema_test.duckdb"
    con = duckdb.connect(str(db_path))

    con.execute(
        """
        CREATE TABLE batch_profile (
            batch_date VARCHAR PRIMARY KEY,
            row_count BIGINT,
            columns JSON,
            dtypes JSON,
            duplicate_row_share DOUBLE,
            profiled_at TIMESTAMP
        );
        CREATE TABLE column_profile (
            batch_date VARCHAR,
            "column" VARCHAR,
            kind VARCHAR,
            null_share DOUBLE,
            mean DOUBLE,
            std DOUBLE,
            min DOUBLE,
            max DOUBLE,
            p05 DOUBLE,
            p25 DOUBLE,
            p50 DOUBLE,
            p75 DOUBLE,
            p95 DOUBLE,
            distinct_count BIGINT,
            top_values JSON
        );
        """
    )
    yield db_path, con
    con.close()


def test_skip_with_insufficient_baseline(test_db_con: tuple[Path, duckdb.DuckDBPyConnection]) -> None:
    """Verify check_schema skips anomaly detection if fewer than 7 earlier profiles exist."""
    db_path, con = test_db_con

    # Seed only 5 baseline batches
    for day in range(1, 6):
        date_str = f"2020-01-{day:02d}"
        seed_fake_profile(con, date_str, ["order_id", "price"], {"order_id": "object", "price": "float64"})

    # Check 6th batch (which has a missing column)
    target_date = "2020-01-06"
    seed_fake_profile(con, target_date, ["order_id"], {"order_id": "object"})

    findings = check_schema(target_date, db_path=db_path)
    assert findings == []


def test_detect_missing_column(test_db_con: tuple[Path, duckdb.DuckDBPyConnection]) -> None:
    """Verify check_schema detects an unmatched missing column with high severity."""
    db_path, con = test_db_con

    # Seed 8 baseline batches
    for day in range(1, 9):
        date_str = f"2020-01-{day:02d}"
        seed_fake_profile(con, date_str, ["order_id", "price", "customer_state"], {"order_id": "object", "price": "float64", "customer_state": "object"})

    # 9th batch misses 'customer_state'
    target_date = "2020-01-09"
    seed_fake_profile(con, target_date, ["order_id", "price"], {"order_id": "object", "price": "float64"})

    findings = check_schema(target_date, db_path=db_path)
    assert len(findings) == 1
    f = findings[0]
    assert f.check == "missing_column"
    assert f.column == "customer_state"
    assert f.severity == "high"
    assert f.score == 1.0


def test_detect_unexpected_column(test_db_con: tuple[Path, duckdb.DuckDBPyConnection]) -> None:
    """Verify check_schema detects an unexpected new column with medium severity."""
    db_path, con = test_db_con

    # Seed 8 baseline batches
    for day in range(1, 9):
        date_str = f"2020-01-{day:02d}"
        seed_fake_profile(con, date_str, ["order_id", "price"], {"order_id": "object", "price": "float64"})

    # 9th batch introduces 'discount'
    target_date = "2020-01-09"
    seed_fake_profile(con, target_date, ["order_id", "price", "discount"], {"order_id": "object", "price": "float64", "discount": "float64"})

    findings = check_schema(target_date, db_path=db_path)
    assert len(findings) == 1
    f = findings[0]
    assert f.check == "unexpected_column"
    assert f.column == "discount"
    assert f.severity == "medium"
    assert f.score == 0.6


def test_detect_dtype_change(test_db_con: tuple[Path, duckdb.DuckDBPyConnection]) -> None:
    """Verify check_schema flags a changed data type with high severity."""
    db_path, con = test_db_con

    for day in range(1, 9):
        date_str = f"2020-01-{day:02d}"
        seed_fake_profile(con, date_str, ["order_id", "price"], {"order_id": "object", "price": "float64"})

    # 9th batch has 'price' as object (text)
    target_date = "2020-01-09"
    seed_fake_profile(con, target_date, ["order_id", "price"], {"order_id": "object", "price": "object"})

    findings = check_schema(target_date, db_path=db_path)
    assert len(findings) == 1
    f = findings[0]
    assert f.check == "dtype_change"
    assert f.column == "price"
    assert f.severity == "high"
    assert f.score == 0.9


def test_detect_likely_rename(test_db_con: tuple[Path, duckdb.DuckDBPyConnection]) -> None:
    """Verify a missing column and a matching new column produce ONE likely_rename finding."""
    db_path, con = test_db_con

    # Seed 8 baseline batches with price
    for day in range(1, 9):
        date_str = f"2020-01-{day:02d}"
        seed_fake_profile(con, date_str, ["order_id", "price"], {"order_id": "object", "price": "float64"}, price_p50=100.0)

    # 9th batch misses 'price' and introduces 'item_price' with matching distribution (p50=102.0)
    target_date = "2020-01-09"
    seed_fake_profile(
        con,
        target_date,
        ["order_id", "item_price"],
        {"order_id": "object", "item_price": "float64"},
        price_p50=102.0,
        price_col_name="item_price",
    )

    findings = check_schema(target_date, db_path=db_path)
    # Must report exactly ONE finding: likely_rename
    assert len(findings) == 1
    f = findings[0]
    assert f.check == "likely_rename"
    assert f.column == "price"
    assert f.severity == "high"
    assert f.score == 1.0
    assert "item_price" in f.message


def test_save_findings_idempotency(tmp_path: Path) -> None:
    """Verify save_findings saves records and replaces them upon rerun."""
    db_path = tmp_path / "findings_test.duckdb"

    f1 = Finding(
        batch_date="2020-01-01",
        layer="schema",
        check="missing_column",
        column="price",
        severity="high",
        score=1.0,
        message="Column 'price' is missing.",
        evidence={"column": "price"},
    )

    # First save
    save_findings([f1], db_path=db_path)

    con = duckdb.connect(str(db_path))
    try:
        rows = con.execute("SELECT * FROM findings").fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "2020-01-01"
        assert rows[0][1] == "schema"
    finally:
        con.close()

    # Second save with updated finding for same batch_date and layer
    f2 = Finding(
        batch_date="2020-01-01",
        layer="schema",
        check="likely_rename",
        column="price",
        severity="high",
        score=1.0,
        message="Likely rename.",
        evidence={"old": "price", "new": "item_price"},
    )
    save_findings([f2], db_path=db_path)

    con = duckdb.connect(str(db_path))
    try:
        rows = con.execute('SELECT "check", message FROM findings').fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "likely_rename"
    finally:
        con.close()
