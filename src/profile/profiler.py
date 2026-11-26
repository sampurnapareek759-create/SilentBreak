"""Data profiling module for SilentBreak.

Extracts batch-level and column-level statistical profiles (distributions,
null shares, percentiles, distinct counts, and random samples) and persists
them idempotently into DuckDB analytical tables.
"""

import json
from pathlib import Path
from typing import Any
import duckdb
import numpy as np
import pandas as pd

from src.config import DB_PATH


def init_profile_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Ensure profile storage tables exist in the DuckDB database.

    Args:
        con: Active DuckDB connection.
    """
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS batch_profile (
            batch_date VARCHAR PRIMARY KEY,
            row_count BIGINT,
            columns JSON,
            dtypes JSON,
            duplicate_row_share DOUBLE,
            profiled_at TIMESTAMP
        );
        """
    )

    con.execute(
        """
        CREATE TABLE IF NOT EXISTS column_profile (
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

    con.execute(
        """
        CREATE TABLE IF NOT EXISTS column_sample (
            batch_date VARCHAR,
            "column" VARCHAR,
            value DOUBLE
        );
        """
    )


def compute_batch_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """Calculate batch-level structural metrics.

    Args:
        df: Input DataFrame.

    Returns:
        Dictionary containing row_count, columns, dtypes, and duplicate_row_share.
    """
    row_count = len(df)
    columns = list(df.columns)
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    dup_share = (
        float(df.duplicated().sum() / row_count) if row_count > 0 else 0.0
    )

    return {
        "row_count": row_count,
        "columns": columns,
        "dtypes": dtypes,
        "duplicate_row_share": round(dup_share, 6),
    }


def compute_column_metrics(
    df: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[tuple[str, float]]]:
    """Calculate individual column statistics and collect numeric samples.

    Args:
        df: Input DataFrame.

    Returns:
        Tuple of (list of column profile dicts, list of (column, sample_value) tuples).
    """
    column_profiles: list[dict[str, Any]] = []
    samples: list[tuple[str, float]] = []

    total_rows = len(df)

    for col in df.columns:
        s = df[col]
        null_count = int(s.isna().sum())
        null_share = (
            float(null_count / total_rows) if total_rows > 0 else 0.0
        )
        valid = s.dropna()

        is_numeric = pd.api.types.is_numeric_dtype(
            s.dtype
        ) and not pd.api.types.is_bool_dtype(s.dtype)

        if is_numeric:
            kind = "numeric"
            if len(valid) > 0:
                mean_val = float(valid.mean())
                std_val = (
                    float(valid.std(ddof=0)) if len(valid) > 1 else 0.0
                )
                min_val = float(valid.min())
                max_val = float(valid.max())
                quantiles = valid.quantile(
                    [0.05, 0.25, 0.50, 0.75, 0.95]
                ).to_dict()
                p05 = float(quantiles.get(0.05, min_val))
                p25 = float(quantiles.get(0.25, min_val))
                p50 = float(quantiles.get(0.50, min_val))
                p75 = float(quantiles.get(0.75, max_val))
                p95 = float(quantiles.get(0.95, max_val))

                sample_size = min(len(valid), 2000)
                sampled_series = valid.sample(n=sample_size, random_state=0)
                for val in sampled_series:
                    samples.append((col, float(val)))
            else:
                mean_val = std_val = min_val = max_val = None
                p05 = p25 = p50 = p75 = p95 = None

            column_profiles.append(
                {
                    "column": col,
                    "kind": kind,
                    "null_share": round(null_share, 6),
                    "mean": mean_val,
                    "std": std_val,
                    "min": min_val,
                    "max": max_val,
                    "p05": p05,
                    "p25": p25,
                    "p50": p50,
                    "p75": p75,
                    "p95": p95,
                    "distinct_count": None,
                    "top_values": None,
                }
            )
        else:
            kind = "text"
            distinct_count = int(valid.nunique())
            if len(valid) > 0:
                val_counts = (
                    valid.astype(str).value_counts(normalize=True).head(10)
                )
                top_values = {
                    str(k): round(float(v), 6) for k, v in val_counts.items()
                }
            else:
                top_values = {}

            column_profiles.append(
                {
                    "column": col,
                    "kind": kind,
                    "null_share": round(null_share, 6),
                    "mean": None,
                    "std": None,
                    "min": None,
                    "max": None,
                    "p05": None,
                    "p25": None,
                    "p50": None,
                    "p75": None,
                    "p95": None,
                    "distinct_count": distinct_count,
                    "top_values": top_values,
                }
            )

    return column_profiles, samples


def save_profile_to_db(
    batch_date: str,
    batch_metrics: dict[str, Any],
    column_profiles: list[dict[str, Any]],
    samples: list[tuple[str, float]],
    db_path: Path,
) -> None:
    """Idempotently insert profile metrics and samples into DuckDB.

    Args:
        batch_date: Ingested batch date string.
        batch_metrics: Batch-level metrics dictionary.
        column_profiles: List of column-level metrics dictionaries.
        samples: List of (column, sampled_value) tuples.
        db_path: Target DuckDB database path.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))

    try:
        init_profile_tables(con)

        con.execute("BEGIN TRANSACTION")
        # Idempotency: clear previous profile entries for this batch date
        con.execute(
            "DELETE FROM batch_profile WHERE batch_date = ?", [batch_date]
        )
        con.execute(
            'DELETE FROM column_profile WHERE batch_date = ?', [batch_date]
        )
        con.execute(
            'DELETE FROM column_sample WHERE batch_date = ?', [batch_date]
        )

        # Insert batch_profile
        con.execute(
            """
            INSERT INTO batch_profile (
                batch_date, row_count, columns, dtypes, duplicate_row_share, profiled_at
            ) VALUES (?, ?, ?, ?, ?, current_timestamp)
            """,
            [
                batch_date,
                batch_metrics["row_count"],
                json.dumps(batch_metrics["columns"]),
                json.dumps(batch_metrics["dtypes"]),
                batch_metrics["duplicate_row_share"],
            ],
        )

        # Insert column_profile records
        for cp in column_profiles:
            top_vals_json = (
                json.dumps(cp["top_values"])
                if cp["top_values"] is not None
                else None
            )
            con.execute(
                """
                INSERT INTO column_profile (
                    batch_date, "column", kind, null_share, mean, std, min, max,
                    p05, p25, p50, p75, p95, distinct_count, top_values
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    batch_date,
                    cp["column"],
                    cp["kind"],
                    cp["null_share"],
                    cp["mean"],
                    cp["std"],
                    cp["min"],
                    cp["max"],
                    cp["p05"],
                    cp["p25"],
                    cp["p50"],
                    cp["p75"],
                    cp["p95"],
                    cp["distinct_count"],
                    top_vals_json,
                ],
            )

        # Insert column_sample records
        if samples:
            sample_rows = [(batch_date, col, val) for col, val in samples]
            con.executemany(
                'INSERT INTO column_sample (batch_date, "column", value) VALUES (?, ?, ?)',
                sample_rows,
            )

        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.close()


def profile_batch(
    df: pd.DataFrame,
    batch_date: str,
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Profile a single batch DataFrame and store the results in DuckDB.

    Args:
        df: Input batch DataFrame.
        batch_date: Date string of the batch ('YYYY-MM-DD').
        db_path: Optional DuckDB database path (defaults to DB_PATH).

    Returns:
        Structured profile dictionary containing batch and column metrics.
    """
    target_db = db_path if db_path is not None else DB_PATH

    batch_metrics = compute_batch_metrics(df)
    column_profiles, samples = compute_column_metrics(df)

    save_profile_to_db(
        batch_date=batch_date,
        batch_metrics=batch_metrics,
        column_profiles=column_profiles,
        samples=samples,
        db_path=target_db,
    )

    return {
        "batch_date": batch_date,
        "batch": batch_metrics,
        "columns": column_profiles,
        "samples_count": len(samples),
    }
