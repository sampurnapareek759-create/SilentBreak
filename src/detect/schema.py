"""Schema anomaly detection (Detection Layer 1).

Monitors batch structural integrity against a rolling 14-day baseline.
Detects missing columns, unexpected columns, data type mutations, and
correlates column renames by comparing statistical distribution signatures.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any
import duckdb
import numpy as np

from src.config import DB_PATH
from src.detect.findings import Finding, save_findings


def get_batch_profile_data(
    con: duckdb.DuckDBPyConnection, batch_date: str
) -> tuple[dict[str, Any] | None, dict[str, dict[str, Any]]]:
    """Retrieve profile and column stats for a specific batch date.

    Args:
        con: Active DuckDB connection.
        batch_date: Date string of the batch.

    Returns:
        Tuple of (batch_profile_dict, dict_of_column_profiles).
    """
    bp_row = con.execute(
        "SELECT row_count, columns, dtypes, duplicate_row_share FROM batch_profile WHERE batch_date = ?",
        [batch_date],
    ).fetchone()

    if not bp_row:
        return None, {}

    batch_meta = {
        "row_count": bp_row[0],
        "columns": json.loads(bp_row[1]),
        "dtypes": json.loads(bp_row[2]),
        "duplicate_row_share": bp_row[3],
    }

    cp_rows = con.execute(
        """
        SELECT "column", kind, null_share, mean, std, min, max,
               p05, p25, p50, p75, p95, distinct_count, top_values
        FROM column_profile
        WHERE batch_date = ?
        """,
        [batch_date],
    ).fetchall()

    col_stats: dict[str, dict[str, Any]] = {}
    for r in cp_rows:
        col_stats[r[0]] = {
            "kind": r[1],
            "null_share": r[2],
            "mean": r[3],
            "std": r[4],
            "min": r[5],
            "max": r[6],
            "p05": r[7],
            "p25": r[8],
            "p50": r[9],
            "p75": r[10],
            "p95": r[11],
            "distinct_count": r[12],
            "top_values": json.loads(r[13]) if r[13] is not None else None,
        }

    return batch_meta, col_stats


def get_baseline_data(
    con: duckdb.DuckDBPyConnection, batch_date: str, window_size: int = 14
) -> list[tuple[dict[str, Any], dict[str, dict[str, Any]]]]:
    """Fetch profile data for up to `window_size` batches strictly prior to batch_date.

    Args:
        con: Active DuckDB connection.
        batch_date: Anchor batch date string.
        window_size: Number of previous batches to query (default: 14).

    Returns:
        List of (batch_meta, col_stats) tuples for baseline batches.
    """
    prev_dates = con.execute(
        """
        SELECT batch_date
        FROM batch_profile
        WHERE batch_date < ?
        ORDER BY batch_date DESC
        LIMIT ?
        """,
        [batch_date, window_size],
    ).fetchall()

    baseline_data: list[tuple[dict[str, Any], dict[str, dict[str, Any]]]] = []
    for (d,) in prev_dates:
        b_meta, c_stats = get_batch_profile_data(con, d)
        if b_meta is not None:
            baseline_data.append((b_meta, c_stats))

    return baseline_data


def compute_expected_schema(
    baseline_data: list[tuple[dict[str, Any], dict[str, dict[str, Any]]]],
) -> tuple[set[str], dict[str, str], dict[str, dict[str, Any]]]:
    """Compute expected columns, typical data types, and baseline signatures.

    Args:
        baseline_data: Historical baseline profile data.

    Returns:
        Tuple of (expected_columns_set, expected_dtypes_dict, column_signatures_dict).
    """
    n_batches = len(baseline_data)
    col_counts: Counter[str] = Counter()
    col_dtypes: dict[str, list[str]] = {}
    col_signatures: dict[str, dict[str, Any]] = {}

    for b_meta, c_stats in baseline_data:
        for col in b_meta["columns"]:
            col_counts[col] += 1
            dtype = b_meta["dtypes"].get(col, "unknown")
            col_dtypes.setdefault(col, []).append(dtype)

    # Columns appearing in majority (>50%) of baseline batches
    expected_columns: set[str] = {
        col for col, count in col_counts.items() if count > (n_batches / 2)
    }

    expected_dtypes: dict[str, str] = {}
    for col in expected_columns:
        most_common_dtype = Counter(col_dtypes[col]).most_common(1)[0][0]
        expected_dtypes[col] = most_common_dtype

    # Aggregate distribution signatures across baseline for rename matching
    for col in expected_columns:
        p05_vals: list[float] = []
        p50_vals: list[float] = []
        p95_vals: list[float] = []
        top_keys: set[str] = set()
        col_kind = "unknown"

        for _, c_stats in baseline_data:
            stats = c_stats.get(col)
            if not stats:
                continue
            col_kind = stats["kind"]
            if stats["kind"] == "numeric":
                if stats["p05"] is not None:
                    p05_vals.append(stats["p05"])
                if stats["p50"] is not None:
                    p50_vals.append(stats["p50"])
                if stats["p95"] is not None:
                    p95_vals.append(stats["p95"])
            elif stats["kind"] == "text" and stats["top_values"]:
                top_keys.update(stats["top_values"].keys())

        col_signatures[col] = {
            "kind": col_kind,
            "p05": float(np.median(p05_vals)) if p05_vals else None,
            "p50": float(np.median(p50_vals)) if p50_vals else None,
            "p95": float(np.median(p95_vals)) if p95_vals else None,
            "top_keys": top_keys,
        }

    return expected_columns, expected_dtypes, col_signatures


def is_rename_candidate(
    missing_col: str,
    new_col: str,
    base_sig: dict[str, Any],
    new_stats: dict[str, Any],
) -> bool:
    """Determine whether an unexpected column matches the signature of a missing column.

    Args:
        missing_col: Name of the expected column missing from the batch.
        new_col: Name of the unexpected column present in the batch.
        base_sig: Baseline statistical signature of missing_col.
        new_stats: Profile statistics of new_col in the current batch.

    Returns:
        True if statistics closely align, indicating a likely column rename.
    """
    if base_sig["kind"] != new_stats.get("kind"):
        return False

    if base_sig["kind"] == "numeric":
        base_p05 = base_sig["p05"]
        base_p50 = base_sig["p50"]
        base_p95 = base_sig["p95"]
        curr_p05 = new_stats.get("p05")
        curr_p50 = new_stats.get("p50")
        curr_p95 = new_stats.get("p95")

        if None in (base_p05, base_p50, base_p95, curr_p05, curr_p50, curr_p95):
            return False

        # Compute relative difference across quantiles
        diffs = [
            abs(curr_p05 - base_p05) / max(abs(base_p05), 1.0),
            abs(curr_p50 - base_p50) / max(abs(base_p50), 1.0),
            abs(curr_p95 - base_p95) / max(abs(base_p95), 1.0),
        ]
        return float(np.mean(diffs)) < 0.35

    if base_sig["kind"] == "text":
        base_keys = base_sig.get("top_keys", set())
        curr_top = new_stats.get("top_values") or {}
        curr_keys = set(curr_top.keys())

        if not base_keys or not curr_keys:
            return False

        intersection = len(base_keys & curr_keys)
        union = len(base_keys | curr_keys)
        return (intersection / union) >= 0.50

    return False


def check_schema(
    batch_date: str, db_path: Path | None = None
) -> list[Finding]:
    """Execute Layer 1 Schema checks against rolling baseline.

    Args:
        batch_date: Date string of the batch to inspect ('YYYY-MM-DD').
        db_path: Optional DuckDB database path (defaults to DB_PATH).

    Returns:
        List of Finding instances detailing schema anomalies.
    """
    target_db = db_path if db_path is not None else DB_PATH
    if not target_db.exists():
        return []

    con = duckdb.connect(str(target_db))
    findings: list[Finding] = []

    try:
        current_meta, current_col_stats = get_batch_profile_data(
            con, batch_date
        )
        if current_meta is None:
            return []

        baseline_data = get_baseline_data(con, batch_date, window_size=14)
        # Skip if fewer than 7 earlier profile snapshots exist
        if len(baseline_data) < 7:
            return []

        expected_columns, expected_dtypes, col_signatures = (
            compute_expected_schema(baseline_data)
        )

        curr_columns = set(current_meta["columns"])
        curr_dtypes = current_meta["dtypes"]

        missing_columns = expected_columns - curr_columns
        unexpected_columns = curr_columns - expected_columns
        common_columns = curr_columns & expected_columns

        # Check for data type mutations on shared columns
        for col in sorted(common_columns):
            exp_dtype = expected_dtypes.get(col)
            actual_dtype = curr_dtypes.get(col)
            if exp_dtype and actual_dtype and exp_dtype != actual_dtype:
                findings.append(
                    Finding(
                        batch_date=batch_date,
                        layer="schema",
                        check="dtype_change",
                        column=col,
                        severity="high",
                        score=0.9,
                        message=f"Column '{col}' changed data type from '{exp_dtype}' to '{actual_dtype}'.",
                        evidence={
                            "expected_dtype": exp_dtype,
                            "actual_dtype": actual_dtype,
                        },
                    )
                )

        # Correlation & Rename detection
        unmatched_missing = set(missing_columns)
        unmatched_unexpected = set(unexpected_columns)

        for m_col in sorted(missing_columns):
            for u_col in sorted(unexpected_columns):
                if u_col not in unmatched_unexpected:
                    continue
                base_sig = col_signatures.get(m_col, {})
                new_stats = current_col_stats.get(u_col, {})

                if is_rename_candidate(m_col, u_col, base_sig, new_stats):
                    findings.append(
                        Finding(
                            batch_date=batch_date,
                            layer="schema",
                            check="likely_rename",
                            column=m_col,
                            severity="high",
                            score=1.0,
                            message=f"Likely rename from '{m_col}' to '{u_col}'.",
                            evidence={
                                "original_column": m_col,
                                "renamed_column": u_col,
                            },
                        )
                    )
                    unmatched_missing.discard(m_col)
                    unmatched_unexpected.discard(u_col)
                    break

        # Flag remaining unmatched missing columns
        for m_col in sorted(unmatched_missing):
            findings.append(
                Finding(
                    batch_date=batch_date,
                    layer="schema",
                    check="missing_column",
                    column=m_col,
                    severity="high",
                    score=1.0,
                    message=f"Expected column '{m_col}' is missing from the batch.",
                    evidence={"column": m_col},
                )
            )

        # Flag remaining unmatched unexpected columns
        for u_col in sorted(unmatched_unexpected):
            findings.append(
                Finding(
                    batch_date=batch_date,
                    layer="schema",
                    check="unexpected_column",
                    column=u_col,
                    severity="medium",
                    score=0.6,
                    message=f"Unexpected column '{u_col}' found in the batch.",
                    evidence={"column": u_col},
                )
            )

        return findings
    finally:
        con.close()


def run_all_schema_checks(db_path: Path | None = None) -> list[Finding]:
    """Execute schema checks across all available batch profiles in DuckDB.

    Args:
        db_path: Optional DuckDB database path (defaults to DB_PATH).

    Returns:
        List of all generated findings.
    """
    target_db = db_path if db_path is not None else DB_PATH
    if not target_db.exists():
        print(f"Database not found at {target_db}.")
        return []

    con = duckdb.connect(str(target_db))
    try:
        dates = [
            r[0]
            for r in con.execute(
                "SELECT batch_date FROM batch_profile ORDER BY batch_date"
            ).fetchall()
        ]
    finally:
        con.close()

    all_findings: list[Finding] = []
    for b_date in dates:
        findings = check_schema(b_date, db_path=target_db)
        save_findings(
            findings, db_path=target_db, batch_date=b_date, layer="schema"
        )
        for f in findings:
            print(
                f"[{f.batch_date}] [{f.severity.upper()}] {f.check} ({f.column}): {f.message}"
            )
            all_findings.append(f)

    return all_findings


def main() -> None:
    """CLI runner for schema detection checks."""
    parser = argparse.ArgumentParser(
        description="Run Layer 1 Schema checks on pipeline batch profiles."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run schema checks across all profiled batches in DuckDB",
    )
    args = parser.parse_args()

    if args.all:
        print("Running schema checks on all profiled batches...")
        findings = run_all_schema_checks()
        print(f"Schema detection complete. Total findings: {len(findings)}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
