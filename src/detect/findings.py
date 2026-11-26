"""Detection findings data structures and DuckDB persistence.

Provides a unified Finding dataclass used across all detection layers
and an idempotent save_findings function.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
import duckdb

from src.config import DB_PATH


@dataclass
class Finding:
    """Represents a single data quality anomaly detected in a batch.

    Attributes:
        batch_date: Date string of the affected batch ('YYYY-MM-DD').
        layer: Detection layer identifier (e.g. 'schema', 'profile', 'drift').
        check: Specific check name (e.g. 'missing_column', 'likely_rename').
        column: Affected column name, or None for batch-level anomalies.
        severity: Anomaly severity tier ('low', 'medium', 'high').
        score: Normalized anomaly confidence score between 0.0 and 1.0.
        message: Plain-English sentence describing the issue.
        evidence: Dictionary containing supporting statistical/schema facts.
    """

    batch_date: str
    layer: str
    check: str
    column: str | None
    severity: str
    score: float
    message: str
    evidence: dict[str, Any]


def init_findings_table(con: duckdb.DuckDBPyConnection) -> None:
    """Ensure the findings table exists in DuckDB.

    Args:
        con: Active DuckDB connection.
    """
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS findings (
            batch_date VARCHAR,
            layer VARCHAR,
            "check" VARCHAR,
            "column" VARCHAR,
            severity VARCHAR,
            score DOUBLE,
            message VARCHAR,
            evidence JSON,
            created_at TIMESTAMP
        );
        """
    )


def save_findings(
    findings: list[Finding],
    db_path: Path | None = None,
    batch_date: str | None = None,
    layer: str | None = None,
) -> None:
    """Idempotently persist a collection of findings into DuckDB.

    Existing findings for the same batch_date and layer are cleared before
    inserting the new records.

    Args:
        findings: List of Finding instances to store.
        db_path: Optional DuckDB database path (defaults to DB_PATH).
        batch_date: Optional explicit batch date to clear if findings is empty.
        layer: Optional explicit layer name to clear if findings is empty.
    """
    target_db = db_path if db_path is not None else DB_PATH
    target_db.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(target_db))

    try:
        init_findings_table(con)
        con.execute("BEGIN TRANSACTION")

        # Identify all distinct (batch_date, layer) scopes to clear
        scopes: set[tuple[str, str]] = set()
        if batch_date and layer:
            scopes.add((batch_date, layer))
        for f in findings:
            scopes.add((f.batch_date, f.layer))

        for b_date, lyr in scopes:
            con.execute(
                "DELETE FROM findings WHERE batch_date = ? AND layer = ?",
                [b_date, lyr],
            )

        # Insert new findings
        for f in findings:
            evidence_json = json.dumps(f.evidence)
            con.execute(
                """
                INSERT INTO findings (
                    batch_date, layer, "check", "column", severity, score,
                    message, evidence, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, current_timestamp)
                """,
                [
                    f.batch_date,
                    f.layer,
                    f.check,
                    f.column,
                    f.severity,
                    f.score,
                    f.message,
                    evidence_json,
                ],
            )

        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.close()
