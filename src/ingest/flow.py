"""Prefect workflow for daily batch ingestion.

Defines the daily_ingest flow and CLI to orchestrate batch discovery,
loading with retries, and logging.
"""

import argparse
from prefect import flow, get_run_logger, task

from src.ingest.load import load_batch, next_unloaded_date


@task(name="find_next_date")
def find_next_date_task() -> str | None:
    """Find the earliest batch date not yet loaded into DuckDB.

    Returns:
        Date string ('YYYY-MM-DD') or None if all batches are loaded.
    """
    return next_unloaded_date()


@task(name="load", retries=2, retry_delay_seconds=1)
def load_task(batch_date: str) -> int:
    """Load a single daily batch Parquet file into the DuckDB database.

    Args:
        batch_date: Date string ('YYYY-MM-DD') to load.

    Returns:
        Number of rows loaded.
    """
    return load_batch(batch_date)


@task(name="report")
def report_task(batch_date: str, rows_loaded: int) -> None:
    """Log the outcome of the batch load.

    Args:
        batch_date: Loaded batch date string.
        rows_loaded: Number of rows successfully inserted.
    """
    logger = get_run_logger()
    logger.info(f"Successfully loaded batch {batch_date}: {rows_loaded} rows.")


@flow(name="daily_ingest")
def daily_ingest() -> str | None:
    """Orchestrate loading the next available daily batch into DuckDB.

    Returns:
        The date string loaded, or None if no batches remain.
    """
    logger = get_run_logger()
    batch_date = find_next_date_task()

    if batch_date is None:
        logger.info("All batches loaded")
        return None

    rows_loaded = load_task(batch_date)
    report_task(batch_date, rows_loaded)
    return batch_date


def run_pipeline(days: int = 1) -> None:
    """Run daily_ingest for a specified number of days sequentially.

    Args:
        days: Number of days to simulate/ingest.
    """
    for day_idx in range(1, days + 1):
        loaded_date = daily_ingest()
        if loaded_date is None:
            break


def main() -> None:
    """Parse command line arguments and execute the flow runner."""
    parser = argparse.ArgumentParser(
        description="Run daily batch ingestion pipeline using Prefect."
    )
    parser.add_argument(
        "--days",
        type=int,
        default=1,
        help="Number of days (batches) to ingest in sequence (default: 1)",
    )
    args = parser.parse_args()
    run_pipeline(days=args.days)


if __name__ == "__main__":
    main()
