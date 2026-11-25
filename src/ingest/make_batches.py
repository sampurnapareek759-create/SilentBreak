"""Batch generation module for SilentBreak.

Extracts order item data from raw Olist CSV files, enriches items with order,
customer, and payment information, and splits the data into daily Parquet batches.
"""

from pathlib import Path
import pandas as pd

from src.config import BATCH_DIR, RAW_DIR

ORDER_ITEM_COLUMNS: list[str] = [
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


def clean_batch_dir(batch_dir: Path) -> int:
    """Delete existing Parquet files in the batch directory for reproducible runs.

    Args:
        batch_dir: Directory containing batch files.

    Returns:
        Number of files deleted.
    """
    if not batch_dir.exists():
        return 0

    deleted_count = 0
    for file_path in batch_dir.glob("*.parquet"):
        file_path.unlink()
        deleted_count += 1
    return deleted_count


def load_raw_data(
    raw_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load required Olist CSV datasets from raw data directory.

    Args:
        raw_dir: Directory containing raw CSV files.

    Returns:
        Tuple of (orders_df, items_df, payments_df, customers_df).
    """
    orders = pd.read_csv(raw_dir / "olist_orders_dataset.csv")
    items = pd.read_csv(raw_dir / "olist_order_items_dataset.csv")
    payments = pd.read_csv(raw_dir / "olist_order_payments_dataset.csv")
    customers = pd.read_csv(raw_dir / "olist_customers_dataset.csv")
    return orders, items, payments, customers


def deduplicate_payments(payments_df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the payment with the highest payment_value for each order.

    Ties are deterministically broken by choosing the lowest payment_sequential
    using a stable sort (mergesort).

    Args:
        payments_df: Raw payments DataFrame containing order_id, payment_value,
            and payment_sequential.

    Returns:
        DataFrame with at most one payment record per order_id.
    """
    sorted_payments = payments_df.sort_values(
        by=["payment_value", "payment_sequential"],
        ascending=[False, True],
        kind="mergesort",
    )
    return sorted_payments.drop_duplicates(subset=["order_id"], keep="first")


def build_order_items_table(
    orders: pd.DataFrame,
    items: pd.DataFrame,
    payments: pd.DataFrame,
    customers: pd.DataFrame,
) -> pd.DataFrame:
    """Join datasets to create a single table with one row per order item.

    Args:
        orders: Orders DataFrame.
        items: Order items DataFrame.
        payments: Deduplicated payments DataFrame.
        customers: Customers DataFrame.

    Returns:
        Merged DataFrame containing all specified order item columns with native datetimes.
    """
    # Merge items with orders to get customer_id, order_status, and timestamp
    merged = items.merge(
        orders[
            [
                "order_id",
                "customer_id",
                "order_status",
                "order_purchase_timestamp",
            ]
        ],
        on="order_id",
        how="inner",
    )

    # Convert order_purchase_timestamp to a real datetime column
    merged["order_purchase_timestamp"] = pd.to_datetime(
        merged["order_purchase_timestamp"]
    )

    # Merge with customers to get customer_state
    merged = merged.merge(
        customers[["customer_id", "customer_state"]],
        on="customer_id",
        how="left",
    )

    # Merge with deduplicated payments
    merged = merged.merge(
        payments[
            ["order_id", "payment_type", "payment_installments", "payment_value"]
        ],
        on="order_id",
        how="left",
    )

    return merged[ORDER_ITEM_COLUMNS]


def save_daily_batches(
    df: pd.DataFrame,
    batch_dir: Path,
    min_rows: int = 20,
) -> tuple[int, str, str, int]:
    """Group rows by purchase date and save daily Parquet files.

    Days with fewer than `min_rows` rows are skipped.

    Args:
        df: Enriched order items DataFrame.
        batch_dir: Target directory for Parquet batch files.
        min_rows: Minimum row count threshold to write a batch file.

    Returns:
        Tuple of (number_of_files_written, first_date, last_date, total_rows_written).
    """
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Extract date strings from the datetime column for grouping
    working_df = df.copy()
    working_df["batch_date"] = working_df["order_purchase_timestamp"].dt.strftime("%Y-%m-%d")

    # Group by date and identify valid batches
    grouped = working_df.groupby("batch_date")
    valid_dates: list[str] = []
    total_rows = 0

    for date_str, group in grouped:
        if len(group) < min_rows:
            continue

        valid_dates.append(str(date_str))
        total_rows += len(group)

        output_path = batch_dir / f"{date_str}.parquet"
        group[ORDER_ITEM_COLUMNS].to_parquet(output_path, index=False)

    if not valid_dates:
        return 0, "", "", 0

    valid_dates.sort()
    first_date = valid_dates[0]
    last_date = valid_dates[-1]
    files_written = len(valid_dates)

    return files_written, first_date, last_date, total_rows


def main() -> None:
    """Execute the batch ingestion process."""
    print(f"Cleaning existing Parquet files in {BATCH_DIR}...")
    deleted = clean_batch_dir(BATCH_DIR)
    print(f"Removed {deleted} previous batch file(s).")

    print("Loading raw CSV datasets...")
    orders, items, payments, customers = load_raw_data(RAW_DIR)

    print("Deduplicating payments (keeping highest payment per order, deterministic tie-breaking)...")
    deduped_payments = deduplicate_payments(payments)

    print("Building unified order items table...")
    unified_df = build_order_items_table(
        orders=orders,
        items=items,
        payments=deduped_payments,
        customers=customers,
    )

    print(f"Writing daily batches to {BATCH_DIR} (skipping days < 20 rows)...")
    files_written, first_date, last_date, total_rows = save_daily_batches(
        df=unified_df,
        batch_dir=BATCH_DIR,
        min_rows=20,
    )

    print(f"Number of files written: {files_written}")
    print(f"First date: {first_date}")
    print(f"Last date: {last_date}")
    print(f"Total rows: {total_rows}")


if __name__ == "__main__":
    main()
