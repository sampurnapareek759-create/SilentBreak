"""Synthetic fault injection module for SilentBreak.

Simulates real-world data pipeline breakages (schema drifts, missing columns,
null spikes, unit errors, duplicates, category drifts, truncated batches, and type changes)
to benchmark pipeline monitoring and automated healing.
"""

import argparse
from pathlib import Path
from typing import Any, Callable
import numpy as np
import pandas as pd

from src.config import BATCH_DIR, BATCHES_FAULTY_DIR, GROUND_TRUTH_PATH

BASELINE_CUTOFF: str = "2017-03-01"


def rename_column(
    df: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rename one numerical or categorical column to an unexpected name.

    Args:
        df: Input DataFrame.
        rng: Numpy random Generator instance.

    Returns:
        Tuple of (corrupted_df, details_dict).
    """
    candidates = ["price", "freight_value", "payment_value", "payment_type"]
    available = [col for col in candidates if col in df.columns]
    target_col = str(rng.choice(available))

    rename_map = {
        "price": "item_price",
        "freight_value": "shipping_fee",
        "payment_value": "total_paid",
        "payment_type": "payment_method",
    }
    new_name = rename_map.get(target_col, f"{target_col}_renamed")

    broken = df.rename(columns={target_col: new_name})
    details = f"renamed column '{target_col}' to '{new_name}'"
    return broken, {"column": target_col, "details": details}


def drop_column(
    df: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drop one essential column entirely.

    Args:
        df: Input DataFrame.
        rng: Numpy random Generator instance.

    Returns:
        Tuple of (corrupted_df, details_dict).
    """
    candidates = ["customer_state", "payment_type", "freight_value"]
    available = [col for col in candidates if col in df.columns]
    target_col = str(rng.choice(available))

    broken = df.drop(columns=[target_col])
    details = f"dropped column '{target_col}'"
    return broken, {"column": target_col, "details": details}


def null_spike(
    df: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Inject null values into 30% to 80% of rows for a single column.

    Args:
        df: Input DataFrame.
        rng: Numpy random Generator instance.

    Returns:
        Tuple of (corrupted_df, details_dict).
    """
    candidates = ["price", "freight_value", "payment_value", "customer_state"]
    available = [col for col in candidates if col in df.columns]
    target_col = str(rng.choice(available))

    fraction = float(rng.uniform(0.3, 0.8))
    broken = df.copy()

    num_nulls = max(1, int(len(broken) * fraction))
    null_indices = rng.choice(broken.index, size=num_nulls, replace=False)
    broken.loc[null_indices, target_col] = None

    details = f"injected {fraction:.1%} nulls into column '{target_col}' ({num_nulls} rows)"
    return broken, {"column": target_col, "details": details}


def unit_change(
    df: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Multiply price/payment by 100 (e.g. cents) or divide by 1000.

    Args:
        df: Input DataFrame.
        rng: Numpy random Generator instance.

    Returns:
        Tuple of (corrupted_df, details_dict).
    """
    candidates = ["price", "payment_value"]
    available = [col for col in candidates if col in df.columns]
    target_col = str(rng.choice(available))

    operation = str(rng.choice(["multiply_100", "divide_1000"]))
    broken = df.copy()

    if operation == "multiply_100":
        broken[target_col] = broken[target_col] * 100.0
        details = f"multiplied column '{target_col}' by 100"
    else:
        broken[target_col] = broken[target_col] / 1000.0
        details = f"divided column '{target_col}' by 1000"

    return broken, {"column": target_col, "details": details}


def duplicate_rows(
    df: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Duplicate 10% to 40% of rows in the batch.

    Args:
        df: Input DataFrame.
        rng: Numpy random Generator instance.

    Returns:
        Tuple of (corrupted_df, details_dict).
    """
    fraction = float(rng.uniform(0.1, 0.4))
    num_dupes = max(1, int(len(df) * fraction))

    dupe_indices = rng.choice(df.index, size=num_dupes, replace=False)
    duplicates = df.loc[dupe_indices]
    broken = pd.concat([df, duplicates], ignore_index=True)

    details = f"duplicated {num_dupes} rows ({fraction:.1%})"
    return broken, {"column": "", "details": details}


def category_drift(
    df: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drift categorical distributions by replacing 70% to 100% with a single value.

    Args:
        df: Input DataFrame.
        rng: Numpy random Generator instance.

    Returns:
        Tuple of (corrupted_df, details_dict).
    """
    options = [("payment_type", "voucher"), ("customer_state", "SP")]
    col_idx = int(rng.choice(len(options)))
    target_col, target_value = options[col_idx]

    fraction = float(rng.uniform(0.7, 1.0))
    broken = df.copy()

    num_drift = max(1, int(len(broken) * fraction))
    drift_indices = rng.choice(broken.index, size=num_drift, replace=False)
    broken.loc[drift_indices, target_col] = target_value

    details = f"overwrote {fraction:.1%} of '{target_col}' with '{target_value}'"
    return broken, {"column": target_col, "details": details}


def partial_batch(
    df: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Truncate the batch, retaining only 5% to 20% of the original rows.

    Args:
        df: Input DataFrame.
        rng: Numpy random Generator instance.

    Returns:
        Tuple of (corrupted_df, details_dict).
    """
    fraction = float(rng.uniform(0.05, 0.20))
    keep_count = max(1, int(len(df) * fraction))

    kept_indices = rng.choice(df.index, size=keep_count, replace=False)
    broken = df.loc[kept_indices].copy().reset_index(drop=True)

    details = f"truncated batch: kept {keep_count}/{len(df)} rows ({fraction:.1%})"
    return broken, {"column": "", "details": details}


def type_change(
    df: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Convert price column from float to currency strings (e.g. 'R$ 12.50').

    Args:
        df: Input DataFrame.
        rng: Numpy random Generator instance.

    Returns:
        Tuple of (corrupted_df, details_dict).
    """
    target_col = "price"
    broken = df.copy()
    broken[target_col] = broken[target_col].apply(lambda x: f"R$ {float(x):.2f}")

    details = f"converted '{target_col}' values to currency text formatted 'R$ X.XX'"
    return broken, {"column": target_col, "details": details}


FAULT_REGISTRY: list[tuple[str, Callable[[pd.DataFrame, np.random.Generator], tuple[pd.DataFrame, dict[str, Any]]]]] = [
    ("rename_column", rename_column),
    ("drop_column", drop_column),
    ("null_spike", null_spike),
    ("unit_change", unit_change),
    ("duplicate_rows", duplicate_rows),
    ("category_drift", category_drift),
    ("partial_batch", partial_batch),
    ("type_change", type_change),
]


def corrupt_batches(
    fault_rate: float = 0.25,
    seed: int = 42,
    batch_dir: Path | None = None,
    faulty_dir: Path | None = None,
    ground_truth_path: Path | None = None,
) -> pd.DataFrame:
    """Generate faulty batch dataset for pipeline testing and evaluation.

    Batches strictly before 2017-03-01 remain completely uncorrupted (clean baseline).
    Later batches have a probability of `fault_rate` of having exactly one fault injected.

    Args:
        fault_rate: Probability of injecting a fault into a non-baseline batch.
        seed: Random seed ensuring reproducibility.
        batch_dir: Source batch directory (defaults to BATCH_DIR).
        faulty_dir: Destination directory for faulty/clean batches (defaults to BATCHES_FAULTY_DIR).
        ground_truth_path: Destination CSV path for fault ground truth (defaults to GROUND_TRUTH_PATH).

    Returns:
        DataFrame containing ground truth records of all injected faults.
    """
    src_dir = batch_dir if batch_dir is not None else BATCH_DIR
    dst_dir = faulty_dir if faulty_dir is not None else BATCHES_FAULTY_DIR
    gt_path = ground_truth_path if ground_truth_path is not None else GROUND_TRUTH_PATH

    dst_dir.mkdir(parents=True, exist_ok=True)
    gt_path.parent.mkdir(parents=True, exist_ok=True)

    # Clean destination directory
    for existing_file in dst_dir.glob("*.parquet"):
        existing_file.unlink()

    batch_files = sorted(src_dir.glob("*.parquet"))
    rng = np.random.default_rng(seed)

    ground_truth_records: list[dict[str, str]] = []

    for batch_file in batch_files:
        batch_date = batch_file.stem
        df = pd.read_parquet(batch_file)

        # Baseline period stays untouched
        if batch_date < BASELINE_CUTOFF:
            df.to_parquet(dst_dir / f"{batch_date}.parquet", index=False)
            continue

        # Decide whether to inject a fault
        if rng.random() < fault_rate:
            fault_idx = int(rng.choice(len(FAULT_REGISTRY)))
            fault_type, fault_fn = FAULT_REGISTRY[fault_idx]
            corrupted_df, details_dict = fault_fn(df, rng)

            corrupted_df.to_parquet(dst_dir / f"{batch_date}.parquet", index=False)
            ground_truth_records.append(
                {
                    "batch_date": batch_date,
                    "fault_type": fault_type,
                    "column": str(details_dict.get("column", "")),
                    "details": str(details_dict.get("details", "")),
                }
            )
        else:
            df.to_parquet(dst_dir / f"{batch_date}.parquet", index=False)

    gt_df = pd.DataFrame(
        ground_truth_records,
        columns=["batch_date", "fault_type", "column", "details"],
    )
    gt_df.to_csv(gt_path, index=False)

    return gt_df


def main() -> None:
    """CLI runner for fault generation."""
    parser = argparse.ArgumentParser(
        description="Inject realistic synthetic faults into pipeline batches."
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=0.25,
        help="Probability of injecting a fault into batches after baseline (default: 0.25)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic generation (default: 42)",
    )
    args = parser.parse_args()

    print(f"Injecting faults with rate={args.rate} and seed={args.seed}...")
    gt_df = corrupt_batches(fault_rate=args.rate, seed=args.seed)
    print(f"Corrupted batches written to: {BATCHES_FAULTY_DIR}")
    print(f"Ground truth written to: {GROUND_TRUTH_PATH}")
    print(f"Total faults injected: {len(gt_df)}")
    print("\nFault breakdown by type:")
    print(gt_df["fault_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
