"""Unit tests for synthetic fault injection and determinism."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.faults.inject import (
    category_drift,
    corrupt_batches,
    drop_column,
    duplicate_rows,
    null_spike,
    partial_batch,
    rename_column,
    type_change,
    unit_change,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a standardized DataFrame for testing fault functions."""
    return pd.DataFrame(
        {
            "order_id": [f"ord_{i}" for i in range(100)],
            "price": [10.0 + i for i in range(100)],
            "freight_value": [5.0] * 100,
            "payment_value": [15.0 + i for i in range(100)],
            "payment_type": ["credit_card"] * 80 + ["boleto"] * 20,
            "customer_state": ["MG"] * 60 + ["RJ"] * 40,
        }
    )


def test_rename_column(sample_df: pd.DataFrame) -> None:
    """Test rename_column renames an expected column."""
    rng = np.random.default_rng(0)
    broken, details = rename_column(sample_df, rng)

    col = details["column"]
    assert col not in broken.columns
    assert len(broken) == len(sample_df)


def test_drop_column(sample_df: pd.DataFrame) -> None:
    """Test drop_column completely removes one target column."""
    rng = np.random.default_rng(0)
    broken, details = drop_column(sample_df, rng)

    col = details["column"]
    assert col not in broken.columns
    assert len(broken.columns) == len(sample_df.columns) - 1


def test_null_spike(sample_df: pd.DataFrame) -> None:
    """Test null_spike introduces 30% to 80% null values in one column."""
    rng = np.random.default_rng(0)
    broken, details = null_spike(sample_df, rng)

    col = details["column"]
    null_count = broken[col].isna().sum()
    null_pct = null_count / len(sample_df)
    assert 0.30 <= null_pct <= 0.80


def test_unit_change(sample_df: pd.DataFrame) -> None:
    """Test unit_change scales price or payment_value by 100 or 1/1000."""
    rng = np.random.default_rng(0)
    broken, details = unit_change(sample_df, rng)

    col = details["column"]
    ratio = broken[col].iloc[0] / sample_df[col].iloc[0]
    assert np.isclose(ratio, 100.0) or np.isclose(ratio, 0.001)


def test_duplicate_rows(sample_df: pd.DataFrame) -> None:
    """Test duplicate_rows adds 10% to 40% duplicate rows."""
    rng = np.random.default_rng(0)
    broken, details = duplicate_rows(sample_df, rng)

    added_rows = len(broken) - len(sample_df)
    pct = added_rows / len(sample_df)
    assert 0.10 <= pct <= 0.40


def test_category_drift(sample_df: pd.DataFrame) -> None:
    """Test category_drift increases prevalence of voucher or SP to 70%-100%."""
    rng = np.random.default_rng(0)
    broken, details = category_drift(sample_df, rng)

    col = details["column"]
    if col == "payment_type":
        target = "voucher"
    else:
        target = "SP"

    target_count = (broken[col] == target).sum()
    assert target_count >= int(0.70 * len(sample_df))


def test_partial_batch(sample_df: pd.DataFrame) -> None:
    """Test partial_batch keeps only 5% to 20% of rows."""
    rng = np.random.default_rng(0)
    broken, details = partial_batch(sample_df, rng)

    retention_pct = len(broken) / len(sample_df)
    assert 0.05 <= retention_pct <= 0.20


def test_type_change(sample_df: pd.DataFrame) -> None:
    """Test type_change converts numeric price to currency string."""
    rng = np.random.default_rng(0)
    broken, details = type_change(sample_df, rng)

    assert pd.api.types.is_string_dtype(broken["price"])
    assert broken["price"].iloc[0].startswith("R$ ")


def test_corrupt_batches_determinism(tmp_path: Path, sample_df: pd.DataFrame) -> None:
    """Test that corrupt_batches produces identical outputs and ground truth for the same seed."""
    batch_dir = tmp_path / "batches"
    batch_dir.mkdir()

    # Create 3 baseline batches and 3 post-baseline batches
    dates = [
        "2017-01-01",
        "2017-02-01",
        "2017-02-15",
        "2017-03-05",
        "2017-03-10",
        "2017-03-15",
    ]
    for d in dates:
        sample_df.to_parquet(batch_dir / f"{d}.parquet", index=False)

    faulty_dir_1 = tmp_path / "faulty_1"
    gt_path_1 = tmp_path / "gt_1.csv"
    gt1 = corrupt_batches(
        fault_rate=0.8,
        seed=123,
        batch_dir=batch_dir,
        faulty_dir=faulty_dir_1,
        ground_truth_path=gt_path_1,
    )

    faulty_dir_2 = tmp_path / "faulty_2"
    gt_path_2 = tmp_path / "gt_2.csv"
    gt2 = corrupt_batches(
        fault_rate=0.8,
        seed=123,
        batch_dir=batch_dir,
        faulty_dir=faulty_dir_2,
        ground_truth_path=gt_path_2,
    )

    # Ground truth files must be identical
    pd.testing.assert_frame_equal(gt1, gt2)
    assert gt_path_1.read_text() == gt_path_2.read_text()

    # Pre-baseline batches must remain clean
    baseline_df = pd.read_parquet(faulty_dir_1 / "2017-01-01.parquet")
    pd.testing.assert_frame_equal(baseline_df, sample_df)
