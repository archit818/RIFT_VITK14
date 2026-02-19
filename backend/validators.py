"""
Input validation module for transaction CSV uploads.
Validates schema, data types, business rules, and edge cases.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, List, Dict, Any


class ValidationError:
    """Represents a single validation error."""
    def __init__(self, error_type: str, message: str, row: int = None, column: str = None):
        self.error_type = error_type
        self.message = message
        self.row = row
        self.column = column

    def to_dict(self) -> Dict[str, Any]:
        result = {"error_type": self.error_type, "message": self.message}
        if self.row is not None:
            result["row"] = self.row
        if self.column is not None:
            result["column"] = self.column
        return result


REQUIRED_COLUMNS = ["transaction_id", "sender_id", "receiver_id", "amount", "timestamp"]


def validate_csv(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Validate a transaction DataFrame.
    Returns (cleaned_df, errors_list).
    If errors are critical, cleaned_df may be empty.
    """
    errors = []
    warnings = []

    # --- Schema validation ---
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(ValidationError(
            "MISSING_COLUMNS",
            f"Missing required columns: {', '.join(missing_cols)}"
        ))
        return pd.DataFrame(), [e.to_dict() for e in errors]

    if df.empty:
        errors.append(ValidationError("EMPTY_FILE", "The uploaded CSV contains no data rows."))
        return pd.DataFrame(), [e.to_dict() for e in errors]

    original_count = len(df)

    # --- Missing values ---
    null_mask = df[REQUIRED_COLUMNS].isnull().any(axis=1)
    null_count = null_mask.sum()
    if null_count > 0:
        null_rows = df[null_mask].index.tolist()[:10]
        warnings.append(ValidationError(
            "MISSING_VALUES",
            f"{null_count} rows have missing values (rows: {null_rows}...). Dropping them.",
        ))
        df = df[~null_mask].copy()

    if df.empty:
        errors.append(ValidationError("ALL_ROWS_INVALID", "All rows had missing values."))
        return pd.DataFrame(), [e.to_dict() for e in errors + warnings]

    # --- Duplicate transaction IDs ---
    dup_mask = df.duplicated(subset=["transaction_id"], keep="first")
    dup_count = dup_mask.sum()
    if dup_count > 0:
        warnings.append(ValidationError(
            "DUPLICATE_TRANSACTIONS",
            f"{dup_count} duplicate transaction_id(s) found. Keeping first occurrence."
        ))
        df = df[~dup_mask].copy()

    # --- Amount validation ---
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    bad_amount = df["amount"].isna()
    if bad_amount.sum() > 0:
        warnings.append(ValidationError(
            "INVALID_AMOUNTS",
            f"{bad_amount.sum()} rows have non-numeric amounts. Dropping."
        ))
        df = df[~bad_amount].copy()

    non_positive = df["amount"] <= 0
    if non_positive.sum() > 0:
        warnings.append(ValidationError(
            "NON_POSITIVE_AMOUNTS",
            f"{non_positive.sum()} rows have zero or negative amounts. Dropping."
        ))
        df = df[~non_positive].copy()

    # --- Outlier detection (IQR-based) ---
    if len(df) > 10:
        q1 = df["amount"].quantile(0.25)
        q3 = df["amount"].quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 10 * iqr  # Very generous bound
        outlier_mask = df["amount"] > upper_bound
        if outlier_mask.sum() > 0:
            warnings.append(ValidationError(
                "OUTLIER_AMOUNTS",
                f"{outlier_mask.sum()} extreme outlier amounts detected (>{upper_bound:.2f}). Flagged but kept."
            ))
            df.loc[outlier_mask, "_outlier"] = True

    # --- Timestamp validation ---
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", format="mixed")
    bad_ts = df["timestamp"].isna()
    if bad_ts.sum() > 0:
        warnings.append(ValidationError(
            "INVALID_TIMESTAMPS",
            f"{bad_ts.sum()} rows have invalid timestamps. Dropping."
        ))
        df = df[~bad_ts].copy()

    # --- Self-transfers ---
    df["sender_id"] = df["sender_id"].astype(str).str.strip()
    df["receiver_id"] = df["receiver_id"].astype(str).str.strip()
    self_transfer = df["sender_id"] == df["receiver_id"]
    if self_transfer.sum() > 0:
        warnings.append(ValidationError(
            "SELF_TRANSFERS",
            f"{self_transfer.sum()} self-transfer(s) detected. Dropping."
        ))
        df = df[~self_transfer].copy()

    if df.empty:
        errors.append(ValidationError(
            "ALL_ROWS_INVALID",
            "All rows were removed after validation."
        ))
        return pd.DataFrame(), [e.to_dict() for e in errors + warnings]

    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Fill _outlier column if not present
    if "_outlier" not in df.columns:
        df["_outlier"] = False
    df["_outlier"] = df["_outlier"].fillna(False)

    all_issues = errors + warnings
    return df, [e.to_dict() for e in all_issues]
