"""Exploratory summaries for observed columns."""

from __future__ import annotations

import pandas as pd


def numeric_summary(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.select_dtypes(include="number")
    return numeric.describe().T if not numeric.empty else pd.DataFrame()


def correlation_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.select_dtypes(include="number")
    return numeric.corr(numeric_only=True) if numeric.shape[1] > 1 else pd.DataFrame()


def categorical_counts(frame: pd.DataFrame, column: str, limit: int = 20) -> pd.Series:
    return frame[column].astype("string").value_counts(dropna=False).head(limit)