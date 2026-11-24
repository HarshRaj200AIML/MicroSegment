"""Schema inspection and loading for supported survey and tabular files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any
import os
import csv
from io import BytesIO

import pandas as pd

from .utils import DatasetLoadError, DatasetValidationError, get_logger


LOGGER = get_logger("microsegment.data_loader")


@dataclass(frozen=True)
class DatasetLimits:
    max_file_bytes: int = 100 * 1024 * 1024
    max_rows: int = 500_000
    max_columns: int = 500
    high_cardinality_threshold: int = 100
    high_cardinality_ratio: float = 0.95


def dataset_limits_from_environment() -> DatasetLimits:
    def limit(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, str(default)))
        except ValueError:
            LOGGER.warning("Ignoring invalid environment limit %s", name)
            return default

    threshold_ratio = float(os.getenv("MICROSEGMENT_HIGH_CARDINALITY_RATIO", "0.95"))
    return DatasetLimits(
        max_file_bytes=limit("MICROSEGMENT_MAX_FILE_BYTES", 100 * 1024 * 1024),
        max_rows=limit("MICROSEGMENT_MAX_ROWS", 500_000),
        max_columns=limit("MICROSEGMENT_MAX_COLUMNS", 500),
        high_cardinality_threshold=limit("MICROSEGMENT_HIGH_CARDINALITY_THRESHOLD", 100),
        high_cardinality_ratio=threshold_ratio,
    )


@dataclass
class DatasetInfo:
    frame: pd.DataFrame
    variable_labels: dict[str, str]
    value_labels: dict[str, dict[Any, str]]
    source: str
    warnings: list[str] = None

    def __post_init__(self) -> None:
        if self.warnings is None:
            self.warnings = []


def load_dataset(source: str | Path | Any, limits: DatasetLimits | None = None) -> DatasetInfo:
    """Load CSV, Excel, or Stata data; Streamlit UploadedFile is supported."""
    name = getattr(source, "name", str(source))
    suffix = Path(name).suffix.lower()
    active_limits = limits or dataset_limits_from_environment()
    if suffix not in {".dta", ".csv", ".xlsx", ".xls"}:
        raise DatasetValidationError("Supported formats are .dta, .csv, .xlsx, and .xls.")
    _validate_source_size(source, name, active_limits)
    if suffix == ".csv":
        try:
            _validate_csv_header(source)
            frame = pd.read_csv(source, low_memory=False)
        except DatasetValidationError:
            raise
        except Exception as exc:
            LOGGER.exception("CSV load failed for source %s", name)
            raise DatasetLoadError(f"Could not read the CSV file: {exc}") from exc
        variable_labels: dict[str, str] = {}
        value_labels: dict[str, dict[Any, str]] = {}
    elif suffix in {".xlsx", ".xls"}:
        try:
            frame = pd.read_excel(source, engine="openpyxl" if suffix == ".xlsx" else "xlrd")
        except Exception as excel_error:
            if suffix != ".xls":
                LOGGER.exception("Excel load failed for source %s", name)
                raise DatasetLoadError(f"Could not read the Excel workbook: {excel_error}") from excel_error
            try:
                frame = _read_html_spreadsheet(source)
            except Exception as html_error:
                LOGGER.exception("Both Excel and HTML loaders failed for source %s", name)
                raise DatasetLoadError(
                    f"Could not read .xls as an Excel workbook or HTML table. "
                    f"Excel error: {excel_error}; HTML fallback error: {html_error}"
                ) from html_error
        variable_labels = {}
        value_labels = {}
    elif suffix == ".dta":
        frame, variable_labels, value_labels = _read_stata(source)
    if frame.empty or len(frame.columns) == 0:
        raise DatasetValidationError("The selected file contains no tabular records or columns.")
    frame, warnings = _validate_frame(frame, active_limits)
    return DatasetInfo(frame, variable_labels, value_labels, name, warnings)


def _validate_source_size(source: str | Path | Any, name: str, limits: DatasetLimits) -> None:
    size: int | None = None
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.exists():
            size = path.stat().st_size
    elif hasattr(source, "getbuffer"):
        try:
            size = len(source.getbuffer())
        except (TypeError, ValueError, AttributeError):
            size = None
    elif hasattr(source, "getvalue"):
        try:
            size = len(source.getvalue())
        except (TypeError, ValueError, AttributeError):
            size = None
    if size is not None and size > limits.max_file_bytes:
        raise DatasetValidationError(
            f"File '{name}' is too large ({size:,} bytes). "
            f"The configured limit is {limits.max_file_bytes:,} bytes."
        )


def _validate_csv_header(source: str | Path | Any) -> None:
    if isinstance(source, (str, Path)):
        with open(source, "r", encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle), [])
    elif hasattr(source, "seek"):
        source.seek(0)
        content = source.read()
        if isinstance(content, str):
            text = content
        else:
            text = content.decode("utf-8-sig")
        header = next(csv.reader(text.splitlines()), [])
        source.seek(0)
    else:
        return
    names = [str(name).strip() for name in header]
    if any(not name for name in names):
        raise DatasetValidationError("The dataset contains one or more blank column names.")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise DatasetValidationError(
            "The dataset contains duplicate column names: " + ", ".join(duplicates)
        )


def _validate_frame(frame: pd.DataFrame, limits: DatasetLimits) -> tuple[pd.DataFrame, list[str]]:
    if len(frame) > limits.max_rows:
        raise DatasetValidationError(
            f"The dataset has {len(frame):,} rows. The configured row limit is {limits.max_rows:,}."
        )
    if len(frame.columns) > limits.max_columns:
        raise DatasetValidationError(
            f"The dataset has {len(frame.columns):,} columns. "
            f"The configured column limit is {limits.max_columns:,}."
        )

    raw_names = [str(column) for column in frame.columns]
    blank_names = [column for column in raw_names if not column.strip()]
    normalized_names = [column.strip() for column in raw_names]
    duplicate_names = sorted({
        name for name in normalized_names
        if normalized_names.count(name) > 1
    })
    if blank_names:
        raise DatasetValidationError("The dataset contains one or more blank column names.")
    if duplicate_names:
        raise DatasetValidationError(
            "The dataset contains duplicate column names: " + ", ".join(duplicate_names)
        )
    frame = frame.copy()
    frame.columns = normalized_names

    warnings: list[str] = []
    for column in frame.columns:
        series = frame[column]
        is_categorical_text = (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
        )
        if not is_categorical_text:
            continue
        non_null = series.dropna()
        if non_null.empty:
            continue
        unique_count = int(non_null.nunique())
        ratio = unique_count / len(non_null)
        if unique_count >= limits.high_cardinality_threshold and ratio >= limits.high_cardinality_ratio:
            warnings.append(
                f"High-cardinality categorical feature '{column}' has "
                f"{unique_count:,} unique values ({ratio:.1%}). "
                "It may create a very large one-hot encoded matrix."
            )
    return frame, warnings


def _read_html_spreadsheet(source: str | Path | Any) -> pd.DataFrame:
    """Read HTML tables saved with an .xls extension (a common Excel export)."""
    from bs4 import BeautifulSoup

    if isinstance(source, (str, Path)):
        content = Path(source).read_bytes()
    elif hasattr(source, "getvalue"):
        content = source.getvalue()
    else:
        if hasattr(source, "seek"):
            source.seek(0)
        content = source.read()
    soup = BeautifulSoup(content, "html.parser")
    rows = [list(row.stripped_strings) for row in soup.find_all("tr")]
    rows = [row for row in rows if row]
    if not rows:
        raise ValueError("The .xls file is not a readable Excel workbook or HTML table.")
    header_index = max(range(len(rows)), key=lambda index: len(rows[index]))
    headers = rows[header_index]
    records = [row for row in rows[header_index + 1:] if len(row) == len(headers)]
    if not records:
        raise ValueError("No rectangular data table was found in the HTML spreadsheet.")
    frame = pd.DataFrame(records, columns=headers)
    for column in frame.columns:
        values = frame[column].astype("string").str.replace(",", "", regex=False)
        converted = pd.to_numeric(values, errors="coerce")
        if converted.notna().sum() == frame[column].notna().sum():
            frame[column] = converted
    return frame


def _read_stata(source: str | Path | Any) -> tuple[pd.DataFrame, dict[str, str], dict[str, dict[Any, str]]]:
    pyreadstat_error: Exception | None = None
    try:
        import pyreadstat
        frame, metadata = pyreadstat.read_dta(source, apply_value_formats=False)
        labels = {key: value for key, value in (metadata.column_names_to_labels or {}).items() if value}
        value_labels = metadata.variable_value_labels or {}
        return frame, labels, value_labels
    except ImportError as exc:
        pyreadstat_error = exc
    except Exception as exc:
        pyreadstat_error = exc
        LOGGER.exception("Pyreadstat could not parse the Stata source")

    try:
        if hasattr(source, "seek"):
            source.seek(0)
        frame = pd.read_stata(source, convert_categoricals=False)
        return frame, {}, {}
    except Exception as pandas_error:
        LOGGER.exception("Pandas Stata fallback also failed")
        raise DatasetLoadError(
            "Could not load this Stata file with either supported loader. "
            "The file may be corrupted or use an unsupported Stata format."
        ) from pandas_error


def infer_column_roles(frame: pd.DataFrame) -> dict[str, list[str]]:
    """Classify columns using observed types, cardinality and identifier-like names."""
    roles: dict[str, list[str]] = {key: [] for key in ("numeric", "categorical", "binary", "identifier", "datetime", "constant")}
    for column in frame.columns:
        series = frame[column]
        name = str(column).lower()
        values = series.dropna()
        unique_count = int(values.nunique())
        is_id_name = bool(re.search(r"(^|[^a-z])(id|code|key|uuid|serial|s\.?\s*no|no)([^a-z]|$)", name))
        is_id_shape = (
            not pd.api.types.is_numeric_dtype(series)
            and len(values) >= 20
            and unique_count * 100 >= len(values) * 95
        )
        if is_id_name or is_id_shape:
            roles["identifier"].append(column)
        elif pd.api.types.is_datetime64_any_dtype(series):
            roles["datetime"].append(column)
        elif unique_count <= 1:
            roles["constant"].append(column)
        elif pd.api.types.is_numeric_dtype(series):
            if unique_count <= 2:
                roles["binary"].append(column)
            else:
                roles["numeric"].append(column)
        elif unique_count <= 2:
            roles["binary"].append(column)
        else:
            roles["categorical"].append(column)
    return roles


def dataset_summary(info: DatasetInfo) -> dict[str, Any]:
    frame = info.frame
    return {
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "missing_cells": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "roles": infer_column_roles(frame),
        "dtypes": {str(key): str(value) for key, value in frame.dtypes.items()},
        "missing_by_column": frame.isna().sum().sort_values(ascending=False),
    }


def discover_datasets(project_root: str | Path) -> list[Path]:
    root = Path(project_root)
    search_roots = [root / "data" / "raw", root.parent]
    found: list[Path] = []
    seen_names: set[str] = set()
    for search_root in search_roots:
        if search_root.exists():
            for path in sorted(search_root.glob("*")):
                if path.is_file() and path.suffix.lower() in {".dta", ".csv", ".xlsx", ".xls"}:
                    key = path.name.casefold()
                    if not path.name.startswith("~$") and key not in seen_names:
                        found.append(path)
                        seen_names.add(key)
    return list(dict.fromkeys(found))