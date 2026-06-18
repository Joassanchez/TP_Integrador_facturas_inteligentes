"""Dataset loader: XLSX (openpyxl) and CSV (UTF-8 / latin-1 fallback)."""

from pathlib import Path

import pandas as pd


def load_dataset(input_path):
    """Load a dataset from XLSX or CSV into a DataFrame with metadata.

    Args:
        input_path (str | Path): Path to the source file.

    Returns:
        tuple[pd.DataFrame, dict]: (dataframe, metadata_dict)
        metadata = {"format": "xlsx"|"csv", "input_path": str, "sheet_name": str|None}

    Raises:
        FileNotFoundError: If input_path does not exist.
        ValueError: If the file extension is not .xlsx or .csv.
    """
    path = Path(input_path).resolve()

    if not path.exists():
        # XLSX missing → try sibling CSV fallback with same basename
        csv_path = path.with_suffix(".csv")
        if csv_path.exists():
            return _load_csv(csv_path)
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".xlsx":
        return _load_xlsx(path)
    elif suffix == ".csv":
        return _load_csv(path)
    else:
        raise ValueError(f"Unsupported file extension: {suffix}. Expected .xlsx or .csv")


def _load_xlsx(path):
    """Load an XLSX file's first sheet."""
    xl = pd.ExcelFile(path, engine="openpyxl")
    sheet_name = xl.sheet_names[0]
    df = xl.parse(sheet_name)
    metadata = {
        "format": "xlsx",
        "input_path": str(path),
        "sheet_name": sheet_name,
    }
    return df, metadata


def _load_csv(path):
    """Load a CSV file, trying UTF-8 first then latin-1 on failure."""
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")

    metadata = {
        "format": "csv",
        "input_path": str(path),
        "sheet_name": None,
    }
    return df, metadata
