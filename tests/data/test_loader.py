"""Tests for src/data/loader.py — RED phase: tests written BEFORE implementation."""

import tempfile
from pathlib import Path

import pandas as pd
import openpyxl
import pytest

# Import the module under test — does NOT exist yet (RED)
from src.data.loader import load_dataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """Create a small known DataFrame for use in temp file creation."""
    return pd.DataFrame({
        "id_factura": ["F0001", "F0002"],
        "proveedor": ["Proveedor A", "Proveedor B"],
        "monto": [100, 200],
    })


@pytest.fixture
def xlsx_path(tmp_path, sample_df):
    """Write a temporary XLSX file and return its path."""
    p = tmp_path / "test.xlsx"
    sample_df.to_excel(p, index=False, engine="openpyxl")
    return p


@pytest.fixture
def csv_utf8_path(tmp_path, sample_df):
    """Write a temporary UTF-8 CSV file and return its path."""
    p = tmp_path / "test_utf8.csv"
    sample_df.to_csv(p, index=False, encoding="utf-8")
    return p


@pytest.fixture
def csv_latin1_path(tmp_path):
    """Write a temporary latin-1 CSV with accented characters."""
    p = tmp_path / "test_latin1.csv"
    df = pd.DataFrame({
        "proveedor": ["Jose García", "Ramón Pérez"],
        "monto": [100, 200],
    })
    df.to_csv(p, index=False, encoding="latin-1")
    return p


# ---------------------------------------------------------------------------
# XLSX tests
# ---------------------------------------------------------------------------

def test_load_xlsx_returns_dataframe_and_metadata(xlsx_path):
    """GIVEN a valid XLSX file WHEN load_dataset is called THEN returns
    a tuple of (DataFrame, dict) with all rows intact."""
    df, meta = load_dataset(xlsx_path)

    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 2
    assert df.shape[1] == 3
    assert isinstance(meta, dict)


def test_load_xlsx_metadata_format(xlsx_path):
    """GIVEN an XLSX file WHEN loaded THEN metadata records format='xlsx'
    and includes the resolved input_path."""
    _, meta = load_dataset(xlsx_path)

    assert meta["format"] == "xlsx"
    assert meta["input_path"] == str(Path(xlsx_path).resolve())
    assert meta["sheet_name"] is not None  # first sheet name


def test_load_xlsx_preserves_all_columns(xlsx_path):
    """GIVEN an XLSX with known columns WHEN loaded THEN all columns appear
    in the DataFrame."""
    df, _ = load_dataset(xlsx_path)

    assert list(df.columns) == ["id_factura", "proveedor", "monto"]


# ---------------------------------------------------------------------------
# CSV tests
# ---------------------------------------------------------------------------

def test_load_csv_utf8_returns_dataframe(csv_utf8_path):
    """GIVEN a valid UTF-8 CSV WHEN loaded THEN returns DataFrame with
    correct shape and metadata format='csv'."""
    df, meta = load_dataset(csv_utf8_path)

    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 2
    assert meta["format"] == "csv"
    assert meta["sheet_name"] is None


def test_load_csv_utf8_all_columns(csv_utf8_path):
    """GIVEN a UTF-8 CSV with known columns WHEN loaded THEN all columns
    are preserved in order."""
    df, _ = load_dataset(csv_utf8_path)

    assert list(df.columns) == ["id_factura", "proveedor", "monto"]


def test_load_csv_latin1_fallback(csv_latin1_path):
    """GIVEN a latin-1 encoded CSV WHEN UTF-8 fails THEN load_dataset
    falls back to latin-1 and returns correct data with accented chars."""
    df, meta = load_dataset(csv_latin1_path)

    assert meta["format"] == "csv"
    assert df.shape[0] == 2
    # Verify accented characters survived the latin-1 round-trip
    values = df["proveedor"].tolist()
    assert any("García" in v for v in values)


def test_load_csv_latin1_metadata(csv_latin1_path):
    """GIVEN a latin-1 CSV WHEN loaded successfully THEN metadata reflects
    csv format with no sheet_name."""
    _, meta = load_dataset(csv_latin1_path)

    assert meta["format"] == "csv"
    assert meta["sheet_name"] is None


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_load_missing_file_raises():
    """GIVEN a non-existent path WHEN load_dataset is called THEN raises
    FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_dataset("nonexistent/file.xlsx")


def test_load_xlsx_missing_csv_fallback(tmp_path, sample_df):
    """GIVEN dataset.xlsx absent but dataset.csv present as sibling
    WHEN load_dataset('dataset.xlsx') is called
    THEN the CSV fallback is loaded with format='csv' and input_path
    pointing to the CSV file actually loaded."""
    csv_path = tmp_path / "dataset.csv"
    sample_df.to_csv(csv_path, index=False, encoding="utf-8")

    xlsx_path = tmp_path / "dataset.xlsx"
    # xlsx_path does NOT exist — only the CSV sibling exists

    df, meta = load_dataset(str(xlsx_path))

    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 2
    assert meta["format"] == "csv"
    assert meta["input_path"] == str(csv_path.resolve())
    assert meta["sheet_name"] is None


def test_load_unsupported_extension_raises(tmp_path):
    """GIVEN a file with .txt extension WHEN load_dataset is called
    THEN raises ValueError."""
    txt = tmp_path / "data.txt"
    txt.write_text("not a dataset")
    with pytest.raises(ValueError):
        load_dataset(txt)
