"""Tests for src/data/validator.py — RED phase: tests written BEFORE implementation."""

import pandas as pd
import pytest

from src.data.validator import (
    REQUIRED_COLUMNS,
    validate_columns,
    validate_uniqueness,
    validate_types,
    validate_fecha_consistency,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_df():
    """A DataFrame satisfying all schema and type constraints."""
    return pd.DataFrame({
        "id_factura": ["F001", "F002", "F003"],
        "proveedor": ["A", "B", "C"],
        "descripcion": ["desc1", "desc2", "desc3"],
        "monto": [100.0, 200.0, 300.0],
        "tipo_comprobante": ["A", "B", "A"],
        "fecha": ["2025-01-15", "2025-06-20", "2025-12-01"],
        "fecha_mes": [1, 6, 12],
        "rubro": ["R1", "R2", "R3"],
        "subrubro": ["S1", "S2", "S3"],
    })


@pytest.fixture
def minimal_df():
    """The absolute minimum that passes validation."""
    return pd.DataFrame({
        "id_factura": ["X001"],
        "proveedor": ["P"],
        "descripcion": ["d"],
        "monto": [50.0],
        "tipo_comprobante": ["A"],
        "fecha": ["2025-03-03"],
        "fecha_mes": [3],
        "rubro": ["R"],
        "subrubro": ["S"],
    })


# ---------------------------------------------------------------------------
# REQUIRED_COLUMNS constant
# ---------------------------------------------------------------------------

def test_required_columns_is_tuple_of_strings():
    """REQUIRED_COLUMNS must be an immutable collection of 9 column names."""
    assert isinstance(REQUIRED_COLUMNS, tuple)
    assert len(REQUIRED_COLUMNS) == 9
    assert all(isinstance(c, str) for c in REQUIRED_COLUMNS)


def test_required_columns_names():
    """REQUIRED_COLUMNS must contain the exact 9 spec columns, excluding categoria."""
    expected = {
        "id_factura", "proveedor", "descripcion", "monto",
        "tipo_comprobante", "fecha", "fecha_mes", "rubro", "subrubro",
    }
    assert set(REQUIRED_COLUMNS) == expected


def test_categoria_not_in_required_columns():
    """categoria MUST NOT be in REQUIRED_COLUMNS (it is optional)."""
    assert "categoria" not in REQUIRED_COLUMNS


# ---------------------------------------------------------------------------
# validate_columns
# ---------------------------------------------------------------------------

def test_validate_columns_all_present(valid_df):
    """GIVEN all 9 required columns WHEN validated THEN no missing columns."""
    result = validate_columns(valid_df)
    assert result["missing_columns"] == []


def test_validate_columns_single_missing():
    """GIVEN a DataFrame missing one required column THEN reports it."""
    df = pd.DataFrame({"id_factura": ["F001"], "proveedor": ["A"]})
    result = validate_columns(df)
    assert "descripcion" in result["missing_columns"]
    assert len(result["missing_columns"]) > 0


def test_validate_columns_multiple_missing():
    """GIVEN a DataFrame missing several columns THEN lists all."""
    df = pd.DataFrame({"id_factura": ["F001"]})
    result = validate_columns(df)
    assert len(result["missing_columns"]) > 1
    assert "proveedor" in result["missing_columns"]
    assert "monto" in result["missing_columns"]


def test_validate_columns_optional_categoria_present():
    """GIVEN categoria column exists WHEN validated THEN required check
    still passes (categoria is optional, not required)."""
    df = pd.DataFrame({
        "id_factura": ["F001"],
        "proveedor": ["A"],
        "descripcion": ["d"],
        "monto": [100],
        "tipo_comprobante": ["A"],
        "fecha": ["2025-01-01"],
        "fecha_mes": [1],
        "rubro": ["R"],
        "subrubro": ["S"],
        "categoria": ["R__S"],
    })
    result = validate_columns(df)
    assert result["missing_columns"] == []


def test_validate_columns_empty_dataframe():
    """GIVEN an empty DataFrame with no columns THEN all required are missing."""
    df = pd.DataFrame()
    result = validate_columns(df)
    assert len(result["missing_columns"]) == 9


# ---------------------------------------------------------------------------
# validate_uniqueness
# ---------------------------------------------------------------------------

def test_validate_uniqueness_all_unique(valid_df):
    """GIVEN all id_factura values are unique THEN duplicate_count=0."""
    result = validate_uniqueness(valid_df)
    assert result["duplicate_count"] == 0
    assert result["example_ids"] == []


def test_validate_uniqueness_duplicates_exist():
    """GIVEN duplicate id_factura values THEN reports count and examples."""
    df = pd.DataFrame({
        "id_factura": ["F001", "F001", "F002", "F002", "F002"],
        "monto": [1, 2, 3, 4, 5],
    })
    result = validate_uniqueness(df)
    assert result["duplicate_count"] == 2  # F001 and F002 are both duplicated
    assert "F001" in result["example_ids"]
    assert "F002" in result["example_ids"]


def test_validate_uniqueness_single_row():
    """GIVEN a single-row DataFrame THEN duplicate_count=0."""
    df = pd.DataFrame({"id_factura": ["F001"], "monto": [100]})
    result = validate_uniqueness(df)
    assert result["duplicate_count"] == 0
    assert result["example_ids"] == []


# ---------------------------------------------------------------------------
# validate_types
# ---------------------------------------------------------------------------

def test_validate_types_all_valid(valid_df):
    """GIVEN monto numeric, fecha parseable, fecha_mes 1-12 THEN no errors."""
    result = validate_types(valid_df)
    assert result["invalid_monto_count"] == 0
    assert result["invalid_fecha_count"] == 0
    assert result["invalid_fecha_mes_count"] == 0


def test_validate_types_invalid_monto():
    """GIVEN monto contains non-numeric strings THEN reports them."""
    df = pd.DataFrame({
        "id_factura": ["F001", "F002"],
        "monto": ["N/A", "abc"],
        "fecha": ["2025-01-01", "2025-06-15"],
        "fecha_mes": [1, 6],
    })
    result = validate_types(df)
    assert result["invalid_monto_count"] == 2


def test_validate_types_coercible_monto_passes():
    """GIVEN monto values are string numbers like '100.5' THEN they are
    coercible and count as valid (pd.to_numeric handles strings)."""
    df = pd.DataFrame({
        "id_factura": ["F001", "F002"],
        "monto": ["100.50", "200"],
        "fecha": ["2025-01-01", "2025-06-15"],
        "fecha_mes": [1, 6],
    })
    result = validate_types(df)
    assert result["invalid_monto_count"] == 0


def test_validate_types_invalid_fecha():
    """GIVEN fecha contains unparseable strings THEN reports them."""
    df = pd.DataFrame({
        "id_factura": ["F001", "F002"],
        "monto": [100, 200],
        "fecha": ["not-a-date", "garbage"],
        "fecha_mes": [1, 6],
    })
    result = validate_types(df)
    assert result["invalid_fecha_count"] == 2


def test_validate_types_consistent_fecha_format():
    """GIVEN fecha in a consistent YYYY-MM-DD format THEN all parse successfully."""
    df = pd.DataFrame({
        "id_factura": ["F001", "F002", "F003"],
        "monto": [100, 200, 300],
        "fecha": ["2025-01-01", "2025-06-15", "2025-12-25"],
        "fecha_mes": [1, 6, 12],
    })
    result = validate_types(df)
    assert result["invalid_fecha_count"] == 0


def test_validate_types_invalid_fecha_mes_out_of_range():
    """GIVEN fecha_mes values outside 1-12 THEN reports them."""
    df = pd.DataFrame({
        "id_factura": ["F001", "F002", "F003"],
        "monto": [100, 200, 300],
        "fecha": ["2025-01-01", "2025-06-15", "2025-12-01"],
        "fecha_mes": [0, 6, 13],
    })
    result = validate_types(df)
    assert result["invalid_fecha_mes_count"] == 2  # 0 and 13 are invalid


def test_validate_types_invalid_fecha_mes_float():
    """GIVEN fecha_mes contains non-integer values THEN they are flagged."""
    df = pd.DataFrame({
        "id_factura": ["F001", "F002"],
        "monto": [100, 200],
        "fecha": ["2025-01-01", "2025-06-15"],
        "fecha_mes": [3.5, 6],
    })
    result = validate_types(df)
    assert result["invalid_fecha_mes_count"] == 1  # 3.5 is not integer


def test_validate_types_missing_optional_columns():
    """GIVEN monto/fecha/fecha_mes columns absent THEN validate_types
    handles gracefully (counts as invalid when column missing)."""
    df = pd.DataFrame({"id_factura": ["F001"]})
    result = validate_types(df)
    # Without the columns, we cannot validate — should report 0 or skip
    # Design: validate_types skips missing columns gracefully
    assert result["invalid_monto_count"] == 0
    assert result["invalid_fecha_count"] == 0
    assert result["invalid_fecha_mes_count"] == 0


# ---------------------------------------------------------------------------
# validate_fecha_consistency
# ---------------------------------------------------------------------------

def test_validate_fecha_consistency_all_match(valid_df):
    """GIVEN every row has month(fecha) == fecha_mes THEN mismatch_count=0."""
    result = validate_fecha_consistency(valid_df)
    assert result["mismatch_count"] == 0


def test_validate_fecha_consistency_mismatch_exists():
    """GIVEN fecha month differs from fecha_mes THEN reports mismatch count."""
    df = pd.DataFrame({
        "id_factura": ["F001", "F002", "F003"],
        "monto": [100, 200, 300],
        "fecha": ["2025-01-15", "2025-06-20", "2025-12-01"],
        "fecha_mes": [2, 6, 11],  # row 0 and row 2 mismatch
    })
    result = validate_fecha_consistency(df)
    assert result["mismatch_count"] == 2


def test_validate_fecha_consistency_year_ignored():
    """GIVEN fecha has a different year but same month THEN it matches
    (only the month component is validated, not the year)."""
    df = pd.DataFrame({
        "id_factura": ["F001"],
        "monto": [100],
        "fecha": ["2023-03-15"],  # year 2023, month 3
        "fecha_mes": [3],
    })
    result = validate_fecha_consistency(df)
    assert result["mismatch_count"] == 0


def test_validate_fecha_consistency_unparseable_fecha():
    """GIVEN fecha has unparseable values THEN they are skipped in
    consistency check (reported by validate_types instead)."""
    df = pd.DataFrame({
        "id_factura": ["F001", "F002"],
        "monto": [100, 200],
        "fecha": ["bad-date", "2025-06-20"],
        "fecha_mes": [1, 6],
    })
    result = validate_fecha_consistency(df)
    # Only the second row can be checked (it matches)
    assert result["mismatch_count"] == 0


def test_validate_fecha_consistency_missing_columns():
    """GIVEN fecha or fecha_mes columns absent THEN returns mismatch_count=0."""
    df = pd.DataFrame({"id_factura": ["F001"], "monto": [100]})
    result = validate_fecha_consistency(df)
    assert result["mismatch_count"] == 0
