"""Tests for src/data_quality.py — RED phase: tests written BEFORE implementation."""

import pandas as pd
import pytest

from src.data_quality import derive_categoria, quality_report


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def df_with_categoria():
    """DataFrame where source categoria exists and matches canonical."""
    return pd.DataFrame({
        "id_factura": ["F001", "F002", "F003"],
        "rubro": ["Alimentos", "Logistica", "Servicios"],
        "subrubro": ["Bebidas", "Combustible", "Internet"],
        "categoria": [
            "Alimentos__Bebidas",
            "Logistica__Combustible",
            "Servicios__Internet",
        ],
        "monto": [100, 200, 300],
        "fecha": ["2025-01-01", "2025-06-15", "2025-12-01"],
        "fecha_mes": [1, 6, 12],
    })


@pytest.fixture
def df_without_categoria():
    """DataFrame without a categoria column."""
    return pd.DataFrame({
        "id_factura": ["F001", "F002"],
        "rubro": ["Alimentos", "Logistica"],
        "subrubro": ["Bebidas", "Combustible"],
        "monto": [100, 200],
        "fecha": ["2025-01-01", "2025-06-15"],
        "fecha_mes": [1, 6],
    })


@pytest.fixture
def df_mismatched_categoria():
    """DataFrame where source categoria differs from canonical."""
    return pd.DataFrame({
        "id_factura": ["F001", "F002"],
        "rubro": ["Alimentos", "Logistica"],
        "subrubro": ["Bebidas", "Combustible"],
        "categoria": [
            "Alimentos__Bebidas",       # matches canonical
            "Logistica__WrongCategory",  # mismatched
        ],
    })


@pytest.fixture
def minimal_metadata():
    """Minimal metadata dict matching what load_dataset returns."""
    return {
        "format": "xlsx",
        "input_path": "C:/data/test.xlsx",
        "sheet_name": "Sheet1",
    }


@pytest.fixture
def all_pass_findings():
    """Validation findings where everything passes."""
    return {
        "missing_columns": [],
        "id_factura": {"duplicate_count": 0, "example_ids": []},
        "types": {"invalid_monto_count": 0, "invalid_fecha_count": 0, "invalid_fecha_mes_count": 0},
        "fecha_consistency": {"mismatch_count": 0},
    }


# ---------------------------------------------------------------------------
# derive_categoria — source absent
# ---------------------------------------------------------------------------

def test_derive_categoria_source_absent(df_without_categoria):
    """GIVEN no categoria column WHEN derive_categoria runs THEN canonical
    categoria is derived and categoria_source is 'derived' for every row."""
    result = derive_categoria(df_without_categoria)

    assert "categoria" in result.columns
    assert result["categoria"].tolist() == ["Alimentos__Bebidas", "Logistica__Combustible"]
    assert (result["categoria_source"] == "derived").all()


def test_derive_categoria_source_absent_no_source_categoria_column(df_without_categoria):
    """GIVEN no categoria column WHEN derived THEN source_categoria column
    SHOULD NOT be added (there is no source to preserve)."""
    result = derive_categoria(df_without_categoria)

    assert "source_categoria" not in result.columns


def test_derive_categoria_source_absent_preserves_original_columns(df_without_categoria):
    """GIVEN no categoria WHEN derive_categoria THEN all original columns
    are preserved plus categoria and categoria_source are added."""
    result = derive_categoria(df_without_categoria)
    original_cols = set(df_without_categoria.columns)

    assert original_cols.issubset(set(result.columns))
    assert "categoria" in result.columns
    assert "categoria_source" in result.columns


# ---------------------------------------------------------------------------
# derive_categoria — source present, verified
# ---------------------------------------------------------------------------

def test_derive_categoria_source_verified(df_with_categoria):
    """GIVEN source categoria matches canonical WHEN derived THEN
    source_categoria is preserved and categoria_source is 'verified'."""
    result = derive_categoria(df_with_categoria)

    assert "source_categoria" in result.columns
    assert result["source_categoria"].tolist() == [
        "Alimentos__Bebidas",
        "Logistica__Combustible",
        "Servicios__Internet",
    ]
    assert (result["categoria_source"] == "verified").all()


def test_derive_categoria_canonical_overwrites_source(df_with_categoria):
    """GIVEN source categoria column exists WHEN derived THEN canonical
    categoria column contains the derived value, not the original."""
    # Change original to avoid coincidental match
    df = df_with_categoria.copy()
    df["categoria"] = ["X", "Y", "Z"]  # arbitrary values

    result = derive_categoria(df)

    # Canonical must be derived
    assert result["categoria"].tolist() == [
        "Alimentos__Bebidas",
        "Logistica__Combustible",
        "Servicios__Internet",
    ]
    # Source preserved
    assert result["source_categoria"].tolist() == ["X", "Y", "Z"]


# ---------------------------------------------------------------------------
# derive_categoria — source present, mismatched
# ---------------------------------------------------------------------------

def test_derive_categoria_source_mismatched(df_mismatched_categoria):
    """GIVEN source categoria differs from canonical for some rows THEN
    those rows are classified as 'mismatched' and others as 'verified'."""
    result = derive_categoria(df_mismatched_categoria)

    sources = result["categoria_source"].tolist()
    assert sources == ["verified", "mismatched"]


def test_derive_categoria_mismatched_preserves_source(df_mismatched_categoria):
    """GIVEN mismatched rows THEN source_categoria preserves original values."""
    result = derive_categoria(df_mismatched_categoria)

    assert result["source_categoria"].tolist() == [
        "Alimentos__Bebidas",
        "Logistica__WrongCategory",
    ]


def test_derive_categoria_empty_dataframe():
    """GIVEN an empty DataFrame THEN derive_categoria returns empty DataFrame
    with categoria and categoria_source columns."""
    df = pd.DataFrame(columns=["rubro", "subrubro"])
    result = derive_categoria(df)

    assert len(result) == 0
    assert "categoria" in result.columns
    assert "categoria_source" in result.columns
    assert (result["categoria_source"] == "derived").all()


# ---------------------------------------------------------------------------
# quality_report — status logic
# ---------------------------------------------------------------------------

def test_quality_report_pass(all_pass_findings, df_with_categoria, minimal_metadata):
    """GIVEN all validations pass and schema ok THEN status is PASS."""
    # Derive categoria first so df has categoria_source
    df = derive_categoria(df_with_categoria)
    report = quality_report(df, all_pass_findings, minimal_metadata)

    assert report["status"] == "PASS"


def test_quality_report_fail_missing_columns(minimal_metadata, df_without_categoria):
    """GIVEN missing required columns THEN status is FAIL."""
    findings = {
        "missing_columns": ["monto", "fecha"],
        "id_factura": {"duplicate_count": 0, "example_ids": []},
        "types": {"invalid_monto_count": 0, "invalid_fecha_count": 0, "invalid_fecha_mes_count": 0},
        "fecha_consistency": {"mismatch_count": 0},
    }
    df = derive_categoria(df_without_categoria)
    report = quality_report(df, findings, minimal_metadata)

    assert report["status"] == "FAIL"


def test_quality_report_warn_duplicates(minimal_metadata, df_without_categoria):
    """GIVEN id_factura duplicates THEN status is WARN."""
    findings = {
        "missing_columns": [],
        "id_factura": {"duplicate_count": 3, "example_ids": ["F001"]},
        "types": {"invalid_monto_count": 0, "invalid_fecha_count": 0, "invalid_fecha_mes_count": 0},
        "fecha_consistency": {"mismatch_count": 0},
    }
    df = derive_categoria(df_without_categoria)
    report = quality_report(df, findings, minimal_metadata)

    assert report["status"] == "WARN"


def test_quality_report_warn_fecha_mismatch(minimal_metadata, df_without_categoria):
    """GIVEN fecha consistency mismatches THEN status is WARN."""
    findings = {
        "missing_columns": [],
        "id_factura": {"duplicate_count": 0, "example_ids": []},
        "types": {"invalid_monto_count": 0, "invalid_fecha_count": 0, "invalid_fecha_mes_count": 0},
        "fecha_consistency": {"mismatch_count": 5},
    }
    df = derive_categoria(df_without_categoria)
    report = quality_report(df, findings, minimal_metadata)

    assert report["status"] == "WARN"


def test_quality_report_warn_invalid_types(minimal_metadata, df_without_categoria):
    """GIVEN invalid types THEN status is WARN."""
    findings = {
        "missing_columns": [],
        "id_factura": {"duplicate_count": 0, "example_ids": []},
        "types": {"invalid_monto_count": 2, "invalid_fecha_count": 0, "invalid_fecha_mes_count": 0},
        "fecha_consistency": {"mismatch_count": 0},
    }
    df = derive_categoria(df_without_categoria)
    report = quality_report(df, findings, minimal_metadata)

    assert report["status"] == "WARN"


def test_quality_report_warn_mismatched_categoria(minimal_metadata, df_mismatched_categoria):
    """GIVEN mismatched categoria rows THEN status is WARN."""
    findings = {
        "missing_columns": [],
        "id_factura": {"duplicate_count": 0, "example_ids": []},
        "types": {"invalid_monto_count": 0, "invalid_fecha_count": 0, "invalid_fecha_mes_count": 0},
        "fecha_consistency": {"mismatch_count": 0},
    }
    df = derive_categoria(df_mismatched_categoria)
    report = quality_report(df, findings, minimal_metadata)

    assert report["status"] == "WARN"


# ---------------------------------------------------------------------------
# quality_report — structure and fields
# ---------------------------------------------------------------------------

def test_quality_report_has_source_block(all_pass_findings, df_with_categoria, minimal_metadata):
    """The report MUST include a 'source' block from metadata."""
    df = derive_categoria(df_with_categoria)
    report = quality_report(df, all_pass_findings, minimal_metadata)

    assert report["source"]["input_path"] == minimal_metadata["input_path"]
    assert report["source"]["format"] == minimal_metadata["format"]
    assert report["source"]["sheet_name"] == minimal_metadata["sheet_name"]


def test_quality_report_has_dataset_block(all_pass_findings, df_with_categoria, minimal_metadata):
    """The report MUST include dataset row_count, column_count, and columns list."""
    df = derive_categoria(df_with_categoria)
    report = quality_report(df, all_pass_findings, minimal_metadata)

    assert report["dataset"]["row_count"] == 3
    assert report["dataset"]["column_count"] == len(df.columns)
    assert isinstance(report["dataset"]["columns"], list)


def test_quality_report_has_schema_block(all_pass_findings, df_with_categoria, minimal_metadata):
    """The report MUST include schema validation results."""
    df = derive_categoria(df_with_categoria)
    report = quality_report(df, all_pass_findings, minimal_metadata)

    assert "required_columns" in report["schema"]
    assert "missing_columns" in report["schema"]
    assert "optional_present" in report["schema"]
    assert "categoria" in report["schema"]["optional_present"]


def test_quality_report_has_validations_block(all_pass_findings, df_with_categoria, minimal_metadata):
    """The report MUST include the validations block with id_factura, types,
    and fecha_consistency sub-blocks."""
    df = derive_categoria(df_with_categoria)
    report = quality_report(df, all_pass_findings, minimal_metadata)

    assert "id_factura" in report["validations"]
    assert "types" in report["validations"]
    assert "fecha_consistency" in report["validations"]


def test_quality_report_has_nulls_per_column(all_pass_findings, df_with_categoria, minimal_metadata):
    """The report MUST include null counts per column."""
    df = derive_categoria(df_with_categoria)
    report = quality_report(df, all_pass_findings, minimal_metadata)

    assert isinstance(report["nulls_per_column"], dict)
    for col in df.columns:
        assert col in report["nulls_per_column"]


def test_quality_report_has_distributions(all_pass_findings, df_with_categoria, minimal_metadata):
    """The report MUST include distributions for rubro, subrubro, and categoria."""
    df = derive_categoria(df_with_categoria)
    report = quality_report(df, all_pass_findings, minimal_metadata)

    assert "rubro" in report["distributions"]
    assert "subrubro" in report["distributions"]
    assert "categoria" in report["distributions"]


def test_quality_report_has_categoria_source_breakdown(all_pass_findings, df_with_categoria, minimal_metadata):
    """The report MUST include a categoria_source breakdown with derived,
    verified, and mismatched counts."""
    df = derive_categoria(df_with_categoria)
    report = quality_report(df, all_pass_findings, minimal_metadata)

    assert "derived" in report["categoria_source"]
    assert "verified" in report["categoria_source"]
    assert "mismatched" in report["categoria_source"]


def test_quality_report_has_generated_at(all_pass_findings, df_with_categoria, minimal_metadata):
    """The report MUST include a generated_at timestamp string."""
    df = derive_categoria(df_with_categoria)
    report = quality_report(df, all_pass_findings, minimal_metadata)

    assert isinstance(report["generated_at"], str)
    assert len(report["generated_at"]) > 0


# ---------------------------------------------------------------------------
# quality_report — nulls and distributions accuracy
# ---------------------------------------------------------------------------

def test_quality_report_nulls_accurate():
    """Null counts MUST reflect actual NaN values in the DataFrame."""
    df = pd.DataFrame({
        "id_factura": ["F001", "F002", "F003"],
        "rubro": ["A", None, "C"],
        "subrubro": ["X", "Y", None],
        "monto": [100, None, 300],
    })
    df = derive_categoria(df)
    findings = {
        "missing_columns": [],
        "id_factura": {"duplicate_count": 0, "example_ids": []},
        "types": {"invalid_monto_count": 0, "invalid_fecha_count": 0, "invalid_fecha_mes_count": 0},
        "fecha_consistency": {"mismatch_count": 0},
    }
    report = quality_report(df, findings, {"format": "csv", "input_path": "/t.csv", "sheet_name": None})

    assert report["nulls_per_column"]["rubro"] == 1
    assert report["nulls_per_column"]["subrubro"] == 1
    assert report["nulls_per_column"]["monto"] == 1


def test_quality_report_distributions_accurate():
    """Distribution counts MUST match actual value frequencies."""
    df = pd.DataFrame({
        "id_factura": ["F001", "F002", "F003"],
        "rubro": ["A", "A", "B"],
        "subrubro": ["X", "Y", "X"],
    })
    df = derive_categoria(df)
    findings = {
        "missing_columns": [],
        "id_factura": {"duplicate_count": 0, "example_ids": []},
        "types": {"invalid_monto_count": 0, "invalid_fecha_count": 0, "invalid_fecha_mes_count": 0},
        "fecha_consistency": {"mismatch_count": 0},
    }
    report = quality_report(df, findings, {"format": "csv", "input_path": "/t.csv", "sheet_name": None})

    assert report["distributions"]["rubro"]["A"] == 2
    assert report["distributions"]["rubro"]["B"] == 1
    assert report["distributions"]["subrubro"]["X"] == 2
    assert report["distributions"]["subrubro"]["Y"] == 1


def test_quality_report_categoria_source_breakdown_accurate(df_mismatched_categoria):
    """The categoria_source breakdown MUST have accurate counts."""
    df = derive_categoria(df_mismatched_categoria)
    findings = {
        "missing_columns": [],
        "id_factura": {"duplicate_count": 0, "example_ids": []},
        "types": {"invalid_monto_count": 0, "invalid_fecha_count": 0, "invalid_fecha_mes_count": 0},
        "fecha_consistency": {"mismatch_count": 0},
    }
    report = quality_report(df, findings, {"format": "csv", "input_path": "/t.csv", "sheet_name": None})

    assert report["categoria_source"]["verified"] == 1
    assert report["categoria_source"]["mismatched"] == 1
    assert report["categoria_source"]["derived"] == 0


# ---------------------------------------------------------------------------
# quality_report — PASS with categoria absent tracks it
# ---------------------------------------------------------------------------

def test_quality_report_optional_present_tracks_categoria(df_with_categoria, minimal_metadata):
    """GIVEN categoria column is present in source THEN optional_present.categoria is True."""
    df = derive_categoria(df_with_categoria)
    findings = {
        "missing_columns": [],
        "id_factura": {"duplicate_count": 0, "example_ids": []},
        "types": {"invalid_monto_count": 0, "invalid_fecha_count": 0, "invalid_fecha_mes_count": 0},
        "fecha_consistency": {"mismatch_count": 0},
    }
    report = quality_report(df, findings, minimal_metadata)

    assert report["schema"]["optional_present"]["categoria"] is True
