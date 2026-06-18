"""Tests for src/data/pipeline.py — RED phase: tests written BEFORE implementation."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.data.pipeline import run_report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _file_hash(path):
    """Return MD5 hex digest of a file's contents."""
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_xlsx(tmp_path):
    """Create a valid minimal XLSX dataset fixture."""
    p = tmp_path / "dataset.xlsx"
    df = pd.DataFrame({
        "id_factura": ["F001", "F002", "F003"],
        "proveedor": ["A", "B", "C"],
        "descripcion": ["d1", "d2", "d3"],
        "monto": [100.0, 200.0, 300.0],
        "tipo_comprobante": ["A", "B", "A"],
        "fecha": ["2025-01-15", "2025-06-20", "2025-12-01"],
        "fecha_mes": [1, 6, 12],
        "rubro": ["R1", "R2", "R3"],
        "subrubro": ["S1", "S2", "S3"],
    })
    df.to_excel(p, index=False, engine="openpyxl")
    return p


@pytest.fixture
def xlsx_with_categoria(tmp_path):
    """XLSX with a categoria column (matching canonical)."""
    p = tmp_path / "with_cat.xlsx"
    df = pd.DataFrame({
        "id_factura": ["F001", "F002"],
        "proveedor": ["A", "B"],
        "descripcion": ["d1", "d2"],
        "monto": [100, 200],
        "tipo_comprobante": ["A", "B"],
        "fecha": ["2025-01-15", "2025-06-20"],
        "fecha_mes": [1, 6],
        "rubro": ["Alimentos", "Logistica"],
        "subrubro": ["Bebidas", "Combustible"],
        "categoria": ["Alimentos__Bebidas", "Logistica__Combustible"],
    })
    df.to_excel(p, index=False, engine="openpyxl")
    return p


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def test_run_report_writes_json(valid_xlsx, tmp_path):
    """GIVEN a valid XLSX WHEN run_report completes THEN a JSON report is
    written to the output directory."""
    output_dir = tmp_path / "results"
    report = run_report(valid_xlsx, output_dir)

    json_path = output_dir / "quality_report.json"
    assert json_path.exists()
    assert json_path.is_file()

    # Read back and verify top-level structure
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["status"] == "PASS"
    assert data["source"]["format"] == "xlsx"
    assert "generated_at" in data
    assert "validations" in data
    assert "nulls_per_column" in data
    assert "distributions" in data
    assert "categoria_source" in data


def test_run_report_json_contains_status(valid_xlsx, tmp_path):
    """The JSON report MUST contain a 'status' field."""
    output_dir = tmp_path / "results"
    run_report(valid_xlsx, output_dir)

    json_path = output_dir / "quality_report.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["status"] in ("PASS", "WARN", "FAIL")


def test_run_report_json_fail_on_missing_columns(tmp_path):
    """GIVEN a dataset missing required columns THEN JSON status is FAIL."""
    p = tmp_path / "bad.xlsx"
    df = pd.DataFrame({
        "id_factura": ["F001"],
        "proveedor": ["A"],
    })
    df.to_excel(p, index=False, engine="openpyxl")

    output_dir = tmp_path / "results"
    run_report(p, output_dir)

    json_path = output_dir / "quality_report.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["status"] == "FAIL"
    assert len(data["schema"]["missing_columns"]) > 0


def test_run_report_json_preserves_source_categoria(xlsx_with_categoria, tmp_path):
    """GIVEN an XLSX with categoria column THEN JSON includes
    categoria_source breakdown with verified > 0."""
    output_dir = tmp_path / "results"
    run_report(xlsx_with_categoria, output_dir)

    json_path = output_dir / "quality_report.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["categoria_source"]["verified"] == 2
    assert data["categoria_source"]["derived"] == 0
    assert data["schema"]["optional_present"]["categoria"] is True


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def test_run_report_prints_summary(valid_xlsx, tmp_path, capsys):
    """GIVEN a valid dataset WHEN run_report runs THEN a summary is
    printed to stdout."""
    output_dir = tmp_path / "results"
    run_report(valid_xlsx, output_dir)

    captured = capsys.readouterr()
    assert "PASS" in captured.out
    # Should mention output path or results
    assert len(captured.out) > 0


def test_run_report_prints_status_in_summary(valid_xlsx, tmp_path, capsys):
    """The console output MUST mention the status word."""
    output_dir = tmp_path / "results"
    run_report(valid_xlsx, output_dir)

    captured = capsys.readouterr()
    assert "PASS" in captured.out


def test_run_report_prints_output_path(valid_xlsx, tmp_path, capsys):
    """The console summary MUST mention the output file path."""
    output_dir = tmp_path / "results"
    run_report(valid_xlsx, output_dir)

    captured = capsys.readouterr()
    assert "quality_report.json" in captured.out


# ---------------------------------------------------------------------------
# Non-destructive test
# ---------------------------------------------------------------------------

def test_run_report_source_file_unchanged(valid_xlsx, tmp_path):
    """GIVEN a source XLSX file WHEN run_report completes THEN the file
    on disk is byte-identical to its pre-run state."""
    hash_before = _file_hash(valid_xlsx)

    output_dir = tmp_path / "results"
    run_report(valid_xlsx, output_dir)

    hash_after = _file_hash(valid_xlsx)
    assert hash_before == hash_after, (
        f"Source file hash changed! Before: {hash_before}, After: {hash_after}"
    )


def test_run_report_source_file_unchanged_with_categoria(xlsx_with_categoria, tmp_path):
    """Non-destructive property holds even when categoria column present."""
    hash_before = _file_hash(xlsx_with_categoria)

    output_dir = tmp_path / "results"
    run_report(xlsx_with_categoria, output_dir)

    hash_after = _file_hash(xlsx_with_categoria)
    assert hash_before == hash_after


# ---------------------------------------------------------------------------
# Return value
# ---------------------------------------------------------------------------

def test_run_report_returns_dict(valid_xlsx, tmp_path):
    """GIVEN a valid dataset WHEN run_report runs THEN returns the report dict."""
    output_dir = tmp_path / "results"
    report = run_report(valid_xlsx, output_dir)

    assert isinstance(report, dict)
    assert "status" in report
    assert "generated_at" in report


def test_run_report_return_matches_json(valid_xlsx, tmp_path):
    """The returned dict MUST match the written JSON (except generated_at
    which may differ by sub-second timing)."""
    output_dir = tmp_path / "results"
    report = run_report(valid_xlsx, output_dir)

    json_path = output_dir / "quality_report.json"
    with open(json_path, "r", encoding="utf-8") as f:
        from_file = json.load(f)

    # Compare structural fields (generated_at may differ at microsecond level)
    for key in ["status", "source", "dataset", "schema", "validations",
                "nulls_per_column", "distributions", "categoria_source"]:
        assert report[key] == from_file[key], f"Mismatch on key: {key}"


# ---------------------------------------------------------------------------
# CLI entry-point test (covers src/data_pipeline.py:100-108)
# ---------------------------------------------------------------------------

def test_cli_entry_point_writes_json(valid_xlsx, tmp_path):
    """GIVEN a valid XLSX WHEN run via `python -m src.data.pipeline <path>`
    THEN a JSON report is written to results/ and exit code is 0."""
    output_dir = tmp_path / "results"

    # Run from project root so `src` package is importable.
    # Pass tmp_path as cwd so results/ is created there, not in the real project.
    result = subprocess.run(
        [sys.executable, "-m", "src.data.pipeline", str(valid_xlsx)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent.parent)},
    )

    assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
    assert "PASS" in result.stdout
    json_path = tmp_path / "results" / "quality_report.json"
    assert json_path.exists()
