"""Pipeline orchestrator: load → validate → derive → report → output.

Entry point: python -m src.data_pipeline
"""

import json
from pathlib import Path

from src.data_loader import load_dataset
from src.data_validator import (
    validate_columns,
    validate_uniqueness,
    validate_types,
    validate_fecha_consistency,
)
from src.data_quality import derive_categoria, quality_report


def run_report(input_path, output_dir="results"):
    """Run the full data quality pipeline end-to-end.

    1. Load dataset from input_path (XLSX or CSV).
    2. Validate columns, uniqueness, types, and fecha consistency.
    3. Derive canonical categoria.
    4. Build quality report.
    5. Write JSON to output_dir/quality_report.json.
    6. Print summary to console.
    7. Return the report dict.

    The source file is never modified.

    Args:
        input_path: Path to XLSX or CSV file.
        output_dir: Directory for quality_report.json (default: "results").

    Returns:
        dict: The complete quality report.
    """
    # 1. Load
    df, metadata = load_dataset(input_path)

    # 2. Validate
    findings = {
        "missing_columns": validate_columns(df)["missing_columns"],
        "id_factura": validate_uniqueness(df),
        "types": validate_types(df),
        "fecha_consistency": validate_fecha_consistency(df),
    }

    # 3. Derive canonical categoria
    df = derive_categoria(df)

    # 4. Build report
    report = quality_report(df, findings, metadata)

    # 5. Write JSON
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "quality_report.json"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    # 6. Console summary
    _print_summary(report, json_path)

    return report


def _print_summary(report, json_path):
    """Print a human-readable summary to stdout."""
    status = report["status"]
    ds = report["dataset"]
    schema = report["schema"]
    validations = report["validations"]
    cat = report["categoria_source"]

    print("=" * 60)
    print("  DATA QUALITY REPORT")
    print("=" * 60)
    print(f"  Status:           {status}")
    print(f"  Source:           {report['source']['input_path']}")
    print(f"  Format:           {report['source']['format']}")
    print(f"  Rows:             {ds['row_count']}")
    print(f"  Columns:          {ds['column_count']}")
    print(f"  Missing columns:  {len(schema['missing_columns'])}")
    if schema["missing_columns"]:
        print(f"    → {', '.join(schema['missing_columns'])}")
    print(f"  Duplicate IDs:    {validations['id_factura']['duplicate_count']}")
    print(f"  Invalid monto:    {validations['types']['invalid_monto_count']}")
    print(f"  Invalid fecha:    {validations['types']['invalid_fecha_count']}")
    print(f"  Invalid fecha_mes:{validations['types']['invalid_fecha_mes_count']}")
    print(f"  Fecha mismatches: {validations['fecha_consistency']['mismatch_count']}")
    print(f"  Categoria source: derived={cat['derived']}, verified={cat['verified']}, mismatched={cat['mismatched']}")
    print(f"  Report written:   {json_path}")
    print("=" * 60)


# Entry point for python -m src.data_pipeline
if __name__ == "__main__":
    import sys

    input_path = "data/raw/dataset_facturas_sistemas_inteligentes.xlsx"
    if len(sys.argv) > 1:
        input_path = sys.argv[1]

    run_report(input_path)
