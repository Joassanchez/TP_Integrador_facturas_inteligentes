"""Data quality: canonical categoria derivation and report assembly."""

from datetime import datetime, timezone

import pandas as pd


def derive_categoria(df: pd.DataFrame) -> pd.DataFrame:
    """Derive canonical categoria from rubro__subrubro.

    If a 'categoria' column already exists in the source, it is preserved
    as 'source_categoria' and compared against the canonical derivation.
    The output DataFrame always has canonical 'categoria' and a
    'categoria_source' column ('derived', 'verified', or 'mismatched').

    Returns a NEW DataFrame (the input is not mutated).
    """
    result = df.copy()

    source_has_categoria = "categoria" in result.columns

    if source_has_categoria:
        # Preserve original before overwriting
        result["source_categoria"] = result["categoria"].copy()

    # Derive canonical categoria — gracefully when rubro/subrubro missing
    has_rubro = "rubro" in result.columns
    has_subrubro = "subrubro" in result.columns

    if has_rubro and has_subrubro:
        result["categoria"] = (
            result["rubro"].astype(str) + "__" + result["subrubro"].astype(str)
        )
    else:
        # Design deviation (documented in apply-progress):
        # When rubro/subrubro are missing (schema FAIL), we emit empty
        # string so the quality report can still be generated and the
        # FAIL status surfaces the missing columns to the user.
        # Downstream consumers in changes #2/#3 will only see canonical
        # categoria on datasets that pass schema validation.
        result["categoria"] = ""

    if source_has_categoria:
        # Compare canonical vs source
        match_mask = result["categoria"] == result["source_categoria"]
        result["categoria_source"] = match_mask.map({True: "verified", False: "mismatched"})
    else:
        result["categoria_source"] = "derived"

    return result


def quality_report(df: pd.DataFrame, findings: dict, metadata: dict) -> dict:
    """Build a quality report dict from the validated DataFrame, findings,
    and metadata.

    Args:
        df: DataFrame after derive_categoria has been applied.
        findings: Combined validation results dict with keys:
            missing_columns, id_factura, types, fecha_consistency.
        metadata: From load_dataset (format, input_path, sheet_name).

    Returns:
        A JSON-safe dict matching the report schema.
    """
    # --- Status ---
    status = _compute_status(df, findings)

    # --- Dataset ---
    dataset = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
    }

    # --- Schema ---
    from src.data_validator import REQUIRED_COLUMNS

    optional_present = {
        "categoria": "source_categoria" in df.columns,
    }
    schema = {
        "required_columns": list(REQUIRED_COLUMNS),
        "missing_columns": findings.get("missing_columns", []),
        "optional_present": optional_present,
    }

    # --- Validations ---
    validations = {
        "id_factura": findings.get("id_factura", {"duplicate_count": 0, "example_ids": []}),
        "types": findings.get("types", {"invalid_monto_count": 0, "invalid_fecha_count": 0, "invalid_fecha_mes_count": 0}),
        "fecha_consistency": findings.get("fecha_consistency", {"mismatch_count": 0}),
    }

    # --- Nulls per column ---
    nulls_per_column = {col: int(df[col].isna().sum()) for col in df.columns}

    # --- Distributions ---
    distributions = {
        "rubro": _value_counts(df, "rubro"),
        "subrubro": _value_counts(df, "subrubro"),
        "categoria": _value_counts(df, "categoria"),
    }

    # --- Categoria source breakdown ---
    if "categoria_source" in df.columns:
        cat_counts = df["categoria_source"].value_counts().to_dict()
    else:
        cat_counts = {}
    categoria_source = {
        "derived": int(cat_counts.get("derived", 0)),
        "verified": int(cat_counts.get("verified", 0)),
        "mismatched": int(cat_counts.get("mismatched", 0)),
    }

    # --- Generated at ---
    generated_at = datetime.now(timezone.utc).isoformat()

    return {
        "status": status,
        "source": {
            "input_path": metadata.get("input_path", ""),
            "format": metadata.get("format", ""),
            "sheet_name": metadata.get("sheet_name"),
        },
        "dataset": dataset,
        "schema": schema,
        "validations": validations,
        "nulls_per_column": nulls_per_column,
        "distributions": distributions,
        "categoria_source": categoria_source,
        "generated_at": generated_at,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_status(df: pd.DataFrame, findings: dict) -> str:
    """Determine PASS / WARN / FAIL from findings and categoria_source."""
    missing = findings.get("missing_columns", [])
    if missing:
        return "FAIL"

    dup_count = findings.get("id_factura", {}).get("duplicate_count", 0)
    types_info = findings.get("types", {})
    invalid_monto = types_info.get("invalid_monto_count", 0)
    invalid_fecha = types_info.get("invalid_fecha_count", 0)
    invalid_fecha_mes = types_info.get("invalid_fecha_mes_count", 0)
    mismatch_count = findings.get("fecha_consistency", {}).get("mismatch_count", 0)

    mismatched_cat = 0
    if "categoria_source" in df.columns:
        mismatched_cat = int((df["categoria_source"] == "mismatched").sum())

    if (
        dup_count > 0
        or invalid_monto > 0
        or invalid_fecha > 0
        or invalid_fecha_mes > 0
        or mismatch_count > 0
        or mismatched_cat > 0
    ):
        return "WARN"

    return "PASS"


def _value_counts(df: pd.DataFrame, column: str) -> dict:
    """Safe value_counts returning a dict (empty dict if column missing)."""
    if column not in df.columns:
        return {}
    counts = df[column].value_counts(dropna=False)
    return {str(k): int(v) for k, v in counts.items()}
