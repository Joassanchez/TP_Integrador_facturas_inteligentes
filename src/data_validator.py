"""Dataset validators: schema, uniqueness, types, and fecha consistency.

All validators return structured findings dicts — they never raise.
"""

import pandas as pd

# Required columns per the supervised schema contract (9 columns).
# categoria is deliberately excluded — it is optional.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "id_factura",
    "proveedor",
    "descripcion",
    "monto",
    "tipo_comprobante",
    "fecha",
    "fecha_mes",
    "rubro",
    "subrubro",
)


def validate_columns(df: pd.DataFrame) -> dict:
    """Check that all REQUIRED_COLUMNS are present in the DataFrame.

    Returns:
        dict with key "missing_columns" (list[str]).
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    return {"missing_columns": missing}


def validate_uniqueness(df: pd.DataFrame) -> dict:
    """Detect duplicate values in id_factura.

    Returns:
        dict with "duplicate_count" (int) and "example_ids" (list[str]).
    """
    if "id_factura" not in df.columns:
        return {"duplicate_count": 0, "example_ids": []}

    dup_mask = df["id_factura"].duplicated(keep=False)
    duplicate_ids = df.loc[dup_mask, "id_factura"].unique()
    return {
        "duplicate_count": int(len(duplicate_ids)),
        "example_ids": list(duplicate_ids),
    }


def validate_types(df: pd.DataFrame) -> dict:
    """Validate data types: monto numeric/coercible, fecha parseable as date,
    fecha_mes integer 1-12.

    Returns:
        dict with keys: invalid_monto_count, invalid_fecha_count,
        invalid_fecha_mes_count.
    """
    invalid_monto = 0
    invalid_fecha = 0
    invalid_fecha_mes = 0

    if "monto" in df.columns:
        numeric = pd.to_numeric(df["monto"], errors="coerce")
        invalid_monto = int(numeric.isna().sum())

    if "fecha" in df.columns:
        parsed = pd.to_datetime(df["fecha"], errors="coerce", format="mixed")
        invalid_fecha = int(parsed.isna().sum())

    if "fecha_mes" in df.columns:
        # Must be integer 1-12 (coerce to numeric first, then check)
        mes = pd.to_numeric(df["fecha_mes"], errors="coerce")
        invalid_mask = (
            mes.isna()
            | (mes != mes.astype(int))
            | (mes < 1)
            | (mes > 12)
        )
        invalid_fecha_mes = int(invalid_mask.sum())

    return {
        "invalid_monto_count": invalid_monto,
        "invalid_fecha_count": invalid_fecha,
        "invalid_fecha_mes_count": invalid_fecha_mes,
    }


def validate_fecha_consistency(df: pd.DataFrame) -> dict:
    """Validate that month(fecha) == fecha_mes for every row where both
    can be parsed. Only the month component is checked; year is ignored.

    Returns:
        dict with key "mismatch_count" (int).
    """
    if "fecha" not in df.columns or "fecha_mes" not in df.columns:
        return {"mismatch_count": 0}

    parsed = pd.to_datetime(df["fecha"], errors="coerce", format="mixed")
    mes_val = pd.to_numeric(df["fecha_mes"], errors="coerce")

    # Only compare rows where both fecha and fecha_mes are valid
    valid_mask = parsed.notna() & mes_val.notna()
    if not valid_mask.any():
        return {"mismatch_count": 0}

    fecha_month = parsed[valid_mask].dt.month
    fecha_mes_int = mes_val[valid_mask].astype(int)

    mismatches = (fecha_month != fecha_mes_int).sum()
    return {"mismatch_count": int(mismatches)}
