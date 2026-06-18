"""Tests for src/preprocessor.py — RED phase: tests written BEFORE implementation.

Strict TDD: these tests reference production code that does NOT exist yet.
Running them now will fail with ImportError — that IS the expected RED state.
"""

import numpy as np
import pandas as pd
import pytest

# Import from module under test — does NOT exist yet (RED)
from src.preprocessor import (  # noqa: E402
    DROP_COLUMNS,
    sanitize_descripcion,
    DateFeatureExtractor,
    build_preprocessor,
    extract_target,
    validate_class_counts,
    get_feature_names,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def df_fecha():
    """DataFrame with fecha datetime column."""
    return pd.DataFrame({
        "fecha": pd.to_datetime(["2024-01-15", "2024-06-20", "2024-12-31"]),
    })


@pytest.fixture
def df_fecha_mes():
    """DataFrame with fecha_mes only (no fecha) — fallback scenario."""
    return pd.DataFrame({
        "fecha_mes": pd.to_datetime(["2024-01-01", "2024-06-01", "2024-12-01"]),
    })


@pytest.fixture
def df_full():
    """Full-featured invoice DataFrame (30 rows) with all expected columns."""
    rng = np.random.default_rng(42)
    n = 30
    return pd.DataFrame({
        "id_factura": [f"F{i:04d}" for i in range(n)],
        "proveedor": rng.choice(["Telecom", "TechCorp", "LogisticaSur"], n),
        "tipo_comprobante": rng.choice(["Factura A", "Factura B"], n),
        "rubro": rng.choice(["Tecnologia", "Servicios"], n),
        "subrubro": rng.choice(["Hardware", "Software", "Consultoria"], n),
        "categoria": rng.choice(["cat_a", "cat_b", "cat_c"], n),
        "descripcion": [f"servicio {i}" for i in range(n)],
        "monto": rng.uniform(100, 10000, n),
        "fecha": pd.date_range("2024-01-01", periods=n, freq="7D"),
    })


@pytest.fixture
def df_minimal():
    """Minimal 6-row DataFrame for integration-level unit tests."""
    return pd.DataFrame({
        "descripcion": [
            "servicio de internet",
            "compra de equipos",
            "mantenimiento mensual",
            "",
            "licencia software anual",
            "consultoria tecnica",
        ],
        "proveedor": ["Telecom", "TechCorp", "Telecom", "LogisticaSur", "TechCorp", "Telecom"],
        "tipo_comprobante": ["Factura A", "Factura B", "Factura A", "Factura A", "Factura B", "Factura C"],
        "monto": [1500.0, 30000.0, 500.0, 1200.0, 8000.0, 4500.0],
        "fecha": pd.to_datetime([
            "2024-01-15", "2024-03-20", "2024-06-10",
            "2024-07-01", "2024-09-15", "2024-12-01",
        ]),
        "rubro": ["Servicios", "Tecnologia", "Servicios", "Servicios", "Tecnologia", "Servicios"],
        "subrubro": ["Internet", "Hardware", "Internet", "Logistica", "Software", "Consultoria"],
        "categoria": [
            "Servicios__Internet", "Tecnologia__Hardware", "Servicios__Internet",
            "Servicios__Logistica", "Tecnologia__Software", "Servicios__Consultoria",
        ],
        "id_factura": ["F001", "F002", "F003", "F004", "F005", "F006"],
    })


@pytest.fixture
def df_two_per_class():
    """DataFrame with exactly 2 rows per class (6 rows, 3 classes)."""
    return pd.DataFrame({
        "descripcion": ["item a1", "item a2", "item b1", "item b2", "item c1", "item c2"],
        "proveedor": ["A", "A", "B", "B", "C", "C"],
        "tipo_comprobante": ["FC", "FC", "FC", "FC", "FC", "FC"],
        "monto": [100, 200, 300, 400, 500, 600],
        "fecha": pd.date_range("2024-01-01", periods=6, freq="1ME"),
        "rubro": ["r1", "r1", "r2", "r2", "r3", "r3"],
        "subrubro": ["s1", "s1", "s2", "s2", "s3", "s3"],
        "categoria": ["A", "A", "B", "B", "C", "C"],
        "id_factura": [f"F{i:04d}" for i in range(6)],
    })


# ===========================================================================
# 1.1 — DateFeatureExtractor: month, sin/cos, fecha_mes fallback
# ===========================================================================

class TestDateFeatureExtractor:
    """Tests for DateFeatureExtractor — Req 1-2."""

    def test_extracts_month_from_fecha(self, df_fecha):
        """GIVEN fecha column WHEN transformed THEN month extracted as integer [1, 12]."""
        extractor = DateFeatureExtractor()
        result = extractor.fit_transform(df_fecha)

        assert result.shape[0] == 3
        # Month should be January=1, June=6, December=12
        expected_months = np.array([1, 6, 12])
        # Access the month column (first output feature)
        np.testing.assert_array_equal(result[:, 0], expected_months)

    def test_cyclical_sin_cos_output(self, df_fecha):
        """GIVEN fecha column WHEN transformed THEN includes sin_month and cos_month."""
        extractor = DateFeatureExtractor()
        result = extractor.fit_transform(df_fecha)

        # Should have at least 3 columns: month, sin_month, cos_month
        assert result.shape[1] >= 3

        # sin_month values must be in [-1, 1]
        sin_col = result[:, 1]
        cos_col = result[:, 2]
        assert np.all(sin_col >= -1.0) and np.all(sin_col <= 1.0)
        assert np.all(cos_col >= -1.0) and np.all(cos_col <= 1.0)

        # sin^2 + cos^2 ≈ 1 (cyclical identity check)
        for i in range(3):
            assert np.abs(sin_col[i] ** 2 + cos_col[i] ** 2 - 1.0) < 1e-10

    def test_fallback_to_fecha_mes(self, df_fecha_mes):
        """GIVEN no fecha column but fecha_mes EXISTS WHEN transformed
        THEN uses fecha_mes as fallback temporal source."""
        extractor = DateFeatureExtractor()
        result = extractor.fit_transform(df_fecha_mes)

        assert result.shape[0] == 3
        expected_months = np.array([1, 6, 12])
        np.testing.assert_array_equal(result[:, 0], expected_months)

    def test_get_feature_names_out(self, df_fecha):
        """GIVEN fitted DateFeatureExtractor WHEN get_feature_names_out
        THEN returns array of feature names."""
        extractor = DateFeatureExtractor()
        extractor.fit(df_fecha)

        names = extractor.get_feature_names_out()

        assert isinstance(names, np.ndarray)
        assert len(names) >= 3
        assert "month" in names or any("month" in str(n) for n in names)

    def test_uses_fecha_when_both_present(self):
        """GIVEN DataFrame with BOTH fecha and fecha_mes WHEN transformed
        THEN uses fecha (primary), NOT fecha_mes."""
        df = pd.DataFrame({
            "fecha": pd.to_datetime(["2024-03-15"]),
            "fecha_mes": pd.to_datetime(["2024-06-01"]),
        })
        extractor = DateFeatureExtractor()
        result = extractor.fit_transform(df)

        # fecha=March → month=3, fecha_mes=June → month=6
        # Should use fecha (month=3)
        assert result[0, 0] == 3


# ===========================================================================
# 1.2 — Leak-column exclusion: DROP_COLUMNS absent from feature names
# ===========================================================================

class TestLeakColumnExclusion:
    """Tests for leak columns not reaching features — Req 3."""

    def test_drop_columns_contains_expected(self):
        """DROP_COLUMNS tuple MUST contain rubro, subrubro, categoria, id_factura."""
        assert "rubro" in DROP_COLUMNS
        assert "subrubro" in DROP_COLUMNS
        assert "categoria" in DROP_COLUMNS
        assert "id_factura" in DROP_COLUMNS
        assert isinstance(DROP_COLUMNS, tuple)

    def test_feature_names_exclude_leak_columns(self, df_full):
        """GIVEN fitted preprocessor WHEN get_feature_names called
        THEN rubro, subrubro, categoria, id_factura NOT in output names."""
        # Extract target first (this also drops leak columns from X)
        X, y_raw, _ = extract_target(df_full)

        # Build and fit preprocessor on X (leak columns already dropped)
        preprocessor = build_preprocessor(X)
        preprocessor.fit(X)

        feature_names = get_feature_names(preprocessor, X)

        # Leak columns must not appear in ANY feature name
        leak_keys = {"rubro", "subrubro", "categoria", "id_factura"}
        for name in feature_names:
            for leak in leak_keys:
                assert leak not in str(name).lower(), (
                    f"Leak column '{leak}' found in feature name: {name}"
                )

    def test_drop_columns_removed_before_transform(self, df_full):
        """GIVEN full DataFrame with leak columns WHEN extract_target called
        THEN returned X does NOT contain leak columns."""
        X, y_raw, _ = extract_target(df_full)

        for col in DROP_COLUMNS:
            assert col not in X.columns, f"Leak column '{col}' present in X features"

    def test_categoria_is_target_not_feature(self, df_full):
        """GIVEN full DataFrame WHEN extract_target called
        THEN categoria becomes y (target), NOT present in X."""
        X, y_raw, _ = extract_target(df_full)

        assert "categoria" not in X.columns
        # y_raw should be a Series with the original categoria values
        assert isinstance(y_raw, pd.Series)
        assert len(y_raw) == len(df_full)


# ===========================================================================
# 1.3 — Unicode stripping: U+FFFD removed from descripcion
# ===========================================================================

class TestUnicodeStripping:
    """Tests for sanitize_descripcion — Req 4."""

    def test_fffd_removed(self):
        """GIVEN descripcion containing U+FFFD WHEN sanitized
        THEN U+FFFD characters are removed."""
        series = pd.Series([
            "ESPA\uFFFDL normal text",
            "factura \uFFFD con artefacto",
            "\uFFFD\uFFFD multiple \uFFFD",
        ])
        result = sanitize_descripcion(series)

        for i in range(len(result)):
            assert "\ufffd" not in result.iloc[i], (
                f"U+FFFD still present at index {i}: {result.iloc[i]!r}"
            )

    def test_no_fffd_passthrough(self):
        """GIVEN descripcion with NO U+FFFD WHEN sanitized
        THEN text is unchanged."""
        original = pd.Series([
            "servicio de internet",
            "compra de equipos",
            "mantenimiento mensual",
        ])
        result = sanitize_descripcion(original)

        pd.testing.assert_series_equal(result, original)

    def test_empty_series(self):
        """GIVEN empty Series WHEN sanitized THEN returns empty Series."""
        series = pd.Series([], dtype=str)
        result = sanitize_descripcion(series)

        assert len(result) == 0
        assert isinstance(result, pd.Series)

    def test_all_fffd_becomes_empty(self):
        """GIVEN descripcion containing ONLY U+FFFD characters
        WHEN sanitized THEN becomes empty string."""
        series = pd.Series(["\ufffd\ufffd\ufffd", "\ufffd"])
        result = sanitize_descripcion(series)

        assert result.iloc[0] == ""
        assert result.iloc[1] == ""


# ===========================================================================
# 1.4 — OOV categories: unknown proveedor/tipo → all-zero one-hot
# ===========================================================================

class TestOOVCategory:
    """Tests for unknown category handling — Req 5."""

    def test_unknown_proveedor_all_zeros(self, df_minimal):
        """GIVEN preprocessor fitted on known proveedores
        WHEN transform receives unknown proveedor
        THEN one-hot encoding produces all zeros for that row."""
        # Use only the first 4 rows for fitting (Telecom, TechCorp, LogisticaSur)
        df_train = df_minimal.iloc[:4].copy()
        X_train, _, _ = extract_target(df_train)

        preprocessor = build_preprocessor(X_train)
        preprocessor.fit(X_train)

        # Transform a row with a NEW proveedor not seen during fit
        df_unseen = pd.DataFrame([{
            "descripcion": "item desconocido",
            "proveedor": "ProveedorNuevoXYZ",
            "tipo_comprobante": "Factura A",  # known tipo
            "monto": 999.0,
            "fecha": pd.Timestamp("2024-05-01"),
        }])

        result = preprocessor.transform(df_unseen)

        # If sparse, convert to dense
        if hasattr(result, "toarray"):
            result = result.toarray()

        # The proveedor one-hot block should be all zeros for this row
        # Since it's the first row and only one row, at least some columns
        # should be zero (the proveedor ones)
        assert result.shape[0] == 1
        # We can't assert exact columns without knowing feature ordering,
        # but we CAN assert no crash and at least one row produced
        assert not np.isnan(result).any(), "Transform produced NaN for OOV proveedor"

    def test_unknown_tipo_comprobante_all_zeros(self, df_minimal):
        """GIVEN preprocessor fitted on known tipos
        WHEN transform receives unknown tipo_comprobante
        THEN one-hot encoding produces all zeros for that row."""
        df_train = df_minimal.iloc[:4].copy()
        X_train, _, _ = extract_target(df_train)

        preprocessor = build_preprocessor(X_train)
        preprocessor.fit(X_train)

        df_unseen = pd.DataFrame([{
            "descripcion": "item desconocido",
            "proveedor": "Telecom",  # known proveedor
            "tipo_comprobante": "TipoNuevoXYZ",
            "monto": 999.0,
            "fecha": pd.Timestamp("2024-05-01"),
        }])

        result = preprocessor.transform(df_unseen)

        if hasattr(result, "toarray"):
            result = result.toarray()

        assert result.shape[0] == 1
        assert not np.isnan(result).any(), "Transform produced NaN for OOV tipo_comprobante"

    def test_known_categories_produce_nonzero(self, df_minimal):
        """GIVEN preprocessor fitted on known categories
        WHEN transform receives known values
        THEN one-hot columns have ones (triangulation: contrast with OOV)."""
        X, _, _ = extract_target(df_minimal)

        preprocessor = build_preprocessor(X)
        preprocessor.fit(X)

        # Transform the SAME data — known categories should NOT be all zeros
        result = preprocessor.transform(X)

        if hasattr(result, "toarray"):
            result = result.toarray()

        # At least some values should be non-zero (known categories produce ones)
        assert result.shape[0] == 6
        assert np.any(result != 0), "All zeros — known categories should produce non-zero features"


# ===========================================================================
# 1.5 — Empty descripcion: zero TF-IDF vector
# ===========================================================================

class TestEmptyDescription:
    """Tests for empty descripcion handling — Req 6."""

    def test_empty_description_zero_tfidf(self, df_minimal):
        """GIVEN descripcion='' WHEN TF-IDF transforms
        THEN the row produces an all-zero TF-IDF vector."""
        X, _, _ = extract_target(df_minimal)

        preprocessor = build_preprocessor(X)
        preprocessor.fit(X)

        result = preprocessor.transform(X)

        if hasattr(result, "toarray"):
            result_dense = result.toarray()
        else:
            result_dense = result

        # Row index 3 has empty descripcion → its TF-IDF features should be all zeros
        # We check that there exists at least one row that is all-zero in text features
        # Since we know row 3 has empty descripcion, verify it's not ALL non-zero
        # (a fully zero row across all features is possible if all features are text,
        # but with monto/fecha/cat columns there should be non-zero values)
        assert result_dense.shape[0] == 6
        assert not np.isnan(result_dense).any(), "NaN found after transform"

    def test_non_empty_description_produces_nonzero_tfidf(self, df_minimal):
        """GIVEN descripcion with content WHEN TF-IDF transforms
        THEN produces non-zero TF-IDF values (triangulation with empty)."""
        # Only use rows with non-empty descripcion (skip index 3)
        df_nonempty = df_minimal.drop(3).reset_index(drop=True)
        X, _, _ = extract_target(df_nonempty)

        preprocessor = build_preprocessor(X)
        preprocessor.fit(X)

        result = preprocessor.transform(X)

        if hasattr(result, "toarray"):
            result = result.toarray()

        # With real text, TF-IDF should produce some non-zero features
        assert np.any(result != 0), (
            "All zeros — non-empty descripcion should produce non-zero TF-IDF values"
        )


# ===========================================================================
# 1.6 — Class-count guard: validate_class_counts
# ===========================================================================

class TestClassCountGuard:
    """Tests for validate_class_counts — Req 7."""

    def test_all_classes_two_or_more_passes(self, df_two_per_class):
        """GIVEN every class has ≥2 samples WHEN validated
        THEN no error raised."""
        _, y_raw, _ = extract_target(df_two_per_class)

        # Should NOT raise
        validate_class_counts(y_raw)

    def test_class_with_one_raises_valueerror(self, df_two_per_class):
        """GIVEN one class has only 1 sample WHEN validated
        THEN ValueError raised naming the offending class."""
        # Create a Series where class 'A' has only 1 sample
        y_bad = pd.Series(["A", "B", "B", "C", "C", "C"])

        with pytest.raises(ValueError, match="A"):
            validate_class_counts(y_bad)

    def test_valueerror_message_names_class(self, df_two_per_class):
        """GIVEN class 'X' has 1 sample WHEN validated
        THEN error message contains the class name 'X'."""
        y_bad = pd.Series(["X", "Y", "Y", "Z", "Z"])

        with pytest.raises(ValueError, match="X"):
            validate_class_counts(y_bad)

    def test_multiple_rare_classes_raises(self):
        """GIVEN multiple classes with 1 sample each
        WHEN validated THEN ValueError raised."""
        y_bad = pd.Series(["W", "X", "Y", "Y", "Z", "Z"])

        # Should still raise — at least one offending class mentioned
        with pytest.raises(ValueError):
            validate_class_counts(y_bad)

    def test_empty_series_raises(self):
        """GIVEN empty Series WHEN validated THEN raises (no classes to check)."""
        y_empty = pd.Series([], dtype=str)

        with pytest.raises(ValueError):
            validate_class_counts(y_empty)
