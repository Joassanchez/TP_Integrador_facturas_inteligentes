"""Supervised preprocessing: ColumnTransformer builder, DateFeatureExtractor,
target extraction, and class-count validation.

Follows sklearn-preprocessing skill: fit ONLY on training data.
All transformers wrapped in ColumnTransformer with remainder='drop'.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Columns that must NEVER enter as features (hard-drop).
# source_categoria and categoria_source are also excluded in extract_target
# but are audit artifacts not part of the canonical schema.
DROP_COLUMNS: tuple[str, ...] = (
    "rubro",
    "subrubro",
    "categoria",
    "id_factura",
)


# ---------------------------------------------------------------------------
# Text sanitization
# ---------------------------------------------------------------------------

def sanitize_descripcion(series: pd.Series) -> pd.Series:
    """Remove Unicode replacement character U+FFFD from descripcion.

    U+FFFD appears as an encoding artifact in Excel sources and inflates
    the TF-IDF vocabulary with spurious tokens.

    Args:
        series: pandas Series of text descriptions.

    Returns:
        pandas Series with U+FFFD stripped.
    """
    return series.str.replace("\ufffd", "", regex=False)


# ---------------------------------------------------------------------------
# Date feature extraction
# ---------------------------------------------------------------------------

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract month and cyclical sin/cos features from a date column.

    Uses 'fecha' as primary source.  Falls back to 'fecha_mes' when
    'fecha' is not present.  Never uses both simultaneously.

    Output features: month (int 1–12), sin_month (float), cos_month (float).
    """

    def fit(self, X, y=None):
        """Detect which date column to use and store its name.

        Args:
            X: DataFrame or Series with 'fecha' or 'fecha_mes'.
            y: Ignored (transformer interface).

        Returns:
            self
        """
        if hasattr(X, "columns"):
            if "fecha" in X.columns:
                self.date_col_ = "fecha"
            elif "fecha_mes" in X.columns:
                self.date_col_ = "fecha_mes"
            else:
                raise ValueError(
                    "DateFeatureExtractor requires 'fecha' or 'fecha_mes' column"
                )
        else:
            # Single column passed — no column name to store
            self.date_col_ = None
        return self

    def transform(self, X):
        """Extract month, sin_month, cos_month from the date column.

        Args:
            X: DataFrame or Series with date data.

        Returns:
            np.ndarray of shape (n_samples, 3).
        """
        if self.date_col_ is not None and hasattr(X, "columns"):
            dates = pd.to_datetime(X[self.date_col_])
        else:
            dates = pd.to_datetime(X)

        month = dates.dt.month.values.astype(float)

        # Cyclical encoding: sin(2π·month/12), cos(2π·month/12)
        sin_month = np.sin(2 * np.pi * month / 12.0)
        cos_month = np.cos(2 * np.pi * month / 12.0)

        return np.column_stack([month, sin_month, cos_month])

    def get_feature_names_out(self, input_features=None):
        """Return feature names for the output columns."""
        return np.array(["month", "sin_month", "cos_month"])


# ---------------------------------------------------------------------------
# Preprocessor builder
# ---------------------------------------------------------------------------

def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Build a ColumnTransformer for supervised preprocessing.

    Selects fecha (primary) or fecha_mes (fallback) for temporal features,
    never both.  Applies TF-IDF to descripcion, OneHotEncoder to proveedor
    and tipo_comprobante, log1p + StandardScaler to monto.

    Args:
        df: Feature DataFrame (target already separated).

    Returns:
        Unfitted ColumnTransformer with remainder='drop'.
    """
    # Determine temporal column
    if "fecha" in df.columns:
        date_col = "fecha"
    elif "fecha_mes" in df.columns:
        date_col = "fecha_mes"
    else:
        date_col = None

    transformers = [
        (
            "text",
            Pipeline([
                (
                    "sanitize",
                    FunctionTransformer(
                        sanitize_descripcion,
                        validate=False,
                        feature_names_out="one-to-one",
                    ),
                ),
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=2,
                        max_features=5000,
                    ),
                ),
            ]),
            "descripcion",
        ),
        (
            "cat_proveedor",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ["proveedor"],
        ),
        (
            "cat_tipo",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ["tipo_comprobante"],
        ),
        (
            "num",
            Pipeline([
                (
                    "log1p",
                    FunctionTransformer(
                        np.log1p, validate=False, feature_names_out="one-to-one"
                    ),
                ),
                ("scale", StandardScaler()),
            ]),
            ["monto"],
        ),
    ]

    if date_col is not None:
        transformers.append(("date", DateFeatureExtractor(), [date_col]))

    return ColumnTransformer(transformers, remainder="drop")


# ---------------------------------------------------------------------------
# Target extraction
# ---------------------------------------------------------------------------

def extract_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, LabelEncoder]:
    """Separate features from target, dropping leak and audit columns.

    Hard-drops rubro, subrubro, categoria, id_factura (and source_categoria /
    categoria_source if present) from the feature space.  Extracts categoria
    as raw target values and fits a LabelEncoder.

    Args:
        df: Full DataFrame with all columns including categoria.

    Returns:
        (X, y_raw, label_encoder) where:
        - X: feature DataFrame (leak columns dropped, target removed)
        - y_raw: raw categoria values as pd.Series (strings)
        - label_encoder: fitted LabelEncoder (caller applies .transform)
    """
    if "categoria" not in df.columns:
        raise ValueError("Target column 'categoria' not found in DataFrame")

    # Raw target (preserve before dropping from X)
    y_raw = df["categoria"].copy()

    # Columns to exclude from X (leak + audit)
    exclude = set(DROP_COLUMNS)
    exclude.update({"source_categoria", "categoria_source"} & set(df.columns))

    # Build X — keep only columns NOT in the exclusion set
    keep_cols = [c for c in df.columns if c not in exclude]
    X = df[keep_cols].copy()

    # Fit LabelEncoder on raw target
    le = LabelEncoder()
    le.fit(y_raw)

    return X, y_raw, le


# ---------------------------------------------------------------------------
# Class-count guard
# ---------------------------------------------------------------------------

def validate_class_counts(y_raw: pd.Series) -> None:
    """Validate every class has ≥2 samples before stratified split.

    Args:
        y_raw: Raw target values (strings, before encoding).

    Raises:
        ValueError: If any class has fewer than 2 samples, naming the
            offending class.
    """
    counts = y_raw.value_counts()

    if len(counts) == 0:
        raise ValueError("No classes found in target — empty series")

    rare = counts[counts < 2]
    if len(rare) > 0:
        offending = rare.index[0]
        raise ValueError(
            f"Class '{offending}' has only {rare.iloc[0]} sample(s). "
            f"Stratified split requires at least 2 samples per class. "
            f"Classes with <2 samples: {list(rare.index)}"
        )


# ---------------------------------------------------------------------------
# Feature name utility
# ---------------------------------------------------------------------------

def get_feature_names(
    preprocessor: ColumnTransformer, df: pd.DataFrame
) -> list[str]:
    """Return clean feature name list from a fitted preprocessor.

    Args:
        preprocessor: Fitted ColumnTransformer.
        df: Original input DataFrame (unused; kept for API compatibility).

    Returns:
        List of output feature name strings.
    """
    return list(preprocessor.get_feature_names_out())
