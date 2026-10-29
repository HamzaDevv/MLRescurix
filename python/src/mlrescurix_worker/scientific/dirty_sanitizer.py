"""Zero-copy / Low-overhead Dirty Data Sanitizer and Type Normalizer (Phase P13 / Phase 2).

Provides:
1. Fast identification and normalization of universal sentinels (?, -999, -9999, N/A, null, etc.) to native NaN/None.
2. Robust parsing and coercion of dirty string numbers:
   - Accounting negative parentheses: "(1,234.50)" -> -1234.50
   - Percentages: "15.4%" -> 0.154
   - Currency & comma-formatted numbers: "$1,200.50", "€45.00" -> 1200.50, 45.00
3. Column promotion rule:
   - If >= 85% of non-null values in a string/object column coerce cleanly to float,
     the column is promoted to Float64, and the coercion rule is recorded.
4. Pure deterministic inference-time transform for bundle serving (`apply_coercions`)
   ensuring zero train-serve skew.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
try:
    import pyarrow as pa
except ImportError:
    pa = None


# Universal missingness sentinel tokens
UNIVERSAL_SENTINELS: Set[str] = {
    "?",
    "-999",
    "-9999",
    "-999.0",
    "-9999.0",
    "N/A",
    "n/a",
    "NA",
    "na",
    "null",
    "NULL",
    "None",
    "none",
    "nan",
    "NaN",
    "NAN",
    "",
    " ",
    "#N/A",
    "#NA",
    "nil",
    "NIL",
    "missing",
    "MISSING",
    "undefined",
    "UNDEFINED",
    "inf",
    "-inf",
    "NoneType",
}

# Compiled regex patterns for fast dirty numeric identification
_ACCOUNTING_PATTERN = re.compile(
    r"^\s*\(?\s*[$€£¥]?\s*\(?\s*([+-]?[0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?|[+-]?\.[0-9]+|[+-]?[0-9]+(?:\.[0-9]+)?)\s*\)?\s*[$€£¥]?\s*\)?\s*$"
)
_OUTER_PARENS_PATTERN = re.compile(r"^\s*\((.*)\)\s*$")
_PERCENTAGE_PATTERN = re.compile(
    r"^\s*[$€£¥]?\s*([+-]?[0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?|[+-]?\.[0-9]+|[+-]?[0-9]+(?:\.[0-9]+)?)\s*%\s*$"
)
_CURRENCY_PATTERN = re.compile(
    r"^\s*[$€£¥]\s*([+-]?[0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?|[+-]?\.[0-9]+|[+-]?[0-9]+(?:\.[0-9]+)?)\s*$"
    r"|^\s*([+-]?[0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?|[+-]?\.[0-9]+|[+-]?[0-9]+(?:\.[0-9]+)?)\s*[$€£¥]\s*$"
)
_PLAIN_NUMBER_PATTERN = re.compile(
    r"^\s*([+-]?[0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?|[+-]?\.[0-9]+|[+-]?[0-9]+(?:\.[0-9]+)?)\s*$"
)


@dataclass
class DirtySanitizerResult:
    """Outcome of dirty data sanitization."""
    df: pd.DataFrame
    coercions: Dict[str, str] = field(default_factory=dict)
    sentinels_cleaned_count: int = 0
    promoted_columns: List[str] = field(default_factory=list)


def parse_dirty_numeric_value(val: Any) -> Tuple[Optional[float], Optional[str]]:
    """Parse a single value into float and record detected category.

    Returns:
        (parsed_float, category_name) or (None, None) if not parseable.
        Categories: "accounting_parentheses", "percentage", "currency", "dirty_numeric"
    """
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None, None

    if isinstance(val, (int, float, np.integer, np.floating)):
        fval = float(val)
        return (fval, "dirty_numeric") if not math.isnan(fval) else (None, None)

    s = str(val).strip()
    if not s or s in UNIVERSAL_SENTINELS or s.lower() in UNIVERSAL_SENTINELS:
        return None, None

    # 1. Accounting negative parentheses e.g. "(1,234.50)" or "(450)"
    paren_match = _OUTER_PARENS_PATTERN.match(s)
    if paren_match:
        inner = paren_match.group(1).strip()
        # Clean currency or commas inside parens
        cleaned = inner.replace("$", "").replace("€", "").replace("£", "").replace("¥", "").replace(",", "").strip()
        try:
            parsed = float(cleaned)
            return -abs(parsed), "accounting_parentheses"
        except ValueError:
            pass

    # 2. Percentage e.g. "15.4%" or " -2.5% "
    pct_match = _PERCENTAGE_PATTERN.match(s)
    if pct_match:
        cleaned = pct_match.group(1).replace(",", "").strip()
        try:
            parsed = float(cleaned) / 100.0
            return parsed, "percentage"
        except ValueError:
            pass

    # 3. Currency e.g. "$1,200.50" or "€ 45"
    curr_match = _CURRENCY_PATTERN.match(s)
    if curr_match:
        matched_str = curr_match.group(1) or curr_match.group(2)
        cleaned = matched_str.replace(",", "").strip()
        try:
            parsed = float(cleaned)
            return parsed, "currency"
        except ValueError:
            pass

    # 4. Standard comma-formatted or plain float string e.g. "1,234.56"
    plain_match = _PLAIN_NUMBER_PATTERN.match(s)
    if plain_match:
        cleaned = plain_match.group(1).replace(",", "").strip()
        try:
            parsed = float(cleaned)
            return parsed, "dirty_numeric"
        except ValueError:
            pass

    return None, None


def is_sentinel(val: Any) -> bool:
    """Check if value matches universal missing sentinels."""
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    if isinstance(val, (int, float)):
        # Check standard integer sentinel -999, -9999
        if val in (-999, -9999, -999.0, -9999.0):
            return True
        return False
    s = str(val).strip()
    return not s or s in UNIVERSAL_SENTINELS or s.lower() in UNIVERSAL_SENTINELS


def sanitize_dataframe(
    df: pd.DataFrame,
    promotion_threshold: float = 0.85,
    inplace: bool = False,
) -> DirtySanitizerResult:
    """Sanitize a DataFrame by normalizing sentinels and promoting dirty string columns to float64.

    Args:
        df: Input DataFrame.
        promotion_threshold: Fraction of non-null values that must parse to float (default 0.85).
        inplace: Whether to mutate the DataFrame in place.

    Returns:
        DirtySanitizerResult with sanitized df, coercions map, and stats.
    """
    res_df = df if inplace else df.copy()
    coercions: Dict[str, str] = {}
    promoted_columns: List[str] = []
    total_sentinels_cleaned = 0

    for col in res_df.columns:
        series = res_df[col]

        # Case 1: Column is object, string, or categorical
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            raw_vals = series.to_numpy(dtype=object, na_value=None)
            n_rows = len(raw_vals)

            # First pass: identify sentinels and count non-null valid strings
            valid_indices = []
            valid_values = []
            sentinel_mask = np.zeros(n_rows, dtype=bool)

            for idx, val in enumerate(raw_vals):
                if is_sentinel(val):
                    sentinel_mask[idx] = True
                    total_sentinels_cleaned += 1
                else:
                    valid_indices.append(idx)
                    valid_values.append(val)

            n_valid = len(valid_values)
            if n_valid == 0:
                # Column is entirely sentinels/empty
                res_df[col] = np.nan
                continue

            # Second pass: test numeric coercibility
            parsed_floats: List[Optional[float]] = []
            categories: List[str] = []
            success_count = 0

            for val in valid_values:
                fval, cat = parse_dirty_numeric_value(val)
                parsed_floats.append(fval)
                if fval is not None:
                    success_count += 1
                    if cat:
                        categories.append(cat)

            coercion_rate = success_count / n_valid

            if coercion_rate >= promotion_threshold:
                # Determine predominant coercion category
                cat_counts: Dict[str, int] = {}
                for c in categories:
                    cat_counts[c] = cat_counts.get(c, 0) + 1
                predominant_cat = max(cat_counts, key=cat_counts.get) if cat_counts else "dirty_numeric"

                coercion_type_tag = f"{predominant_cat}_to_float64"
                coercions[col] = coercion_type_tag
                promoted_columns.append(col)

                # Construct new float64 array
                new_arr = np.full(n_rows, np.nan, dtype=np.float64)
                for pos, orig_idx in enumerate(valid_indices):
                    fval = parsed_floats[pos]
                    if fval is not None:
                        new_arr[orig_idx] = fval

                res_df[col] = new_arr
            else:
                # Not promoted to numeric, but replace sentinels with None
                if sentinel_mask.any():
                    new_vals = list(raw_vals)
                    for idx in np.where(sentinel_mask)[0]:
                        new_vals[idx] = None
                    res_df[col] = pd.Series(new_vals, index=res_df.index, dtype=object)

        # Case 2: Column is already numeric
        elif pd.api.types.is_numeric_dtype(series):
            # Check for standard numerical sentinel values -999, -9999 if rest of column is strictly non-negative
            non_null_s = series.dropna()
            if len(non_null_s) > 0:
                has_sentinel = ((non_null_s == -999) | (non_null_s == -9999)).any()
                if has_sentinel:
                    # Check if all other values are non-negative (e.g. Age, Fare, counts)
                    non_sentinel_vals = non_null_s[(non_null_s != -999) & (non_null_s != -9999)]
                    if len(non_sentinel_vals) > 0 and (non_sentinel_vals >= 0).all():
                        sentinel_mask = (series == -999) | (series == -9999)
                        total_sentinels_cleaned += int(sentinel_mask.sum())
                        res_df.loc[sentinel_mask, col] = np.nan

    return DirtySanitizerResult(
        df=res_df,
        coercions=coercions,
        sentinels_cleaned_count=total_sentinels_cleaned,
        promoted_columns=promoted_columns,
    )


def apply_coercions(df: pd.DataFrame, coercions: Dict[str, str]) -> pd.DataFrame:
    """Apply recorded column coercions to an input DataFrame during inference/serving.

    Ensures zero train-serve skew by running the exact identical parsing logic.
    """
    out = df.copy()
    for col, coercion_tag in coercions.items():
        if col not in out.columns:
            continue
        series = out[col]
        raw_vals = series.to_numpy(dtype=object, na_value=None)
        new_arr = np.full(len(raw_vals), np.nan, dtype=np.float64)

        for idx, val in enumerate(raw_vals):
            if is_sentinel(val):
                continue
            fval, _ = parse_dirty_numeric_value(val)
            if fval is not None:
                new_arr[idx] = fval

        out[col] = new_arr
    return out


def sanitize_dataset(
    data: Union[pd.DataFrame, Any, str, Path],
    promotion_threshold: float = 0.85,
) -> DirtySanitizerResult:
    """Ingest and sanitize dataset from DataFrame, Arrow Table, CSV, Parquet, or Feather path."""
    if isinstance(data, pd.DataFrame):
        return sanitize_dataframe(data, promotion_threshold=promotion_threshold)

    if pa is not None and isinstance(data, pa.Table):
        df = data.to_pandas()
        return sanitize_dataframe(df, promotion_threshold=promotion_threshold)

    p = Path(data)
    if not p.is_file():
        raise FileNotFoundError(f"Dataset path not found: {p}")

    ext = p.suffix.lower()
    if ext in (".parquet", ".pq"):
        import pyarrow.parquet as pq
        table = pq.read_table(p)
        df = table.to_pandas()
    elif ext in (".feather", ".arrow", ".ipc"):
        import pyarrow.feather as feather
        table = feather.read_table(p)
        df = table.to_pandas()
    else:
        # Default CSV
        df = pd.read_csv(p)

    return sanitize_dataframe(df, promotion_threshold=promotion_threshold)
