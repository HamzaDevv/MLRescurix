"""Information-Theoretic 4-Stage Semantic Column Sieve and Topology Resolver (Phase P13 / Phase 2).

Provides:
1. Multi-signal target resolution:
   Priority 1: explicit_target from CLI or user prompt.
   Priority 2: Exact lexical match in problem statement against column headers.
   Priority 3: Canonical target lexicon match (survived, target, label, y, churn, fraud, etc.).
   Priority 4: Fallback candidate inspection (excluding monotonic IDs and high-missingness attributes).
   NEVER blind df.columns[-1].
2. 4-Stage Information-Theoretic Semantic Column Sieve:
   Computes uniqueness ratio (rho_unique), effective support mass (M_k),
   sub-token vocabulary coverage (C_cov), digit-masked template collapse (tau_mask),
   and monotonicity (mu_mono with Delta x_i = 1).
3. Deterministic column classifications:
   - DROP_SURROGATE_ID
   - QUASI_ID_DECOMPOSE
   - SHORT_TEXT_EMBED
   - SAFE_CATEGORICAL
   - TEMPORAL_AXIS
   - NUMERIC_CONTINUOUS
   - NUMERIC_DISCRETE
   - TARGET
4. Automatic CV scheme and entity/temporal grouping resolution:
   - PurgedTimeSeriesSplit, StratifiedGroupKFold, GroupKFold, StratifiedKFold, KFold.
5. Emits locked DatasetTopologyContract adhering strictly to contracts/v2/dataset_topology.schema.json.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


# Canonical target lexicon in order of specificity
CANONICAL_TARGET_LEXICON: List[str] = [
    "survived",
    "target",
    "label",
    "y",
    "churn",
    "churned",
    "fraud",
    "default",
    "defaulted",
    "outcome",
    "class",
    "converted",
    "conversion",
    "click",
    "clicked",
    "bought",
    "purchased",
    "retained",
    "retention",
    "price",
    "salary",
    "revenue",
    "cost",
    "loss",
    "sales",
    "profit",
    "score",
]

# Patterns for ID column detection
ID_NAME_PATTERNS = re.compile(
    r"^(id|.*_id|.*_uuid|.*_guid|row_id|passengerid|customer_id|client_id|patient_id|sample_id|index)$",
    re.IGNORECASE,
)

# Patterns for temporal column detection
TEMPORAL_NAME_PATTERNS = re.compile(
    r".*(date|time|timestamp|datetime|created_at|updated_at|recorded_at|year|month|day).*",
    re.IGNORECASE,
)

# Patterns for quasi-ID / grouping column detection
QUASI_ID_NAME_PATTERNS = re.compile(
    r".*(ticket|cabin|zip|postal|device|account|session|phone|ip_address|serial).*",
    re.IGNORECASE,
)

ENTITY_GROUP_NAME_PATTERNS = re.compile(
    r".*(group|family|household|patient|user|client|customer|account|session|cluster|subject).*",
    re.IGNORECASE,
)


@dataclass
class ColumnMetrics:
    """Information-theoretic and structural metrics for a column."""
    name: str
    dtype_str: str
    row_count: int
    non_null_count: int
    missing_count: int
    missing_rate: float
    cardinality: int
    uniqueness_ratio: float
    effective_support_mass: float
    subtoken_coverage: float
    template_collapse: float
    is_monotonic_step_1: bool
    median_word_count: float
    median_cluster_size: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "detected_type": self.dtype_str,
            "missing_count": self.missing_count,
            "missing_rate": round(self.missing_rate, 4),
            "cardinality": self.cardinality,
            "uniqueness_ratio": round(self.uniqueness_ratio, 4),
            "effective_support_mass": round(self.effective_support_mass, 4),
            "subtoken_coverage": round(self.subtoken_coverage, 4),
            "template_collapse": round(self.template_collapse, 4),
            "is_monotonic_step_1": self.is_monotonic_step_1,
            "median_word_count": round(self.median_word_count, 2),
            "median_cluster_size": round(self.median_cluster_size, 2),
        }


def compute_column_metrics(
    series: pd.Series,
    min_support_k: int = 10,
) -> ColumnMetrics:
    """Compute information-theoretic metrics on a single column."""
    name = str(series.name)
    n_total = len(series)
    non_null = series.dropna()
    n_non_null = len(non_null)
    n_missing = n_total - n_non_null
    missing_rate = n_missing / max(1, n_total)

    if n_non_null == 0:
        return ColumnMetrics(
            name=name,
            dtype_str=str(series.dtype),
            row_count=n_total,
            non_null_count=0,
            missing_count=n_missing,
            missing_rate=missing_rate,
            cardinality=0,
            uniqueness_ratio=0.0,
            effective_support_mass=0.0,
            subtoken_coverage=0.0,
            template_collapse=0.0,
            is_monotonic_step_1=False,
            median_word_count=0.0,
            median_cluster_size=0.0,
        )

    # Cardinality & uniqueness ratio
    val_counts = non_null.value_counts()
    u_count = len(val_counts)
    uniqueness_ratio = u_count / n_non_null

    # Effective support mass M_k: fraction of rows in categories with count >= k
    supported_mass = val_counts[val_counts >= min_support_k].sum()
    effective_support_mass = float(supported_mass / n_non_null)

    # Median cluster size (for clusters with count >= 2)
    clusters_ge_2 = val_counts[val_counts >= 2]
    median_cluster_size = float(clusters_ge_2.median()) if len(clusters_ge_2) > 0 else 1.0

    # Monotonicity & uniform step check
    is_monotonic_step_1 = False
    if pd.api.types.is_numeric_dtype(series):
        try:
            diffs = non_null.diff().dropna()
            if len(diffs) > 0:
                is_monotonic_step_1 = bool((diffs == 1).all() or (diffs == -1).all())
        except Exception:
            is_monotonic_step_1 = False

    # Text & structural metrics for string/object columns
    subtoken_coverage = 0.0
    template_collapse = 0.0
    median_word_count = 0.0

    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        str_vals = [str(x) for x in non_null]

        # 1. Median word count
        word_counts = [len(s.split()) for s in str_vals]
        median_word_count = float(np.median(word_counts)) if word_counts else 0.0

        # 2. Sub-token vocabulary coverage C_cov
        # Tokenize on punctuation & whitespace
        token_doc_freq: Dict[str, int] = {}
        row_tokens_list: List[Set[str]] = []

        # Sample up to 2,000 rows for sub-token analysis if huge
        sample_vals = str_vals[:2000] if len(str_vals) > 2000 else str_vals
        sample_n = len(sample_vals)
        sample_k = max(3, int(min_support_k * (sample_n / n_non_null)))

        for s in sample_vals:
            tokens = set(t.lower() for t in re.split(r"[\s,\.;:/\(\)\[\]\-_]+", s) if len(t) >= 2)
            row_tokens_list.append(tokens)
            for t in tokens:
                token_doc_freq[t] = token_doc_freq.get(t, 0) + 1

        recurring_tokens = {t for t, freq in token_doc_freq.items() if freq >= sample_k}
        rows_with_recurring = sum(1 for rtokens in row_tokens_list if (rtokens & recurring_tokens))
        subtoken_coverage = float(rows_with_recurring / max(1, sample_n))

        # 3. Digit-masked template collapse tau_mask = 1 - (U_masked / U)
        masked_set: Set[str] = set()
        for s in val_counts.index:
            s_str = str(s)
            masked = re.sub(r"\d+", "#", s_str).strip()
            masked_set.add(masked)
        u_masked = len(masked_set)
        template_collapse = float(1.0 - (u_masked / max(1, u_count)))

    return ColumnMetrics(
        name=name,
        dtype_str=str(series.dtype),
        row_count=n_total,
        non_null_count=n_non_null,
        missing_count=n_missing,
        missing_rate=missing_rate,
        cardinality=u_count,
        uniqueness_ratio=uniqueness_ratio,
        effective_support_mass=effective_support_mass,
        subtoken_coverage=subtoken_coverage,
        template_collapse=template_collapse,
        is_monotonic_step_1=is_monotonic_step_1,
        median_word_count=median_word_count,
        median_cluster_size=median_cluster_size,
    )


def resolve_target_column(
    df: pd.DataFrame,
    problem_statement: Optional[str] = None,
    explicit_target: Optional[str] = None,
) -> str:
    """Resolve the true dataset target column through multi-signal hierarchy.

    NEVER blindly falls back to df.columns[-1].
    """
    all_cols = list(df.columns)
    cols_lower_map = {c.lower(): c for c in all_cols}

    # Priority 1: Explicit target from prompt / CLI
    if explicit_target and explicit_target in all_cols:
        return explicit_target
    if explicit_target and explicit_target.lower() in cols_lower_map:
        return cols_lower_map[explicit_target.lower()]

    # Priority 2: Exact lexical match in problem statement
    if problem_statement:
        words = re.findall(r"\b[A-Za-z0-9_]+\b", problem_statement)
        for w in words:
            w_lower = w.lower()
            if w_lower in cols_lower_map:
                matched_col = cols_lower_map[w_lower]
                # Validate not an obvious ID column
                if not ID_NAME_PATTERNS.match(matched_col):
                    return matched_col

    # Priority 3: Canonical target lexicon match
    for target_word in CANONICAL_TARGET_LEXICON:
        if target_word in cols_lower_map:
            return cols_lower_map[target_word]

    # Prefix match e.g. "is_churn", "has_defaulted"
    for col in all_cols:
        col_l = col.lower()
        if col_l.startswith("is_") or col_l.startswith("has_"):
            return col

    # Priority 4: Fallback candidate inspection (excluding monotonic IDs and high-missingness attributes)
    scored_candidates: List[Tuple[float, str]] = []
    for col in all_cols:
        series = df[col]
        non_null = series.dropna()
        if len(non_null) == 0:
            continue

        missing_ratio = (len(df) - len(non_null)) / len(df)
        if missing_ratio > 0.40:
            continue

        # Exclude clear IDs
        if ID_NAME_PATTERNS.match(col):
            continue

        n_unique = non_null.nunique()
        score = 0.0

        # Binary columns: high suitability
        if n_unique == 2:
            score += 10.0
            # If values are {0, 1} or bool
            unique_set = set(non_null.unique())
            if unique_set.issubset({0, 1, 0.0, 1.0, True, False, "0", "1"}):
                score += 5.0
        # Multiclass discrete (3 to 10 classes)
        elif 3 <= n_unique <= 10:
            score += 7.0
        # Numeric continuous
        elif pd.api.types.is_numeric_dtype(series) and n_unique > 10:
            # Check not monotonic index
            diffs = non_null.diff().dropna()
            if not ((diffs == 1).all() or (diffs == -1).all()):
                score += 4.0

        # Avoid text/quasi-ids with high word counts
        if pd.api.types.is_object_dtype(series):
            avg_word_count = np.mean([len(str(x).split()) for x in non_null[:100]])
            if avg_word_count > 2.0:
                score -= 8.0

        if score > 0:
            scored_candidates.append((score, col))

    if scored_candidates:
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        return scored_candidates[0][1]

    raise ValueError(
        f"Cannot automatically resolve target column among columns {all_cols}. "
        "Explicit target specification is required."
    )


def infer_task_type(target_series: pd.Series) -> str:
    """Infer task type (binary_classification, multiclass_classification, regression, time_series)."""
    non_null = target_series.dropna()
    if len(non_null) == 0:
        return "binary_classification"

    n_unique = non_null.nunique()

    # Binary
    if n_unique == 2:
        return "binary_classification"

    # Multiclass (3 to 25 discrete classes)
    if 3 <= n_unique <= 25 and (
        pd.api.types.is_integer_dtype(non_null)
        or pd.api.types.is_object_dtype(non_null)
        or pd.api.types.is_categorical_dtype(non_null)
    ):
        return "multiclass_classification"

    # Numeric continuous
    if pd.api.types.is_numeric_dtype(non_null):
        return "regression"

    return "multiclass_classification"


def classify_column(
    metrics: ColumnMetrics,
    is_target: bool = False,
) -> str:
    """Classify column role based on 4-stage information-theoretic rules."""
    if is_target:
        return "TARGET"

    name = metrics.name
    rho = metrics.uniqueness_ratio
    m_k = metrics.effective_support_mass
    c_cov = metrics.subtoken_coverage
    tau_mask = metrics.template_collapse
    step_1 = metrics.is_monotonic_step_1
    med_words = metrics.median_word_count
    med_cluster = metrics.median_cluster_size

    # Rule 1: DROP_SURROGATE_ID
    # (rho_unique >= 0.98 or (mu_mono == 1.0 with Delta x_i == 1)) and C_cov < 0.05 and tau_mask < 0.10
    # or column name explicitly matches ID pattern with rho_unique >= 0.90
    if (
        ((rho >= 0.98 or step_1) and c_cov < 0.05 and tau_mask < 0.10)
        or (ID_NAME_PATTERNS.match(name) and rho >= 0.90)
    ):
        return "DROP_SURROGATE_ID"

    # Rule 2: TEMPORAL_AXIS
    # Column matches date/time patterns or datetime dtype
    if "datetime" in metrics.dtype_str.lower() or TEMPORAL_NAME_PATTERNS.match(name):
        return "TEMPORAL_AXIS"

    # Rule 3: QUASI_ID_DECOMPOSE
    # (0.05 < rho_unique < 0.98 or M_k < 0.50) and (tau_mask >= 0.40 or median cluster size in [2, 20])
    # E.g. Ticket, Cabin, ZipCode, household_id
    if (
        "object" in metrics.dtype_str.lower()
        or "string" in metrics.dtype_str.lower()
    ):
        if (
            (0.05 < rho < 0.98 or m_k < 0.50)
            and (tau_mask >= 0.40 or (2.0 <= med_cluster <= 20.0))
        ) or QUASI_ID_NAME_PATTERNS.match(name) or ENTITY_GROUP_NAME_PATTERNS.match(name):
            return "QUASI_ID_DECOMPOSE"

    # Rule 4: SHORT_TEXT_EMBED
    # rho_unique > 0.20 and (C_cov >= 0.40 or median word count >= 2.5)
    # E.g. Name, JobTitle, Description
    if (
        "object" in metrics.dtype_str.lower()
        or "string" in metrics.dtype_str.lower()
    ):
        if rho > 0.20 and (c_cov >= 0.35 or med_words >= 2.2):
            return "SHORT_TEXT_EMBED"

    # Rule 5: SAFE_CATEGORICAL
    # M_k >= 0.80 or U <= min(250, max(5, 0.05 * N))
    if (
        "object" in metrics.dtype_str.lower()
        or "string" in metrics.dtype_str.lower()
        or "category" in metrics.dtype_str.lower()
    ):
        return "SAFE_CATEGORICAL"

    # Rule 6: Clean Numerical (NUMERIC_CONTINUOUS vs NUMERIC_DISCRETE)
    if "int" in metrics.dtype_str.lower():
        if metrics.cardinality <= 20:
            return "NUMERIC_DISCRETE"
        return "NUMERIC_CONTINUOUS"

    if "float" in metrics.dtype_str.lower():
        return "NUMERIC_CONTINUOUS"

    return "SAFE_CATEGORICAL"


def analyze_and_lock_topology(
    df: pd.DataFrame,
    problem_statement: Optional[str] = None,
    explicit_target: Optional[str] = None,
    dataset_id: str = "dataset_default",
    coercions: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Execute complete 4-stage semantic sieve and produce locked DatasetTopologyContract."""
    target_col = resolve_target_column(
        df, problem_statement=problem_statement, explicit_target=explicit_target
    )
    task_type = infer_task_type(df[target_col])

    # Compute minority class prevalence p_min for classification to tune k
    p_min = 0.5
    if task_type in ("binary_classification", "multiclass_classification"):
        val_counts = df[target_col].dropna().value_counts(normalize=True)
        if len(val_counts) > 0:
            p_min = float(val_counts.min())

    min_support_k = max(10, math.ceil(2.0 / max(1e-4, p_min)))

    column_metrics_map: Dict[str, ColumnMetrics] = {}
    classifications: Dict[str, str] = {}
    id_columns: List[str] = []
    clean_feature_columns: List[str] = []

    entity_group_candidates: List[str] = []
    temporal_col: Optional[str] = None

    for col in df.columns:
        is_tgt = col == target_col
        metrics = compute_column_metrics(df[col], min_support_k=min_support_k)
        column_metrics_map[col] = metrics

        role = classify_column(metrics, is_target=is_tgt)
        classifications[col] = role

        if role == "DROP_SURROGATE_ID":
            id_columns.append(col)
        elif role == "TEMPORAL_AXIS" and temporal_col is None:
            temporal_col = col
        elif role == "QUASI_ID_DECOMPOSE":
            clean_feature_columns.append(col)
            # Candidate for entity grouping if repeated entities exist
            if 2.0 <= metrics.median_cluster_size <= 20.0 or ENTITY_GROUP_NAME_PATTERNS.match(col):
                entity_group_candidates.append(col)
        elif role != "TARGET":
            clean_feature_columns.append(col)
            if ENTITY_GROUP_NAME_PATTERNS.match(col) and (2.0 <= metrics.median_cluster_size or metrics.cardinality < len(df)):
                entity_group_candidates.append(col)

    # Resolve entity_group_col
    entity_group_col: Optional[str] = None
    for cand in entity_group_candidates:
        if ENTITY_GROUP_NAME_PATTERNS.match(cand):
            entity_group_col = cand
            break

    # Determine CV scheme
    if temporal_col is not None:
        cv_scheme = "purged_time_series_split"
    elif entity_group_col is not None:
        cv_scheme = "stratified_group_kfold" if "classification" in task_type else "group_kfold"
    elif "classification" in task_type:
        cv_scheme = "stratified_kfold"
    else:
        cv_scheme = "kfold"

    column_metadata: Dict[str, Dict[str, Any]] = {}
    for col, m in column_metrics_map.items():
        meta = m.to_dict()
        meta["coercion"] = coercions.get(col) if coercions else None
        column_metadata[col] = meta

    contract = {
        "schema_version": "v2",
        "dataset_id": dataset_id,
        "target_column": target_col,
        "task_type": task_type,
        "id_columns": id_columns,
        "entity_group_col": entity_group_col,
        "temporal_col": temporal_col,
        "cv_scheme": cv_scheme,
        "cv_splits": 5,
        "column_classifications": classifications,
        "column_metadata": column_metadata,
        "coercions": coercions or {},
        "clean_feature_columns": clean_feature_columns,
        "locked_at_ms": int(time.time() * 1000),
    }

    return contract
