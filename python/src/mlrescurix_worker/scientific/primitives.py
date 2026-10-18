"""Leakage-Proof Scientific Primitives for MLRescurix Feature Engineering (Phase P14 / Phase 3).

Provides scikit-learn compatible `BaseEstimator` and `TransformerMixin` classes with
strict in-fold fitting (zero label leakage) and seamless `joblib` serialization compatibility:

1. `NestedOOFEmpiricalBayesEncoder`:
   In-fold K_inner = 5 split target encoding with Empirical Bayes shrinkage:
     m = max(min_smoothing, sigma^2_c / tau^2_global)
     y_hat_c = (n_c * y_bar_c + m * y_bar_global) / (n_c + m)
   Gaussian noise N(0, noise_level * std(y)) during training transform to prevent tree
   memorization, and global prior fallback for unseen categories during transform().

2. `QuasiIdDecomposer`:
   Drops raw ID strings, extracts <col>__group_size, numeric aggregations per member
   (<col>__<num>_mean, <num> / <col>__group_size), alphabetic prefix, structural mask
   (\\d+ -> #), string length, and digit ratio.

3. `LightweightTextEmbedder`:
   Sub-token indicator extraction with target Mutual Information gating (extracts honorifics
   like 'Mr', 'Mrs', 'Miss', 'Dr', 'Master'), surname cluster size, and in-fold
   Char + Word TF-IDF followed by TruncatedSVD(n_components=16).

4. `InformativeMissingnessAndInteractions`:
   Binary <col>__is_nan indicators for missingness rate in [0.5%, 99.5%], and candidate
   pairwise numeric interactions (A / (|B| + eps), A * B, A - B) gated by fast
   Mutual Information on the training fold.
"""

from __future__ import annotations

import math
import re
import warnings
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import FeatureUnion


class NestedOOFEmpiricalBayesEncoder(BaseEstimator, TransformerMixin):
    """Nested Out-Of-Fold Empirical Bayes Target Encoder.
    
    Prevents categorical memorization and target leakage:
    - Inner K-fold cross-validation during `fit_transform()` guarantees that target values
      are never encoded using the row's own fold.
    - Empirical Bayes shrinkage pulls categories with low sample counts or high variance
      toward the global prior:
        m = max(min_smoothing, sigma^2_c / tau^2_global)
        y_hat_c = (n_c * y_bar_c + m * y_bar_global) / (n_c + m)
    - Deterministic Gaussian noise N(0, noise_level * std(y)) is injected during training
      transform so downstream gradient boosting trees cannot isolate exact target floats.
    - Subsequent `transform()` calls on validation/test sets use the full-fold posterior
      lookup table; unseen categories or categories with n_c < 2 fall back to y_bar_global.
    """

    def __init__(
        self,
        categorical_columns: Optional[Sequence[str]] = None,
        n_splits: int = 5,
        min_smoothing: float = 10.0,
        noise_level: float = 0.01,
        random_state: Optional[int] = 42,
        drop_original: bool = True,
        encoded_suffix: str = "__eb",
    ):
        self.categorical_columns = list(categorical_columns) if categorical_columns else None
        self.n_splits = max(2, int(n_splits))
        self.min_smoothing = float(min_smoothing)
        self.noise_level = float(noise_level)
        self.random_state = random_state
        self.drop_original = bool(drop_original)
        self.encoded_suffix = encoded_suffix

        # Learned state (joblib serializable)
        self.cols_to_encode_: List[str] = []
        self.global_mean_: float = 0.0
        self.global_var_: float = 1.0
        self.y_std_: float = 1.0
        self.mapping_: Dict[str, Dict[str, float]] = {}
        self.counts_: Dict[str, Dict[str, int]] = {}

    def _resolve_cols(self, X: pd.DataFrame) -> List[str]:
        if self.categorical_columns is not None:
            return [c for c in self.categorical_columns if c in X.columns]
        # Auto-detect string, object, category
        res = []
        for col in X.columns:
            dt = X[col].dtype
            if dt == "object" or isinstance(dt, pd.CategoricalDtype) or str(dt) == "string":
                res.append(col)
        return res

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray, Sequence],
        groups: Optional[Sequence] = None,
    ) -> "NestedOOFEmpiricalBayesEncoder":
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X)
        else:
            X_df = X

        if y is None:
            raise ValueError("NestedOOFEmpiricalBayesEncoder requires target y for fitting.")

        y_arr = np.asarray(y, dtype=float)
        self.cols_to_encode_ = self._resolve_cols(X_df)

        self.global_mean_ = float(np.nanmean(y_arr))
        var = float(np.nanvar(y_arr))
        self.global_var_ = var if var > 1e-9 else 1.0
        std = float(np.nanstd(y_arr))
        self.y_std_ = std if std > 1e-9 else 1.0

        self.mapping_ = {}
        self.counts_ = {}

        for col in self.cols_to_encode_:
            series = X_df[col].astype(str).reset_index(drop=True)
            col_map: Dict[str, float] = {}
            col_counts: Dict[str, int] = {}

            for val, group in series.groupby(series):
                idx = group.index
                n_c = len(idx)
                y_c = y_arr[idx]
                col_counts[str(val)] = n_c

                if n_c < 1:
                    continue
                y_bar_c = float(np.nanmean(y_c))
                if n_c >= 2:
                    sigma2_c = float(np.nanvar(y_c, ddof=1))
                else:
                    sigma2_c = self.global_var_

                m = max(self.min_smoothing, sigma2_c / self.global_var_)
                hat_y_c = (n_c * y_bar_c + m * self.global_mean_) / (n_c + m)
                col_map[str(val)] = float(hat_y_c)

            self.mapping_[col] = col_map
            self.counts_[col] = col_counts

        return self

    def fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray, Sequence],
        groups: Optional[Sequence] = None,
        **fit_params: Any,
    ) -> pd.DataFrame:
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X)
        else:
            X_df = X

        if y is None:
            raise ValueError("NestedOOFEmpiricalBayesEncoder requires target y for fit_transform.")

        y_arr = np.asarray(y, dtype=float)
        n_samples = len(X_df)

        # First fit global stats for future transform()
        self.fit(X_df, y_arr, groups=groups)

        X_out = X_df.copy()
        if not self.cols_to_encode_:
            return X_out

        # Select inner splitter
        unique_y = np.unique(y_arr[~np.isnan(y_arr)])
        is_classification = len(unique_y) <= 20 and np.all(np.equal(np.mod(unique_y, 1), 0))

        if groups is not None:
            if is_classification:
                splitter = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            else:
                splitter = GroupKFold(n_splits=self.n_splits)
        elif is_classification:
            # Check minority class count
            min_class_count = min(np.sum(y_arr == u) for u in unique_y)
            if min_class_count >= self.n_splits:
                splitter = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            else:
                splitter = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        else:
            splitter = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        rng = np.random.RandomState(self.random_state)

        for col in self.cols_to_encode_:
            series = X_df[col].astype(str).reset_index(drop=True)
            oof_encoded = np.full(n_samples, self.global_mean_, dtype=float)

            splits = list(splitter.split(X_df, y_arr, groups=groups))
            for train_idx, val_idx in splits:
                s_tr = series.iloc[train_idx].reset_index(drop=True)
                y_tr = y_arr[train_idx]
                fold_mean = float(np.nanmean(y_tr))
                fold_var = float(np.nanvar(y_tr))
                if fold_var <= 1e-9:
                    fold_var = 1.0

                fold_map: Dict[str, float] = {}
                for val, grp in s_tr.groupby(s_tr):
                    idx = grp.index
                    n_c = len(idx)
                    # Unseen or n_c < 2 smoothly maps to fold_mean
                    if n_c < 2:
                        continue
                    y_c = y_tr[idx]
                    y_bar_c = float(np.nanmean(y_c))
                    sigma2_c = float(np.nanvar(y_c, ddof=1))
                    m = max(self.min_smoothing, sigma2_c / fold_var)
                    fold_map[str(val)] = float((n_c * y_bar_c + m * fold_mean) / (n_c + m))

                s_val = series.iloc[val_idx]
                val_mapped = s_val.map(fold_map).fillna(fold_mean).to_numpy(dtype=float)
                oof_encoded[val_idx] = val_mapped

            # Inject small Gaussian noise during training transform to prevent tree memorization
            if self.noise_level > 0.0 and self.y_std_ > 0.0:
                noise = rng.normal(0.0, self.noise_level * self.y_std_, size=n_samples)
                oof_encoded += noise

            target_col_name = f"{col}{self.encoded_suffix}" if self.encoded_suffix else col
            X_out[target_col_name] = oof_encoded
            if self.drop_original and target_col_name != col and col in X_out.columns:
                X_out.drop(columns=[col], inplace=True)

        return X_out

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X)
        else:
            X_df = X

        X_out = X_df.copy()
        for col in self.cols_to_encode_:
            if col not in X_df.columns:
                continue
            series = X_df[col].astype(str)
            col_map = self.mapping_.get(col, {})
            col_counts = self.counts_.get(col, {})

            # Categories with n_c < 2 fall back to global prior
            valid_map = {k: v for k, v in col_map.items() if col_counts.get(k, 0) >= 2}
            encoded = series.map(valid_map).fillna(self.global_mean_).to_numpy(dtype=float)

            target_col_name = f"{col}{self.encoded_suffix}" if self.encoded_suffix else col
            X_out[target_col_name] = encoded
            if self.drop_original and target_col_name != col and col in X_out.columns:
                X_out.drop(columns=[col], inplace=True)

        return X_out


class QuasiIdDecomposer(BaseEstimator, TransformerMixin):
    """Decomposes Quasi-ID columns (Ticket, Cabin, ZipCode) into structural & group features.
    
    Drops raw ID strings while extracting high-signal group and structural features:
    - `<col>__group_size`: Cluster count across rows.
    - `<col>__<num>_mean`: In-fold mean of numeric covariates per group.
    - `<col>__<num>_per_member`: Numeric covariate divided by group size (`<num> / <col>__group_size`).
    - `<col>__prefix`: Alphabetic structural prefix (e.g., 'PC 17599' -> 'PC', 'C85' -> 'C').
    - `<col>__mask`: Digit-masked template ('\\d+' -> '#').
    - `<col>__str_len`: String length.
    - `<col>__digit_ratio`: Proportion of characters that are digits.
    - Optional Empirical Bayes encoding of low-cardinality prefixes.
    """

    def __init__(
        self,
        id_columns: Optional[Sequence[str]] = None,
        numeric_columns: Optional[Sequence[str]] = None,
        encode_prefix: bool = True,
        drop_original: bool = True,
        drop_structural_strings: bool = True,
        random_state: Optional[int] = 42,
    ):
        self.id_columns = list(id_columns) if id_columns else None
        self.numeric_columns = list(numeric_columns) if numeric_columns else None
        self.encode_prefix = bool(encode_prefix)
        self.drop_original = bool(drop_original)
        self.drop_structural_strings = bool(drop_structural_strings)
        self.random_state = random_state

        # Learned state
        self.resolved_id_cols_: List[str] = []
        self.resolved_num_cols_: List[str] = []
        self.group_sizes_: Dict[str, Dict[str, int]] = {}
        self.numeric_means_: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.global_num_means_: Dict[str, float] = {}
        self.prefix_freqs_: Dict[str, Dict[str, float]] = {}
        self.mask_freqs_: Dict[str, Dict[str, float]] = {}
        self.prefix_eb_encoders_: Dict[str, NestedOOFEmpiricalBayesEncoder] = {}

    @staticmethod
    def extract_prefix(val: Any) -> str:
        if val is None or pd.isna(val):
            return "MISSING"
        s = str(val).strip()
        if not s:
            return "EMPTY"
        match = re.match(r"^([a-zA-Z]+)", s)
        if match:
            return match.group(1).upper()
        return "NUM"

    @staticmethod
    def extract_mask(val: Any) -> str:
        if val is None or pd.isna(val):
            return "MISSING"
        s = str(val).strip()
        if not s:
            return "EMPTY"
        return re.sub(r"\d+", "#", s)

    def _resolve_cols(self, X: pd.DataFrame) -> Tuple[List[str], List[str]]:
        if self.id_columns is not None:
            id_cols = [c for c in self.id_columns if c in X.columns]
        else:
            id_cols = [
                c for c in X.columns
                if X[c].dtype == "object" or isinstance(X[c].dtype, pd.CategoricalDtype) or str(X[c].dtype) == "string"
            ]

        if self.numeric_columns is not None:
            num_cols = [c for c in self.numeric_columns if c in X.columns and pd.api.types.is_numeric_dtype(X[c])]
        else:
            num_cols = [c for c in X.columns if c not in id_cols and pd.api.types.is_numeric_dtype(X[c])]

        return id_cols, num_cols

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray, Sequence]] = None,
        groups: Optional[Sequence] = None,
    ) -> "QuasiIdDecomposer":
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.resolved_id_cols_, self.resolved_num_cols_ = self._resolve_cols(X_df)

        self.group_sizes_ = {}
        self.numeric_means_ = {}
        self.global_num_means_ = {}
        self.prefix_freqs_ = {}
        self.mask_freqs_ = {}
        self.prefix_eb_encoders_ = {}

        # Store global numeric means
        for num_col in self.resolved_num_cols_:
            self.global_num_means_[num_col] = float(X_df[num_col].dropna().mean()) if len(X_df[num_col].dropna()) > 0 else 0.0

        n_rows = len(X_df)
        for col in self.resolved_id_cols_:
            s = X_df[col].astype(str)
            # Group size map
            counts = s.value_counts().to_dict()
            self.group_sizes_[col] = counts

            # Numeric means per group
            self.numeric_means_[col] = {}
            for num_col in self.resolved_num_cols_:
                valid = X_df[[col, num_col]].dropna()
                if len(valid) > 0:
                    means = valid.groupby(col)[num_col].mean().to_dict()
                    self.numeric_means_[col][num_col] = {str(k): float(v) for k, v in means.items()}
                else:
                    self.numeric_means_[col][num_col] = {}

            # Prefixes & masks
            prefixes = s.map(self.extract_prefix)
            prefix_counts = prefixes.value_counts(normalize=True).to_dict()
            self.prefix_freqs_[col] = {str(k): float(v) for k, v in prefix_counts.items()}

            masks = s.map(self.extract_mask)
            mask_counts = masks.value_counts(normalize=True).to_dict()
            self.mask_freqs_[col] = {str(k): float(v) for k, v in mask_counts.items()}

            # Fit Empirical Bayes on prefixes if y provided
            if self.encode_prefix and y is not None:
                eb = NestedOOFEmpiricalBayesEncoder(
                    categorical_columns=[f"{col}__prefix"],
                    n_splits=5,
                    min_smoothing=10.0,
                    noise_level=0.01,
                    random_state=self.random_state,
                    drop_original=True,
                    encoded_suffix="_eb",
                )
                df_prefix = pd.DataFrame({f"{col}__prefix": prefixes})
                eb.fit(df_prefix, y, groups=groups)
                self.prefix_eb_encoders_[col] = eb

        return self

    def fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray, Sequence]] = None,
        groups: Optional[Sequence] = None,
        **fit_params: Any,
    ) -> pd.DataFrame:
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.fit(X_df, y=y, groups=groups)
        return self._transform_internal(X_df, y=y, is_fit=True, groups=groups)

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        return self._transform_internal(X_df, y=None, is_fit=False)

    def _transform_internal(
        self,
        X_df: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray, Sequence]] = None,
        is_fit: bool = False,
        groups: Optional[Sequence] = None,
    ) -> pd.DataFrame:
        X_out = X_df.copy()

        for col in self.resolved_id_cols_:
            if col not in X_df.columns:
                continue

            s = X_df[col].astype(str)
            raw_s = X_df[col]

            # 1. Group size
            size_map = self.group_sizes_.get(col, {})
            # Unseen groups default to 1
            grp_size = s.map(size_map).fillna(1).to_numpy(dtype=float)
            X_out[f"{col}__group_size"] = grp_size

            # 2. Numeric aggregations
            for num_col in self.resolved_num_cols_:
                if num_col in X_df.columns:
                    col_means = self.numeric_means_.get(col, {}).get(num_col, {})
                    global_mean = self.global_num_means_.get(num_col, 0.0)
                    mean_val = s.map(col_means).fillna(global_mean).to_numpy(dtype=float)
                    X_out[f"{col}__{num_col}_mean"] = mean_val

                    # Per member aggregation: <num> / <col>__group_size
                    num_vals = pd.to_numeric(X_df[num_col], errors="coerce").to_numpy(dtype=float)
                    per_member = num_vals / np.maximum(grp_size, 1.0)
                    X_out[f"{col}__{num_col}_per_member"] = per_member
                    X_out[f"{col}__{num_col}_div_group_size"] = per_member

            # 3. String length & digit ratio
            def get_str_len(val: Any) -> float:
                if val is None or pd.isna(val):
                    return 0.0
                return float(len(str(val)))

            def get_digit_ratio(val: Any) -> float:
                if val is None or pd.isna(val):
                    return 0.0
                st = str(val)
                if not st:
                    return 0.0
                d_cnt = sum(ch.isdigit() for ch in st)
                return float(d_cnt / len(st))

            X_out[f"{col}__str_len"] = raw_s.map(get_str_len).to_numpy(dtype=float)
            X_out[f"{col}__digit_ratio"] = raw_s.map(get_digit_ratio).to_numpy(dtype=float)

            # 4. Prefix & Mask features
            prefixes = raw_s.map(self.extract_prefix)
            masks = raw_s.map(self.extract_mask)

            p_map = self.prefix_freqs_.get(col, {})
            m_map = self.mask_freqs_.get(col, {})
            X_out[f"{col}__prefix_freq"] = prefixes.map(p_map).fillna(0.0).to_numpy(dtype=float)
            X_out[f"{col}__mask_freq"] = masks.map(m_map).fillna(0.0).to_numpy(dtype=float)

            if not self.drop_structural_strings:
                X_out[f"{col}__prefix"] = prefixes
                X_out[f"{col}__structural_mask"] = masks

            # Empirical Bayes encoded prefix
            if col in self.prefix_eb_encoders_:
                eb = self.prefix_eb_encoders_[col]
                df_p = pd.DataFrame({f"{col}__prefix": prefixes})
                if is_fit and y is not None:
                    encoded_p = eb.fit_transform(df_p, y, groups=groups)
                else:
                    encoded_p = eb.transform(df_p)
                for c in encoded_p.columns:
                    X_out[c] = encoded_p[c]

            # 5. Drop raw ID string column
            if self.drop_original and col in X_out.columns:
                X_out.drop(columns=[col], inplace=True)

        return X_out


class LightweightTextEmbedder(BaseEstimator, TransformerMixin):
    """Extracts sub-token indicators & in-fold TF-IDF SVD embeddings from text columns (Name, Description).
    
    Extracts high-signal information at zero LLM token cost:
    - Target Mutual Information gated sub-tokens: Captures recurring titles ('Mr', 'Mrs', 'Miss', 'Dr').
    - Surname / prefix cluster size: Captures family groupings ('family_surname_count').
    - In-fold Word + Char N-gram TF-IDF SVD: TruncatedSVD(n_components=16) fitted strictly in-fold.
    """

    def __init__(
        self,
        text_columns: Optional[Sequence[str]] = None,
        n_svd_components: int = 16,
        top_k_subtokens: int = 25,
        min_subtoken_freq: int = 3,
        random_state: int = 42,
        drop_original: bool = True,
    ):
        self.text_columns = list(text_columns) if text_columns else None
        self.n_svd_components = max(2, int(n_svd_components))
        self.top_k_subtokens = max(1, int(top_k_subtokens))
        self.min_subtoken_freq = max(1, int(min_subtoken_freq))
        self.random_state = random_state
        self.drop_original = bool(drop_original)

        # Learned state
        self.resolved_text_cols_: List[str] = []
        self.selected_subtokens_: Dict[str, List[str]] = {}
        self.surname_counts_: Dict[str, Dict[str, int]] = {}
        self.tfidf_svd_pipelines_: Dict[str, Tuple[FeatureUnion, TruncatedSVD]] = {}

    @staticmethod
    def extract_surname(val: Any) -> str:
        if val is None or pd.isna(val):
            return "UNKNOWN"
        s = str(val).strip()
        if not s:
            return "UNKNOWN"
        if "," in s:
            return s.split(",")[0].strip().lower()
        parts = s.split()
        return parts[0].strip().lower() if parts else "UNKNOWN"

    @staticmethod
    def tokenize(text: str) -> List[str]:
        if not text or pd.isna(text):
            return []
        tokens = re.split(r"[\s,\.;:/\(\)\[\]\-_\"\'\?]+", str(text).lower())
        return [t for t in tokens if len(t) >= 2 and not t.isdigit()]

    def _resolve_cols(self, X: pd.DataFrame) -> List[str]:
        if self.text_columns is not None:
            return [c for c in self.text_columns if c in X.columns]
        res = []
        for col in X.columns:
            dt = X[col].dtype
            if dt == "object" or isinstance(dt, pd.CategoricalDtype) or str(dt) == "string":
                # Check median word count
                sample = X[col].dropna().astype(str).head(100)
                if len(sample) > 0:
                    med_words = np.median([len(s.split()) for s in sample])
                    if med_words >= 2.0:
                        res.append(col)
        return res

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray, Sequence]] = None,
    ) -> "LightweightTextEmbedder":
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.resolved_text_cols_ = self._resolve_cols(X_df)

        self.selected_subtokens_ = {}
        self.surname_counts_ = {}
        self.tfidf_svd_pipelines_ = {}

        y_arr = np.asarray(y, dtype=float) if y is not None else None
        n_samples = len(X_df)

        for col in self.resolved_text_cols_:
            s = X_df[col].fillna("").astype(str)

            # 1. Surname cluster size
            surnames = s.map(self.extract_surname)
            counts = surnames.value_counts().to_dict()
            self.surname_counts_[col] = {str(k): int(v) for k, v in counts.items()}

            # 2. Sub-token indicator extraction with target MI gating
            all_token_lists = [self.tokenize(text) for text in s]
            token_doc_counts: Dict[str, int] = {}
            for t_list in all_token_lists:
                for token in set(t_list):
                    token_doc_counts[token] = token_doc_counts.get(token, 0) + 1

            candidate_tokens = [
                token for token, cnt in token_doc_counts.items()
                if cnt >= self.min_subtoken_freq and len(token) >= 2
            ]

            if candidate_tokens and y_arr is not None:
                # Build binary indicator matrix
                M = np.zeros((n_samples, len(candidate_tokens)), dtype=np.int8)
                token_to_idx = {t: i for i, t in enumerate(candidate_tokens)}
                for row_idx, t_list in enumerate(all_token_lists):
                    for t in set(t_list):
                        if t in token_to_idx:
                            M[row_idx, token_to_idx[t]] = 1

                # Fast Mutual Information gating
                unique_y = np.unique(y_arr[~np.isnan(y_arr)])
                is_classification = len(unique_y) <= 20 and np.all(np.equal(np.mod(unique_y, 1), 0))

                valid_mask = ~np.isnan(y_arr)
                if np.sum(valid_mask) > 10:
                    if is_classification:
                        mi_scores = mutual_info_classif(
                            M[valid_mask], y_arr[valid_mask], random_state=self.random_state
                        )
                    else:
                        mi_scores = mutual_info_regression(
                            M[valid_mask], y_arr[valid_mask], random_state=self.random_state
                        )
                    top_indices = np.argsort(mi_scores)[::-1][: self.top_k_subtokens]
                    self.selected_subtokens_[col] = [candidate_tokens[idx] for idx in top_indices]
                else:
                    sorted_by_freq = sorted(candidate_tokens, key=lambda t: token_doc_counts[t], reverse=True)
                    self.selected_subtokens_[col] = sorted_by_freq[: self.top_k_subtokens]
            else:
                sorted_by_freq = sorted(candidate_tokens, key=lambda t: token_doc_counts[t], reverse=True)
                self.selected_subtokens_[col] = sorted_by_freq[: self.top_k_subtokens]

            # 3. In-fold Word + Char TF-IDF followed by TruncatedSVD
            union = FeatureUnion([
                ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=2500)),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=2500)),
            ])
            n_comp = min(self.n_svd_components, max(2, n_samples - 1))
            svd = TruncatedSVD(n_components=n_comp, random_state=self.random_state)

            try:
                tfidf_mat = union.fit_transform(s)
                svd.fit(tfidf_mat)
                self.tfidf_svd_pipelines_[col] = (union, svd)
            except Exception as e:
                warnings.warn(f"TF-IDF SVD fitting failed for column {col}: {e}")

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        X_out = X_df.copy()

        for col in self.resolved_text_cols_:
            if col not in X_df.columns:
                continue

            s = X_df[col].fillna("").astype(str)

            # 1. Surname cluster size
            surnames = s.map(self.extract_surname)
            s_map = self.surname_counts_.get(col, {})
            # Unseen surnames default to 1
            cluster_sizes = surnames.map(s_map).fillna(1).to_numpy(dtype=float)
            X_out[f"{col}__surname_cluster_size"] = cluster_sizes
            X_out[f"{col}__family_surname_count"] = cluster_sizes

            # 2. Sub-token indicator features
            selected_tokens = self.selected_subtokens_.get(col, [])
            token_lists = [set(self.tokenize(text)) for text in s]
            for token in selected_tokens:
                clean_name = re.sub(r"\W+", "_", token)
                indicator = np.array([1 if token in t_set else 0 for t_set in token_lists], dtype=np.int8)
                X_out[f"{col}__has_{clean_name}"] = indicator

            # 3. In-fold Char + Word TF-IDF + SVD embeddings
            if col in self.tfidf_svd_pipelines_:
                union, svd = self.tfidf_svd_pipelines_[col]
                try:
                    tfidf_mat = union.transform(s)
                    svd_mat = svd.transform(tfidf_mat)
                    for k in range(svd_mat.shape[1]):
                        X_out[f"{col}__svd_{k}"] = svd_mat[:, k].astype(float)
                except Exception as e:
                    warnings.warn(f"TF-IDF SVD transform failed for column {col}: {e}")

            # 4. Drop raw text strings
            if self.drop_original and col in X_out.columns:
                X_out.drop(columns=[col], inplace=True)

        return X_out


class InformativeMissingnessAndInteractions(BaseEstimator, TransformerMixin):
    """Informative missingness indicators & fast MI-gated pairwise numeric interactions.
    
    1. Creates binary `<col>__is_nan` indicators for every column with missingness rate in [0.5%, 99.5%].
    2. Generates candidate pairwise numeric interactions (A / (|B| + eps), A * B, A - B) among
       top numeric features, scores them via fast Mutual Information against target y on the training fold,
       and retains only winning interactions that exceed baseline parent MI.
    """

    def __init__(
        self,
        numeric_columns: Optional[Sequence[str]] = None,
        missing_rate_bounds: Tuple[float, float] = (0.005, 0.995),
        max_candidate_features: int = 8,
        max_interactions: int = 10,
        epsilon: float = 1e-5,
        random_state: int = 42,
    ):
        self.numeric_columns = list(numeric_columns) if numeric_columns else None
        self.missing_rate_bounds = missing_rate_bounds
        self.max_candidate_features = max(2, int(max_candidate_features))
        self.max_interactions = max(1, int(max_interactions))
        self.epsilon = float(epsilon)
        self.random_state = random_state

        # Learned state
        self.informative_missing_cols_: List[str] = []
        self.selected_interactions_: List[Tuple[str, str, str, str]] = []  # (colA, colB, op, out_name)
        self.parent_medians_: Dict[str, float] = {}

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray, Sequence]] = None,
    ) -> "InformativeMissingnessAndInteractions":
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        n_samples = len(X_df)
        min_rate, max_rate = self.missing_rate_bounds

        # 1. Detect informative missingness
        self.informative_missing_cols_ = []
        for col in X_df.columns:
            rate = float(X_df[col].isna().mean())
            if min_rate <= rate <= max_rate:
                self.informative_missing_cols_.append(col)

        # 2. Select numeric columns
        if self.numeric_columns is not None:
            num_cols = [c for c in self.numeric_columns if c in X_df.columns and pd.api.types.is_numeric_dtype(X_df[c])]
        else:
            num_cols = [c for c in X_df.columns if pd.api.types.is_numeric_dtype(X_df[c])]

        # Filter out constants or columns with < 3 unique values
        valid_num_cols = [c for c in num_cols if X_df[c].nunique(dropna=True) >= 3]

        self.parent_medians_ = {}
        for col in valid_num_cols:
            self.parent_medians_[col] = float(X_df[col].median()) if not pd.isna(X_df[col].median()) else 0.0

        self.selected_interactions_ = []

        if len(valid_num_cols) < 2:
            return self

        # If y is provided, score candidate features via MI
        y_arr = np.asarray(y, dtype=float) if y is not None else None
        selected_parent_cols = valid_num_cols

        if y_arr is not None and len(valid_num_cols) > self.max_candidate_features:
            valid_mask = ~np.isnan(y_arr)
            X_clean = np.column_stack([
                X_df[col].fillna(self.parent_medians_[col]).to_numpy(dtype=float)
                for col in valid_num_cols
            ])
            unique_y = np.unique(y_arr[valid_mask])
            is_classification = len(unique_y) <= 20 and np.all(np.equal(np.mod(unique_y, 1), 0))

            if is_classification:
                mi = mutual_info_classif(X_clean[valid_mask], y_arr[valid_mask], random_state=self.random_state)
            else:
                mi = mutual_info_regression(X_clean[valid_mask], y_arr[valid_mask], random_state=self.random_state)

            top_idx = np.argsort(mi)[::-1][: self.max_candidate_features]
            selected_parent_cols = [valid_num_cols[i] for i in top_idx]
        elif len(valid_num_cols) > self.max_candidate_features:
            selected_parent_cols = valid_num_cols[: self.max_candidate_features]

        # Generate candidate interactions and gate by MI against parents
        if y_arr is not None:
            valid_mask = ~np.isnan(y_arr)
            unique_y = np.unique(y_arr[valid_mask])
            is_classification = len(unique_y) <= 20 and np.all(np.equal(np.mod(unique_y, 1), 0))

            # Compute parent MIs
            parent_mi: Dict[str, float] = {}
            for col in selected_parent_cols:
                s_clean = X_df[col].fillna(self.parent_medians_[col]).to_numpy(dtype=float)[valid_mask].reshape(-1, 1)
                if is_classification:
                    parent_mi[col] = float(mutual_info_classif(s_clean, y_arr[valid_mask], random_state=self.random_state)[0])
                else:
                    parent_mi[col] = float(mutual_info_regression(s_clean, y_arr[valid_mask], random_state=self.random_state)[0])

            candidate_specs: List[Tuple[str, str, str, str, np.ndarray, float]] = []

            for i in range(len(selected_parent_cols)):
                for j in range(i + 1, len(selected_parent_cols)):
                    col_a = selected_parent_cols[i]
                    col_b = selected_parent_cols[j]
                    val_a = X_df[col_a].fillna(self.parent_medians_[col_a]).to_numpy(dtype=float)
                    val_b = X_df[col_b].fillna(self.parent_medians_[col_b]).to_numpy(dtype=float)

                    max_parent_mi = max(parent_mi[col_a], parent_mi[col_b])

                    # 1. Ratio A / (|B| + eps)
                    ratio_ab = val_a / (np.abs(val_b) + self.epsilon)
                    candidate_specs.append((col_a, col_b, "div", f"{col_a}__div__{col_b}", ratio_ab, max_parent_mi))

                    # 2. Reverse Ratio B / (|A| + eps)
                    ratio_ba = val_b / (np.abs(val_a) + self.epsilon)
                    candidate_specs.append((col_b, col_a, "div", f"{col_b}__div__{col_a}", ratio_ba, max_parent_mi))

                    # 3. Product A * B
                    prod = val_a * val_b
                    candidate_specs.append((col_a, col_b, "mul", f"{col_a}__mul__{col_b}", prod, max_parent_mi))

                    # 4. Difference A - B
                    diff = val_a - val_b
                    candidate_specs.append((col_a, col_b, "sub", f"{col_a}__sub__{col_b}", diff, max_parent_mi))

            # Evaluate MI for all candidates
            winning_candidates = []
            for col_a, col_b, op, out_name, arr, max_p_mi in candidate_specs:
                clean_arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)[valid_mask].reshape(-1, 1)
                if is_classification:
                    cand_mi = float(mutual_info_classif(clean_arr, y_arr[valid_mask], random_state=self.random_state)[0])
                else:
                    cand_mi = float(mutual_info_regression(clean_arr, y_arr[valid_mask], random_state=self.random_state)[0])

                lift = cand_mi - max_p_mi
                if lift > 0.005:  # Strict positive lift over parent
                    winning_candidates.append((lift, cand_mi, (col_a, col_b, op, out_name)))

            # Sort by lift descending
            winning_candidates.sort(key=lambda x: x[0], reverse=True)
            self.selected_interactions_ = [item[2] for item in winning_candidates[: self.max_interactions]]

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        X_out = X_df.copy()

        # 1. Informative missingness indicators
        for col in self.informative_missing_cols_:
            if col in X_df.columns:
                X_out[f"{col}__is_nan"] = X_df[col].isna().astype(np.int8)

        # 2. Pairwise numeric interactions
        for col_a, col_b, op, out_name in self.selected_interactions_:
            if col_a not in X_df.columns or col_b not in X_df.columns:
                continue

            med_a = self.parent_medians_.get(col_a, 0.0)
            med_b = self.parent_medians_.get(col_b, 0.0)

            val_a = pd.to_numeric(X_df[col_a], errors="coerce").fillna(med_a).to_numpy(dtype=float)
            val_b = pd.to_numeric(X_df[col_b], errors="coerce").fillna(med_b).to_numpy(dtype=float)

            if op == "div":
                res = val_a / (np.abs(val_b) + self.epsilon)
            elif op == "mul":
                res = val_a * val_b
            elif op == "sub":
                res = val_a - val_b
            else:
                continue

            res = np.nan_to_num(res, nan=0.0, posinf=1e6, neginf=-1e6)
            X_out[out_name] = res

        return X_out
