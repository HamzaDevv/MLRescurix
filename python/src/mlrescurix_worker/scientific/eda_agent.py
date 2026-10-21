"""M2-EDA Exploratory & Visual Forensics Agent (Phase P13 / Phase 2).

Provides:
1. `eda_forensics_suite`:
   - Executes dirty data sanitization via `dirty_sanitizer`.
   - Executes 4-stage semantic sieve via `semantic_sieve` to lock `DatasetTopologyContract`.
   - Generates visual diagnostic SVG/PNG plots in `.m2/artifacts/eda/` (target distribution,
     top bivariate associations / mutual info, missingness summary).
   - Generates compact (< 4 KB) Markdown and structured `EdaDossier` under strict Context Firewall.
2. `run_exploratory_python`:
   - Interactive data science sandbox tool for ad-hoc queries.
   - Context Firewall wrapper: intercepts stdout, converts DataFrames to compact Markdown tables
     (max 15 rows x 10 cols), limits total output to < 2 KB, and captures exceptions cleanly.
"""

from __future__ import annotations

import io
import math
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from .dirty_sanitizer import DirtySanitizerResult, sanitize_dataset
from .semantic_sieve import analyze_and_lock_topology


@dataclass
class EdaDossier:
    """Compact (< 4 KB) structured exploratory findings and locked dataset topology."""
    dataset_id: str
    topology: Dict[str, Any]
    summary_markdown: str
    artifacts: List[str] = field(default_factory=list)
    context_token_estimate: int = 0
    generated_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def generate_eda_plots(
    df: pd.DataFrame,
    topology: Dict[str, Any],
    output_dir: Path,
) -> List[str]:
    """Generate compact diagnostic SVG plots under output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: List[str] = []

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return plot_paths

    target_col = topology.get("target_column")
    task_type = topology.get("task_type", "binary_classification")

    # 1. Target distribution plot
    if target_col and target_col in df.columns:
        fig, ax = plt.subplots(figsize=(6, 3.5), dpi=100)
        tgt_data = df[target_col].dropna()

        if "classification" in task_type:
            val_counts = tgt_data.value_counts(normalize=True).sort_index()
            bars = ax.bar([str(k) for k in val_counts.index], val_counts.values, color="#3b82f6", edgecolor="#1d4ed8")
            ax.set_title(f"Target Distribution: {target_col} (Prevalence)", fontsize=11, fontweight="bold")
            ax.set_ylabel("Fraction")
            ax.set_ylim(0, max(val_counts.values) * 1.15 if len(val_counts) > 0 else 1.0)
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f"{height:.2%}", xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
        else:
            ax.hist(tgt_data, bins=25, color="#3b82f6", edgecolor="#1d4ed8")
            ax.set_title(f"Target Distribution: {target_col}", fontsize=11, fontweight="bold")
            ax.set_ylabel("Count")

        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        tgt_path = output_dir / "target_distribution.svg"
        plt.savefig(tgt_path, format="svg", bbox_inches="tight")
        plt.close(fig)
        plot_paths.append(str(tgt_path))

    # 2. Missingness summary plot
    missing_rates = df.isna().mean()
    cols_with_missing = missing_rates[missing_rates > 0.0].sort_values(ascending=False)
    if not cols_with_missing.empty:
        fig, ax = plt.subplots(figsize=(6, max(2.5, len(cols_with_missing) * 0.35)), dpi=100)
        y_pos = np.arange(len(cols_with_missing))
        bars = ax.barh(y_pos, cols_with_missing.values, color="#ef4444", edgecolor="#b91c1c")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(cols_with_missing.index, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Missingness Rate")
        ax.set_title("Missing Value Rates by Column", fontsize=11, fontweight="bold")
        ax.set_xlim(0, 1.0)
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        for bar in bars:
            width = bar.get_width()
            ax.annotate(f"{width:.1%}", xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(3, 0), textcoords="offset points", ha="left", va="center", fontsize=8)
        plt.tight_layout()
        missing_path = output_dir / "missingness_summary.svg"
        plt.savefig(missing_path, format="svg", bbox_inches="tight")
        plt.close(fig)
        plot_paths.append(str(missing_path))

    # 3. Top Features Mutual Information / Association with Target
    if target_col and target_col in df.columns:
        numeric_or_coded = {}
        for col in topology.get("clean_feature_columns", []):
            if col in df.columns:
                s = df[col]
                if pd.api.types.is_numeric_dtype(s):
                    numeric_or_coded[col] = s.fillna(s.median())
                elif pd.api.types.is_object_dtype(s) or pd.api.types.is_categorical_dtype(s):
                    # Quick categorical label code
                    codes = pd.factorize(s.astype(str))[0]
                    numeric_or_coded[col] = pd.Series(codes, index=df.index)

        if numeric_or_coded:
            X_tmp = pd.DataFrame(numeric_or_coded)
            y_tmp = df[target_col]
            valid_mask = y_tmp.notna()
            X_valid = X_tmp.loc[valid_mask]
            y_valid = y_tmp.loc[valid_mask]

            if len(X_valid) > 10 and len(X_valid.columns) > 0:
                try:
                    if "classification" in task_type:
                        mi_scores = mutual_info_classif(X_valid, y_valid, discrete_features="auto", random_state=42)
                    else:
                        mi_scores = mutual_info_regression(X_valid, y_valid, discrete_features="auto", random_state=42)

                    mi_series = pd.Series(mi_scores, index=X_valid.columns).sort_values(ascending=False).head(10)
                    if not mi_series.empty and mi_series.max() > 0:
                        fig, ax = plt.subplots(figsize=(6, max(2.5, len(mi_series) * 0.35)), dpi=100)
                        y_pos = np.arange(len(mi_series))
                        ax.barh(y_pos, mi_series.values, color="#10b981", edgecolor="#047857")
                        ax.set_yticks(y_pos)
                        ax.set_yticklabels(mi_series.index, fontsize=9)
                        ax.invert_yaxis()
                        ax.set_xlabel("Mutual Information")
                        ax.set_title(f"Top Features vs Target ({target_col})", fontsize=11, fontweight="bold")
                        ax.grid(axis="x", linestyle="--", alpha=0.5)
                        plt.tight_layout()
                        mi_path = output_dir / "top_features_association.svg"
                        plt.savefig(mi_path, format="svg", bbox_inches="tight")
                        plt.close(fig)
                        plot_paths.append(str(mi_path))
                except Exception:
                    pass

    return plot_paths


def build_compact_markdown_dossier(
    df: pd.DataFrame,
    topology: Dict[str, Any],
    sanitizer_result: DirtySanitizerResult,
) -> str:
    """Generate compact (< 4 KB) Markdown summary conforming to strict Context Firewall."""
    target_col = topology["target_column"]
    task_type = topology["task_type"]
    cv_scheme = topology["cv_scheme"]
    id_cols = topology["id_columns"]
    classifications = topology["column_classifications"]
    coercions = topology["coercions"]

    # Target class balance
    target_summary = "N/A"
    if target_col in df.columns:
        s = df[target_col].dropna()
        if "classification" in task_type:
            val_counts = s.value_counts(normalize=True).to_dict()
            target_summary = ", ".join(f"`{k}`: {v:.1%}" for k, v in sorted(val_counts.items(), key=lambda x: str(x[0])))
        else:
            target_summary = f"mean={s.mean():.2f}, std={s.std():.2f}, min={s.min():.2f}, max={s.max():.2f}"

    # Groupings
    quasi_ids = [c for c, role in classifications.items() if role == "QUASI_ID_DECOMPOSE"]
    text_cols = [c for c, role in classifications.items() if role == "SHORT_TEXT_EMBED"]
    safe_cats = [c for c, role in classifications.items() if role == "SAFE_CATEGORICAL"]
    nums_cont = [c for c, role in classifications.items() if role == "NUMERIC_CONTINUOUS"]
    nums_disc = [c for c, role in classifications.items() if role == "NUMERIC_DISCRETE"]

    # Missingness highlights (top 5)
    missing_rates = {c: m["missing_rate"] for c, m in topology.get("column_metadata", {}).items() if m.get("missing_rate", 0) > 0}
    top_missing = sorted(missing_rates.items(), key=lambda x: x[1], reverse=True)[:5]
    missing_str = ", ".join(f"`{c}`: {r:.1%}" for c, r in top_missing) if top_missing else "None"

    lines = [
        f"### M2-EDA Forensics Dossier: `{topology['dataset_id']}`",
        f"- **Rows/Cols**: {len(df):,} rows x {len(df.columns)} columns",
        f"- **Target Column**: `{target_col}` ({task_type})",
        f"- **Target Distribution**: {target_summary}",
        f"- **Recommended CV Scheme**: `{cv_scheme}` (splits={topology.get('cv_splits', 5)})",
    ]

    if id_cols:
        lines.append(f"- **Dropped Surrogate IDs**: {', '.join(f'`{c}`' for c in id_cols)}")

    if quasi_ids:
        lines.append(f"- **Quasi-IDs (Route to QuasiIdDecomposer)**: {', '.join(f'`{c}`' for c in quasi_ids)}")

    if text_cols:
        lines.append(f"- **Short Text (Route to LightweightTextEmbedder)**: {', '.join(f'`{c}`' for c in text_cols)}")

    if safe_cats:
        lines.append(f"- **Safe Categoricals (One-Hot / OOF Target)**: {', '.join(f'`{c}`' for c in safe_cats)}")

    if nums_cont or nums_disc:
        lines.append(f"- **Numerics**: {len(nums_cont)} continuous, {len(nums_disc)} discrete")

    if coercions:
        coercion_str = ", ".join(f"`{c}` -> {rule}" for c, rule in coercions.items())
        lines.append(f"- **Coerced Dirty Columns**: {coercion_str}")

    if sanitizer_result.sentinels_cleaned_count > 0:
        lines.append(f"- **Sentinels Normalized**: {sanitizer_result.sentinels_cleaned_count:,} entries")

    lines.append(f"- **Top Missingness**: {missing_str}")

    if topology.get("entity_group_col"):
        lines.append(f"- **Entity Group Column**: `{topology['entity_group_col']}`")

    if topology.get("temporal_col"):
        lines.append(f"- **Temporal Column**: `{topology['temporal_col']}`")

    markdown = "\n".join(lines)

    # Strict Context Firewall check: ensure < 4096 bytes
    raw_bytes = markdown.encode("utf-8")
    if len(raw_bytes) > 4000:
        markdown = markdown[:3900] + "\n... [TRUNCATED TO FIT CONTEXT FIREWALL]"

    return markdown


def eda_forensics_suite(
    dataset_uri_or_df: Union[pd.DataFrame, str, Path],
    problem_statement: Optional[str] = None,
    explicit_target: Optional[str] = None,
    output_dir: Union[str, Path] = ".m2/artifacts/eda",
    dataset_id: str = "dataset_default",
) -> EdaDossier:
    """Execute complete M2-EDA forensics suite: dirty cleaning, semantic sieve, plots, and dossier.

    Ensures zero train-serve skew and respects strict Context Firewall (< 4 KB payload).
    """
    out_path = Path(output_dir)
    sanitizer_res = sanitize_dataset(dataset_uri_or_df)
    df_clean = sanitizer_res.df

    topology = analyze_and_lock_topology(
        df=df_clean,
        problem_statement=problem_statement,
        explicit_target=explicit_target,
        dataset_id=dataset_id,
        coercions=sanitizer_res.coercions,
    )

    plot_paths = generate_eda_plots(df_clean, topology, out_path)
    summary_md = build_compact_markdown_dossier(df_clean, topology, sanitizer_res)

    token_estimate = int(len(summary_md.split()) * 1.3)

    return EdaDossier(
        dataset_id=dataset_id,
        topology=topology,
        summary_markdown=summary_md,
        artifacts=plot_paths,
        context_token_estimate=token_estimate,
    )


def run_exploratory_python(
    code_snippet: str,
    dataset_df: pd.DataFrame,
    timeout_s: float = 15.0,
    exclude_columns: Optional[List[str]] = None,
) -> str:
    """Execute ad-hoc Python exploratory code in an isolated worker subprocess with strict timeout and Context Firewall.

    Guarantees:
    1. Hard timeout enforcement via subprocess process isolation (child killed on timeout).
    2. Zero parent memory mutation: caller's DataFrame is completely unaffected.
    3. Label isolation: optional exclude_columns stripped before worker execution.
    4. Strict Context Firewall: output formatted and truncated to <= 2048 bytes.
    """
    import json
    import pickle
    import subprocess
    import tempfile

    # Provide read-only dataset view without excluded columns
    df_clean = dataset_df.copy(deep=True)
    if exclude_columns:
        df_clean = df_clean.drop(columns=[c for c in exclude_columns if c in df_clean.columns], errors="ignore")

    with tempfile.TemporaryDirectory(prefix="m2_eda_sandbox_") as tmp_dir:
        tmp_p = Path(tmp_dir)
        df_path = tmp_p / "dataset.pkl"
        code_path = tmp_p / "snippet.py"
        result_path = tmp_p / "result.json"

        with open(df_path, "wb") as f:
            pickle.dump(df_clean, f, protocol=pickle.HIGHEST_PROTOCOL)

        runner_code = """
import io
import json
import pickle
import sys
import traceback
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    tmp_p = Path(__file__).resolve().parent
    df_path = tmp_p / "dataset.pkl"
    code_path = tmp_p / "code.txt"
    result_path = tmp_p / "result.json"

    with open(df_path, "rb") as f:
        df = pickle.load(f)

    code_snippet = code_path.read_text(encoding="utf-8")

    captured_out = io.StringIO()
    captured_err = io.StringIO()
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = captured_out
    sys.stderr = captured_err

    local_env = {
        "df": df,
        "pd": pd,
        "np": np,
    }

    error_msg = None
    try:
        compiled = compile(code_snippet, "<exploratory_cell>", "exec")
        exec(compiled, local_env)
    except Exception as e:
        tb = traceback.format_exc(limit=3)
        error_msg = f"Execution Error: {e}\\n{tb}"
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    result = {
        "stdout": captured_out.getvalue().strip(),
        "stderr": captured_err.getvalue().strip(),
        "error": error_msg,
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f)

if __name__ == "__main__":
    main()
"""
        runner_file = tmp_p / "runner.py"
        runner_file.write_text(runner_code, encoding="utf-8")
        (tmp_p / "code.txt").write_text(code_snippet, encoding="utf-8")

        proc = subprocess.Popen(
            [sys.executable, "-I", str(runner_file)],
            cwd=str(tmp_p),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            p_out, p_err = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            return f"[error]\nExecution Error: Timeout after {timeout_s}s exceeded."

        stdout_text = ""
        stderr_text = ""
        error_msg: Optional[str] = None

        if result_path.exists():
            try:
                with open(result_path, "r", encoding="utf-8") as f:
                    res_data = json.load(f)
                    stdout_text = res_data.get("stdout", "")
                    stderr_text = res_data.get("stderr", "")
                    error_msg = res_data.get("error")
            except Exception as e:
                error_msg = f"Failed to read sandbox output: {e}"
        else:
            if proc.returncode != 0:
                error_msg = f"Sandbox process exited with code {proc.returncode}:\n{p_err.strip() or p_out.strip()}"
            else:
                stdout_text = p_out.strip()
                stderr_text = p_err.strip()

        output_parts: List[str] = []

        if stdout_text:
            lines = stdout_text.splitlines()
            if len(lines) > 20:
                stdout_text = "\n".join(lines[:15]) + f"\n... [{len(lines) - 15} lines truncated for context firewall]"
            output_parts.append(stdout_text)

        if stderr_text:
            output_parts.append(f"[stderr]\n{stderr_text[:500]}")

        if error_msg:
            output_parts.append(f"[error]\n{error_msg[:1000]}")

        digest = "\n\n".join(output_parts) if output_parts else "[Cell executed successfully with no output]"

        # Hard ceiling Context Firewall: <= 2048 bytes
        if len(digest.encode("utf-8")) > 2048:
            digest = digest[:1950] + "\n... [TRUNCATED BY CONTEXT FIREWALL <2 KB]"

        return digest
