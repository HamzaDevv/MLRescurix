"""Validator for ProgramManifest, FitScopeBinding, and transitive provenance graphs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import jsonschema

def _find_contracts_dir() -> Path:
    current = Path(__file__).resolve().parent
    for _ in range(8):
        candidate = current / "contracts" / "v3"
        if candidate.is_dir():
            return candidate
        current = current.parent
    return Path(__file__).resolve().parents[5] / "contracts" / "v3"


CONTRACTS_V3 = _find_contracts_dir()


def _load_schema(schema_name: str) -> Dict[str, Any]:
    schema_path = CONTRACTS_V3 / schema_name
    if not schema_path.is_file():
        raise FileNotFoundError(f"Schema not found: {schema_path}")
    return json.loads(schema_path.read_text(encoding="utf-8"))


PROGRAM_SCHEMA = _load_schema("program.schema.json")
FIT_SCOPE_SCHEMA = _load_schema("fit_scope.schema.json")


def validate_program_manifest(manifest: Dict[str, Any]) -> bool:
    """Validate a ProgramManifest JSON dictionary against schema and semantic constraints.

    Raises jsonschema.ValidationError or ValueError on invalid input.
    """
    jsonschema.validate(instance=manifest, schema=PROGRAM_SCHEMA)
    ops = manifest.get("operations", {})
    if not ops:
        raise ValueError("Program manifest operations dictionary cannot be empty")
    for op_name, op in ops.items():
        if not op_name.strip():
            raise ValueError("Operation name cannot be empty or whitespace")
        argv = op.get("argv", [])
        if not argv:
            raise ValueError(f"Operation '{op_name}' argv cannot be empty")
        for arg in argv:
            if not isinstance(arg, str) or not arg.strip():
                raise ValueError(f"Operation '{op_name}' contains empty or non-string argument")
    return True


def validate_fit_scope_binding(binding: Dict[str, Any]) -> bool:
    """Validate a FitScopeBinding JSON dictionary against schema and semantic constraints.

    Raises jsonschema.ValidationError or ValueError on invalid input.
    """
    jsonschema.validate(instance=binding, schema=FIT_SCOPE_SCHEMA)
    kind = binding.get("fit_scope_kind")
    split_id = binding.get("split_id", "")
    if kind not in ("stateless", "fixed_pretrained") and (not split_id or split_id.lower() == "none"):
        raise ValueError(f"Fitted scope kind '{kind}' requires a valid non-empty split_id")
    return True


def check_fit_scope_compatibility(
    producer_binding: Dict[str, Any],
    consumer_binding: Dict[str, Any],
    evaluated_label_hash: Optional[str] = None,
    evaluated_label_rows: Optional[Set[int]] = None,
    producer_label_rows: Optional[Set[int]] = None,
    consumer_exclusion_rows: Optional[Set[int]] = None,
    producer_feature_rows: Optional[Set[int]] = None,
) -> Tuple[bool, str]:
    """Verify that a consumer can safely ingest state or features produced by producer.

    Enforces:
    - Protocol identity: producer and consumer must be bound to the exact same protocol_sha256.
    - Stateless and fixed_pretrained producers can be consumed by any scope, but cannot have fitted label rows.
    - Outer/inner fold consumers cannot ingest state fitted on different folds (cross-fold leakage).
    - Fold-level consumers cannot ingest full_development_train fitted state.
    - Producer must not be fitted on the evaluation holdout labels (both hash-based and row-set intersection checks).
    - Exclusion constraints must not be violated (both hash-based and row-set intersection checks).
    """
    validate_fit_scope_binding(producer_binding)
    validate_fit_scope_binding(consumer_binding)

    # 1. Exact protocol identity binding
    prod_proto = producer_binding.get("protocol_sha256")
    cons_proto = consumer_binding.get("protocol_sha256")
    if prod_proto != cons_proto:
        return (
            False,
            f"protocol mismatch: producer protocol '{prod_proto}' does not match consumer protocol '{cons_proto}'",
        )

    prod_kind = producer_binding["fit_scope_kind"]
    cons_kind = consumer_binding["fit_scope_kind"]
    prod_split = producer_binding.get("split_id")
    cons_split = consumer_binding.get("split_id")

    # 2. Row-set level label leakage check (partial overlap detection: producer_label_rows & evaluated_label_rows)
    if producer_label_rows is not None and evaluated_label_rows is not None:
        overlap = set(producer_label_rows).intersection(set(evaluated_label_rows))
        if overlap:
            return (
                False,
                f"label leakage: producer fitted labels overlap {len(overlap)} evaluated row(s): {sorted(list(overlap))[:10]}",
            )

    # 3. Row-set level feature exclusion check (partial overlap detection: producer_feature_rows & consumer_exclusion_rows)
    if producer_feature_rows is not None and consumer_exclusion_rows is not None:
        overlap = set(producer_feature_rows).intersection(set(consumer_exclusion_rows))
        if overlap:
            return (
                False,
                f"exclusion violation: producer features overlap {len(overlap)} excluded row(s): {sorted(list(overlap))[:10]}",
            )

    # 4. Stateless and pretrained models carry no fold-specific fitted state
    if prod_kind in ("stateless", "fixed_pretrained"):
        if producer_label_rows is not None and len(producer_label_rows) > 0:
            return (
                False,
                f"producer scope violation: '{prod_kind}' producer cannot have fitted label rows (has {len(producer_label_rows)} fitted rows)",
            )
        return True, f"compatible: producer is {prod_kind}"

    # 5. Evaluated label hash check
    if evaluated_label_hash is not None:
        if producer_binding.get("permitted_label_row_ids_hash") == evaluated_label_hash:
            return False, "label leakage: producer was fitted on evaluation holdout labels"

    # 6. Cross-fold split mismatch check
    if prod_split != cons_split:
        return (
            False,
            f"cross-fold leakage: consumer split '{cons_split}' cannot consume state from producer split '{prod_split}'",
        )

    # 7. Full development train cannot be consumed by individual fold models
    if prod_kind == "full_development_train" and cons_kind in (
        "outer_fold_train",
        "inner_fold_train",
        "meta_fold_train",
    ):
        return (
            False,
            "scope violation: fold-level consumer cannot consume full_development_train state",
        )

    # 8. Exclusion check: if consumer explicitly excludes producer's feature set
    consumer_exclusion = consumer_binding.get("exclusion_row_ids_hash")
    if consumer_exclusion and consumer_exclusion == producer_binding.get("permitted_feature_row_ids_hash"):
        return (
            False,
            "exclusion violation: producer features match consumer's exclusion row hash",
        )

    return True, "compatible"


def validate_transitive_provenance(operations_graph: Dict[str, Dict[str, Any]]) -> Tuple[bool, str]:
    """Validate acyclicity and transitive fit-scope compatibility for an operations graph.

    Graph structure:
    {
        "op_name": {
            "fit_scope": {...FitScopeBinding...},
            "dependencies": ["dep_name_1", ...]
        }
    }
    """
    if not operations_graph:
        return False, "operations graph is empty"

    # 1. Verify existence and validate bindings
    for name, op in operations_graph.items():
        if "fit_scope" not in op:
            return False, f"operation '{name}' is missing fit_scope"
        try:
            validate_fit_scope_binding(op["fit_scope"])
        except Exception as e:
            return False, f"operation '{name}' has invalid fit_scope: {e}"
        for dep in op.get("dependencies", []):
            if dep not in operations_graph:
                return False, f"operation '{name}' references unknown dependency '{dep}'"

    # 2. Cycle detection (topological sort / DFS)
    visited: Dict[str, int] = {}  # 0: unvisited, 1: visiting, 2: visited

    def has_cycle(node: str) -> bool:
        visited[node] = 1
        for dep in operations_graph[node].get("dependencies", []):
            state = visited.get(dep, 0)
            if state == 1:
                return True
            if state == 0 and has_cycle(dep):
                return True
        visited[node] = 2
        return False

    for node in operations_graph:
        if visited.get(node, 0) == 0:
            if has_cycle(node):
                return False, "cyclical dependency detected in operations graph"

    # 3. Transitive dependency collection and compatibility check
    def get_all_ancestors(node: str, memo: Dict[str, Set[str]]) -> Set[str]:
        if node in memo:
            return memo[node]
        ancestors: Set[str] = set()
        for dep in operations_graph[node].get("dependencies", []):
            ancestors.add(dep)
            ancestors.update(get_all_ancestors(dep, memo))
        memo[node] = ancestors
        return ancestors

    memo: Dict[str, Set[str]] = {}
    for node, op in operations_graph.items():
        ancestors = get_all_ancestors(node, memo)
        consumer_scope = op["fit_scope"]
        cons_eval_rows = op.get("evaluated_label_rows") or op.get("evaluation_rows")
        cons_excl_rows = op.get("consumer_exclusion_rows") or op.get("exclusion_rows")
        cons_eval_hash = op.get("evaluated_label_hash")

        for ancestor_name in ancestors:
            ancestor_op = operations_graph[ancestor_name]
            producer_scope = ancestor_op["fit_scope"]
            prod_label_rows = ancestor_op.get("producer_label_rows") or ancestor_op.get("fitted_label_rows")
            prod_feat_rows = ancestor_op.get("producer_feature_rows") or ancestor_op.get("fitted_feature_rows")

            ok, reason = check_fit_scope_compatibility(
                producer_scope,
                consumer_scope,
                evaluated_label_hash=cons_eval_hash,
                producer_label_rows=prod_label_rows,
                evaluated_label_rows=cons_eval_rows,
                producer_feature_rows=prod_feat_rows,
                consumer_exclusion_rows=cons_excl_rows,
            )
            if not ok:
                return False, f"transitive fit provenance violation between '{ancestor_name}' and '{node}': {reason}"

    return True, "valid transitive provenance"
