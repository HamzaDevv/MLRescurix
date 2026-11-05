"""Tests for holdout evaluation, release provenance, and bundle anti-tampering."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_holdout_evaluation_and_bundle_provenance(tmp_path):
    bundle_dir = None
    for cand in sorted(REPO_ROOT.glob(".mlrescurix-state/amazon-2025/live-*/bundle")):
        if cand.is_dir() and (cand / "manifest.json").is_file():
            bundle_dir = cand
            break

    if bundle_dir is None:
        pytest.skip("No candidate bundle found in workspace for holdout evaluation test")

    eval_script = REPO_ROOT / "scripts" / "evaluate_holdout.py"
    assert eval_script.is_file(), "evaluate_holdout.py must exist"

    # Use synthetic isolated holdout data to test evaluation machinery without touching exposed data
    synthetic_holdout = tmp_path / "synthetic_holdout.csv"
    synthetic_rows = ["sample_id,catalog_content,price"]
    for i in range(15):
        synthetic_rows.append(f"{i},sample product description {i} with high quality features,{10.0 + i * 2.5}")
    synthetic_holdout.write_text("\n".join(synthetic_rows) + "\n", encoding="utf-8")

    out_json = tmp_path / "holdout_eval_test.json"

    # Run evaluate_holdout.py in a clean subprocess with -B
    proc = subprocess.run(
        [
            sys.executable,
            "-B",
            str(eval_script),
            "--bundle",
            str(bundle_dir),
            "--holdout-csv",
            str(synthetic_holdout),
            "--output",
            str(out_json),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"evaluate_holdout.py failed:\nSTDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
    assert out_json.is_file(), "Output JSON must be created"

    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["status"] == "success"
    assert data["manifest_integrity_verified"] is True
    assert data["subprocess_replay_verified"] is True
    assert data["harness_bundle_verification"] == "integrity_verified"
    assert data["holdout_rows"] == 15
    assert data["holdout_smape"] is not None and data["holdout_smape"] >= 0
    assert data["generalization_gap"] is not None

    # Verify that the bundle was NOT polluted with bytecode (.pyc or __pycache__)
    pycache_dirs = list(bundle_dir.rglob("__pycache__"))
    pyc_files = list(bundle_dir.rglob("*.pyc"))
    assert len(pycache_dirs) == 0, f"Bytecode cache directories found in bundle: {pycache_dirs}"
    assert len(pyc_files) == 0, f"Bytecode files found in bundle: {pyc_files}"


def test_holdout_evaluation_rejects_tampered_bundle(tmp_path):
    """Confirm evaluate_holdout.py fails closed when bundle files do not match manifest."""
    bundle_dir = None
    for cand in sorted(REPO_ROOT.glob(".mlrescurix-state/amazon-2025/live-*/bundle")):
        if cand.is_dir() and (cand / "manifest.json").is_file():
            bundle_dir = cand
            break
    if bundle_dir is None:
        cand_s1 = REPO_ROOT / "docs" / "evidence" / "refactor" / "S1" / "cli_state" / "delivery_bundle_cand_proj_s1_evidence_demo_309ff456"
        if cand_s1.is_dir() and (cand_s1 / "manifest.json").is_file():
            bundle_dir = cand_s1

    holdout_csv = REPO_ROOT / ".mlrescurix-state" / "amazon-2025" / "holdout_lockbox_15k.csv"
    if not holdout_csv.is_file():
        holdout_csv = tmp_path / "dummy_holdout.csv"
        holdout_csv.write_text("sample_id,catalog_content,price\n1,test,10.0\n", encoding="utf-8")

    if bundle_dir is None:
        pytest.skip("No candidate bundle found in workspace for anti-tampering test")

    # Create a clone of the bundle in tmp_path
    import shutil
    tampered_bundle = tmp_path / "tampered_bundle"
    shutil.copytree(bundle_dir, tampered_bundle)

    # Tamper with scorecard.json
    sc_path = tampered_bundle / "scorecard.json"
    sc = json.loads(sc_path.read_text(encoding="utf-8"))
    sc["primary_metric"] = 1.0  # Falsified score
    sc_path.write_text(json.dumps(sc), encoding="utf-8")

    eval_script = REPO_ROOT / "scripts" / "evaluate_holdout.py"
    out_json = tmp_path / "out_tampered.json"

    proc = subprocess.run(
        [
            sys.executable,
            "-B",
            str(eval_script),
            "--bundle",
            str(tampered_bundle),
            "--holdout-csv",
            str(holdout_csv),
            "--output",
            str(out_json),
        ],
        capture_output=True,
        text=True,
    )
    # Must fail closed because scorecard.json SHA-256 no longer matches manifest.json
    assert proc.returncode != 0
    assert "hash mismatch" in proc.stderr or "hash mismatch" in proc.stdout
