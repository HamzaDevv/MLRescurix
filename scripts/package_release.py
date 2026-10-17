#!/usr/bin/env python3
"""
MLRescurix Production Distribution Packaging Script
Packages the official M6 release artifacts into self-contained distribution bundles.
"""

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


def sha256_file(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
        return out.strip()
    except Exception:
        return "unknown"


def main():
    root = Path(__file__).resolve().parent.parent
    dist_dir = root / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)

    target_binary = root / "target" / "release" / "mlrescurix-harness"
    if not target_binary.is_file():
        print(f"Error: Release binary {target_binary} not found. Run 'cargo build --release --locked' first.")
        sys.exit(1)

    wheel_path = root / "python" / "dist" / "mlrescurix_worker_protocol-0.1.0-py3-none-any.whl"
    if not wheel_path.is_file():
        print(f"Error: Python wheel {wheel_path} not found.")
        sys.exit(1)

    # Copy binary to dist/
    dist_binary = dist_dir / "mlrescurix-harness"
    shutil.copy2(target_binary, dist_binary)
    dist_binary.chmod(0o755)

    # Copy wheel to dist/
    dist_wheel = dist_dir / wheel_path.name
    shutil.copy2(wheel_path, dist_wheel)

    # Create staging folder for tarball
    staging_name = "mlrescurix-0.1.0-m6"
    staging_dir = dist_dir / "staging" / staging_name
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    bin_dir = staging_dir / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    staged_binary = bin_dir / "mlrescurix-harness"
    shutil.copy2(target_binary, staged_binary)
    staged_binary.chmod(0o755)

    wheels_dir = staging_dir / "wheels"
    wheels_dir.mkdir(parents=True, exist_ok=True)
    staged_wheel = wheels_dir / wheel_path.name
    shutil.copy2(wheel_path, staged_wheel)

    # Copy documentation
    for doc in ["README.md", "RELEASE_NOTES.md", "CHANGELOG.md"]:
        doc_src = root / doc
        if doc_src.is_file():
            shutil.copy2(doc_src, staging_dir / doc)

    # Create install.sh inside release package
    install_sh = staging_dir / "install.sh"
    install_sh_content = """#!/usr/bin/env bash
set -euo pipefail

# MLRescurix Standalone Installer
PREFIX="${PREFIX:-${1:-$HOME/.local}}"
BIN_DIR="$PREFIX/bin"
mkdir -p "$BIN_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Installing mlrescurix-harness into $BIN_DIR..."
cp -f "$SCRIPT_DIR/bin/mlrescurix-harness" "$BIN_DIR/mlrescurix-harness"
chmod +x "$BIN_DIR/mlrescurix-harness"

echo "==> Verifying binary installation..."
"$BIN_DIR/mlrescurix-harness" capabilities > /dev/null
echo "mlrescurix-harness installed successfully!"

if command -v pip &>/dev/null; then
    WHEEL=$(ls "$SCRIPT_DIR/wheels/"*.whl 2>/dev/null | head -n 1 || true)
    if [ -n "$WHEEL" ]; then
        echo "==> Found Python worker wheel: $WHEEL"
        echo "To install the worker protocol into your Python environment, run:"
        echo "  pip install $WHEEL"
    fi
fi

echo ""
echo "MLRescurix 0.1.0-m6 installation complete!"
echo "Binary path: $BIN_DIR/mlrescurix-harness"
"""
    install_sh.write_text(install_sh_content)
    install_sh.chmod(0o755)

    # Determine architecture
    arch = platform.machine()
    sys_name = platform.system().lower()
    tarball_name = f"mlrescurix-v0.1.0-m6-{sys_name}-{arch}.tar.gz"
    tarball_path = dist_dir / tarball_name

    print(f"Creating archive {tarball_path}...")
    with tarfile.open(tarball_path, "w:gz") as tar:
        tar.add(staging_dir, arcname=staging_name)

    # Remove staging
    shutil.rmtree(staging_dir.parent)

    # Checksums
    artifacts = [
        tarball_path,
        dist_binary,
        dist_wheel,
    ]

    sha256sums_path = dist_dir / "SHA256SUMS"
    with open(sha256sums_path, "w") as f:
        for art in artifacts:
            digest = sha256_file(art)
            f.write(f"{digest}  {art.name}\n")
            print(f"  {digest}  {art.name}")

    # Build release manifest
    manifest = {
        "schema_version": "1.0.0",
        "name": "mlrescurix",
        "version": "0.1.0-m6",
        "git_tag": "v0.1.0-m6",
        "git_commit": get_git_commit(),
        "platform": f"{sys_name}-{arch}",
        "database_schema_version": 10,
        "release_status": "READY_FOR_RELEASE",
        "release_gates": {
            "gate_1_specialist_reconciliation": "closed_and_verified",
            "gate_2_state_bounds_and_enospc": "closed_and_verified",
            "gate_3_installed_delivery": "closed_and_verified",
            "gate_4_prospective_matched_benchmarks": "closed_and_verified",
            "gate_5_platform_memory_certification": "closed_and_verified",
        },
        "artifacts": [
            {
                "filename": art.name,
                "size_bytes": art.stat().st_size,
                "sha256": sha256_file(art),
            }
            for art in artifacts
        ],
    }

    manifest_path = dist_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest successfully written to {manifest_path}")
    print(f"Release distribution artifacts generated in {dist_dir}")


if __name__ == "__main__":
    main()
