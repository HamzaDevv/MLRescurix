#!/usr/bin/env bash
# Local/offline review checks. No provider calls, benchmarks or publication.
set -euo pipefail
cd "$(dirname "$0")/.."
review_root="$PWD"
review_image="$(cat docker/scientific/image.ref)"
if [[ ! "$review_image" =~ ^[^[:space:]]+@sha256:[0-9a-f]{64}$ ]]; then
  echo 'A digest-pinned scientific image is required.' >&2
  exit 1
fi
export MLRESCURIX_BUNDLE_IMAGE="$review_image"
review_evidence="$review_root/docs/evidence/m6"
mkdir -p "$review_evidence"
review_docker=(docker run --rm --pull=never --network none --read-only
  --user 1000:1000 --cap-drop ALL --security-opt no-new-privileges:true
  --memory 1g --cpus 1 --pids-limit 64
  --tmpfs /tmp:rw,noexec,nosuid,size=64m,uid=1000,gid=1000
  -v "$review_root/python:/python:ro" -v "$review_root/contracts:/contracts:ro"
  -e PYTHONPATH=/python/src -e MLRESCURIX_TEST_BOUNDED_TMPFS=64m)

"${review_docker[@]}" -i -v "$review_root/docker/scientific/requirements.lock:/review.lock:ro" \
  "$review_image" python - > "$review_evidence/review-image-dependencies.json" <<'PY'
import hashlib, importlib.metadata as metadata, json, pathlib, re, subprocess, sys, tomllib
lock = pathlib.Path('/opt/mlrescurix/requirements.lock').read_bytes()
assert lock == pathlib.Path('/review.lock').read_bytes(), 'image/repository lock mismatch'
versions = dict(re.findall(r'^([A-Za-z0-9_.-]+)==([^\s\\]+)', lock.decode(), re.M))
assert versions
for name, version in versions.items():
    assert metadata.version(name) == version, (name, version, metadata.version(name))
requirements = tomllib.loads(pathlib.Path('/python/pyproject.toml').read_text())['project']['dependencies']
for requirement in requirements:
    name, version = requirement.split('==')
    assert metadata.version(name) == version
check = subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True, capture_output=True, text=True)
print(json.dumps(dict(python=sys.version, lock_sha256=hashlib.sha256(lock).hexdigest(),
    versions=versions, worker_requirements=requirements, pip_check=check.stdout.strip(),
    repository_lock_matches_image=True), indent=2))
PY

cargo test --locked --lib --test agent_m4 --test foundation --test recovery_cycles --test release_m6 \
  > "$review_evidence/review-rust-final.log" 2>&1
cargo test --locked --test bundle_scientific_docker -- --ignored --nocapture \
  > "$review_evidence/review-bundle-docker-final.log" 2>&1
cargo test --locked --test pipeline_docker -- --ignored --nocapture \
  > "$review_evidence/review-finalization-docker-final.log" 2>&1

"${review_docker[@]}" -i "$review_image" python - \
  > "$review_evidence/review-python-docker-scoped.log" 2>&1 <<'PY'
import sys, unittest
suite = unittest.defaultTestLoader.discover('/python/tests')
def flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item
excluded = 'test_search.DashboardSmokeTests.test_dashboard_login_and_snapshot_smoke'
print('Explicit exclusion: ' + excluded + ' requires a native Rust binary and is outside this Linux scientific-image test scope', flush=True)
selected = unittest.TestSuite(test for test in flatten(suite) if test.id() != excluded)
result = unittest.TextTestRunner(verbosity=2).run(selected)
sys.exit(not result.wasSuccessful())
PY
cargo run --locked --quiet --bin mlrescurix-harness -- capabilities \
  > "$review_evidence/review-capabilities.json"
echo "Review checks passed; evidence is in $review_evidence. See docs/STATUS_AND_ROADMAP.md for current status and remaining gates."
