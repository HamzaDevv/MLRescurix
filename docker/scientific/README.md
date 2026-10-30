# Pinned scientific worker image

This recipe builds the M6 worker for Linux arm64 from the digest-pinned CPython
base. All runtime and build dependencies are installed from the checked-in
offline wheelhouse using `requirements.lock` with hash enforcement. The image
build runs `pip check`; `image-dependencies.json` records the versions checked
inside the resulting image. The wheelhouse is local build input and is excluded
from Git; recreate it from the matching, hash-verified CPython 3.13 arm64 wheels
before rebuilding.

From the repository root, populate `docker/scientific/wheels/` with the locked
wheel files and the source package as `docker/scientific/worker/src` plus its
`pyproject.toml` at `docker/scientific/worker/`. Then build locally without
pushing:

```sh
docker buildx build --platform linux/arm64 --output type=image,name=mlrescurix-scientific:latest,push=false docker/scientific
docker image inspect mlrescurix-scientific:latest --format '{{index .RepoDigests 0}}'
```

Pin the returned repository digest in `image.ref` and set
`MLRESCURIX_BUNDLE_IMAGE` to that exact value. The Rust verifier uses
`--pull=never`; it will refuse if the pinned image is unavailable locally.

The verified image for this completion pass is
`mlrescurix-scientific@sha256:d094b6becbbd1a1d86b944fd39370bfe26bb96f6290b429c0a51c512faf5f1bd`.
Build provenance and in-image dependency evidence are in `docs/evidence/m6/`.
The tag is local and was not pushed.

## 2026-10-23 dependency blocker

The cached historical image above passes isolated stdlib cgroup allocation/OOM
fixtures but cannot import the current scientific package: `jsonschema` is absent.
The worker metadata now declares `jsonschema==4.26.0` and `psutil==6.1.1`, matching
the versions used by the host offline regressions. The existing Linux lock and
local wheelhouse do not yet include these dependencies and their transitive wheels.
A fresh build must fail `pip check` rather than treating this old image as a complete
current runtime. Restore authoritative hash-verified Linux CPython 3.13 arm64 wheels,
extend the hash lock, rebuild, and record a new image digest before claiming full
scientific-container execution. Do not change `image.ref` to an unbuilt image or
remove imports/dependency checks to make the bounded-tmpfs test pass. The blocked
attempt is recorded in `docs/evidence/v2/autonomy/20261023-prerequisite-repairs/`.

## 2026-10-30 synthetic delivery test

The lock now includes the Linux aarch64 `safetensors==0.6.2` wheel and its SHA-256. A local test-only rebuild from this lock and the locally available worker build context produced `mlrescurix-scientific-test@sha256:7458197a2dbf0535b7cc2a759ffcbd6be4c8bc6a7534548b78f764237bfcc4b3`. With that digest, the synthetic `pipeline_docker` ignored test passed after inference package imports were made lazy. The existing `image.ref` is not a newly verified release image and may not match the updated lock. Rebuild from the current checked-in worker source and verify provenance and full delivery gates before changing the release reference.
