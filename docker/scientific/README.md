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
