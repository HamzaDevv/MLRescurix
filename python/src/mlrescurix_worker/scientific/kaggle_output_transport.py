"""Bounded Kaggle output download, run as a separate credential-scoped process.

Remote names and URLs are untrusted. Only flat, regular files and known HTTPS
storage hosts are accepted. The caller validates the manifest before publication.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit

MAX_BYTES = 100_000_000
MAX_FILES = 1000
MAX_PAGES = 100


def fetch_outputs(client, request_type, requests_module, owner, slug, destination):
    seen = set()
    token = ""
    total = 0
    for _ in range(MAX_PAGES):
        request = request_type()
        request.user_name = owner
        request.kernel_slug = slug
        request.page_size = 100
        if token:
            request.page_token = token
        result = client.kernels.kernels_api_client.list_kernel_session_output(request)
        for item in result.files or []:
            name = item.file_name
            if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,199}", name)
                    or name in seen or len(seen) >= MAX_FILES):
                raise ValueError("invalid or duplicate remote filename")
            seen.add(name)
            url = urlsplit(item.url)
            host = (url.hostname or "").lower()
            if (url.scheme != "https" or url.username is not None or url.password is not None
                    or url.port not in (None, 443) or
                    not (host == "storage.googleapis.com" or host.endswith(".storage.googleapis.com"))):
                raise ValueError("remote output host is not permitted")
            with requests_module.get(item.url, stream=True, allow_redirects=False, timeout=(10, 30),
                                     headers={"Accept-Encoding": "identity"}) as response:
                response.raise_for_status()
                if 300 <= response.status_code < 400:
                    raise ValueError("remote output redirect refused")
                if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                    raise ValueError("compressed remote output refused")
                declared = response.headers.get("Content-Length")
                if declared is not None and (not declared.isdigit() or total + int(declared) > MAX_BYTES):
                    raise ValueError("remote output byte limit exceeded")
                fd = os.open(Path(destination) / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
                with os.fdopen(fd, "wb") as out:
                    for chunk in response.iter_content(65536):
                        total += len(chunk)
                        if total > MAX_BYTES:
                            raise ValueError("remote output byte limit exceeded")
                        out.write(chunk)
                    out.flush()
                    os.fsync(out.fileno())
        next_token = result.next_page_token or ""
        if not next_token:
            return
        if next_token == token:
            raise ValueError("remote output pagination did not advance")
        token = next_token
    raise ValueError("remote output page limit exceeded")


def main():
    import kaggle
    import requests
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    with kaggle.api.build_kaggle_client() as client:
        fetch_outputs(client, ApiListKernelSessionOutputRequest, requests, sys.argv[1], sys.argv[2], sys.argv[3])


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # SDK/network exceptions may contain credentials or signed URLs.
        print("Kaggle output transport failed", file=sys.stderr)
        sys.exit(1)
