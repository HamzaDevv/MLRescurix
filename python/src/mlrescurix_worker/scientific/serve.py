"""Minimal standalone HTTP serving interface for MLRescurix candidate bundles.

Provides:
- GET  /health: returns liveness and bundle metadata.
- POST /predict: executes schema-validated predictions using locked inference-only dependencies.

Uses only Python standard library http.server; zero external server frameworks required.
Development harness, git, compiler, and LLM agent dependencies are completely excluded.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any


class BundleServingHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler for standalone bundle prediction and health monitoring."""

    server_start_time = time.time()
    bundle_path: Path = Path(".")
    candidate_id: str = "unknown"
    bundle_format: str = "mlrescurix.bundle.v2"
    inference_module: Any = None

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress standard access logs to avoid terminal bloat; errors still print."""
        pass

    def _send_json(self, status: int, data: dict[str, Any]) -> None:
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        """Handle health probe requests."""
        if self.path in ("/health", "/healthz"):
            uptime = time.time() - self.server_start_time
            self._send_json(HTTPStatus.OK, {
                "status": "healthy",
                "candidate_id": self.candidate_id,
                "format": self.bundle_format,
                "uptime_s": round(uptime, 2),
            })
        else:
            self._send_json(HTTPStatus.NOT_FOUND, {
                "status": "not_found",
                "message": f"Path '{self.path}' not found",
            })

    def do_POST(self) -> None:
        """Handle raw-input prediction requests."""
        if self.path in ("/predict", "/v1/predict"):
            content_len = int(self.headers.get("Content-Length", 0))
            if content_len == 0 or content_len > 16 * 1024 * 1024:
                self._send_json(HTTPStatus.BAD_REQUEST, {
                    "status": "error",
                    "message": "Empty body or body exceeded 16 MiB payload ceiling",
                })
                return

            body = self.rfile.read(content_len)
            try:
                data = json.loads(body.decode("utf-8"))
            except Exception as e:
                self._send_json(HTTPStatus.BAD_REQUEST, {
                    "status": "error",
                    "message": f"Malformed JSON payload: {e}",
                })
                return

            # Extract list of input records
            records: list[dict[str, Any]]
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict):
                records = data.get("inputs") or data.get("records")
                if records is None:
                    self._send_json(HTTPStatus.BAD_REQUEST, {
                        "status": "error",
                        "message": "Request JSON object must contain 'inputs' or 'records' list",
                    })
                    return
            else:
                self._send_json(HTTPStatus.BAD_REQUEST, {
                    "status": "error",
                    "message": "Request body must be a JSON array of row objects or an object with 'inputs'",
                })
                return

            try:
                res = self.inference_module.predict(records)
                self._send_json(HTTPStatus.OK, {
                    "status": "ok",
                    "result": res,
                })
            except (ValueError, KeyError) as e:
                self._send_json(HTTPStatus.BAD_REQUEST, {
                    "status": "error",
                    "error_type": type(e).__name__,
                    "message": str(e),
                })
            except Exception as e:
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {
                    "status": "error",
                    "error_type": type(e).__name__,
                    "message": f"Inference execution error: {e}",
                })
        else:
            self._send_json(HTTPStatus.NOT_FOUND, {
                "status": "not_found",
                "message": f"Path '{self.path}' not found",
            })

    def do_PUT(self) -> None:
        self._send_json(HTTPStatus.METHOD_NOT_ALLOWED, {"status": "method_not_allowed"})

    def do_DELETE(self) -> None:
        self._send_json(HTTPStatus.METHOD_NOT_ALLOWED, {"status": "method_not_allowed"})


def create_serving_server(bundle_dir: str | Path, host: str = "127.0.0.1", port: int = 8080) -> HTTPServer:
    """Instantiate standalone serving server for candidate bundle."""
    bundle_path = Path(bundle_dir).resolve()
    if not bundle_path.is_dir():
        raise FileNotFoundError(f"Bundle directory does not exist: {bundle_path}")

    # Load manifest if available
    cand_id = "unknown"
    bundle_fmt = "mlrescurix.bundle.v2"
    manifest_p = bundle_path / "manifest.json"
    if manifest_p.is_file():
        try:
            m = json.loads(manifest_p.read_text(encoding="utf-8"))
            cand_id = m.get("candidate_id", cand_id)
            bundle_fmt = m.get("format", bundle_fmt)
        except Exception:
            pass

    # Import bundle inference module
    if str(bundle_path) not in sys.path:
        sys.path.insert(0, str(bundle_path))

    import importlib
    inf = importlib.import_module("inference")
    if not hasattr(inf, "predict"):
        raise AttributeError("bundle 'inference' module missing required 'predict(records)' entrypoint")

    # Configure handler class
    BundleServingHandler.bundle_path = bundle_path
    BundleServingHandler.candidate_id = cand_id
    BundleServingHandler.bundle_format = bundle_fmt
    BundleServingHandler.inference_module = inf
    BundleServingHandler.server_start_time = time.time()

    server = HTTPServer((host, port), BundleServingHandler)
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone MLRescurix Model Serving Interface")
    parser.add_argument("--host", default="127.0.0.1", help="Binding host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Binding port (default: 8080)")
    parser.add_argument("--bundle-dir", default=".", help="Path to bundle directory (default: current directory)")
    args = parser.parse_args()

    server = create_serving_server(args.bundle_dir, host=args.host, port=args.port)
    print(f"MLRescurix Standalone Serving Server listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server gracefully...")
        server.server_close()


if __name__ == "__main__":
    main()
