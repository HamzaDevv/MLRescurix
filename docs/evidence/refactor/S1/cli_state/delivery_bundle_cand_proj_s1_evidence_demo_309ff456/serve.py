import sys
import os
import json
import argparse
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

bundle_path = Path(__file__).resolve().parent

class InferenceHandler(BaseHTTPRequestHandler):
    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "serving_status": "healthy"})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/predict":
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length).decode("utf-8")
                inputs = json.loads(body)

                sys.path.insert(0, str(bundle_path))
                import inference
                preds = inference.predict(inputs, bundle_dir=bundle_path)
                self._send_json(200, {"status": "ok", "predictions": preds})
            except Exception as e:
                self._send_json(400, {"status": "error", "error": str(e)})
        else:
            self._send_json(404, {"error": "not found"})

    def log_message(self, format, *args):
        pass

class InferenceServer(HTTPServer):
    allow_reuse_address = True

def main():
    global bundle_path
    parser = argparse.ArgumentParser(description="Standalone Model Serving Recipe")
    parser.add_argument("--port", type=int, required=True, help="Port to bind")
    parser.add_argument("--bundle-dir", default=None, help="Bundle directory")
    args = parser.parse_args()

    if args.bundle_dir:
        bundle_path = Path(args.bundle_dir).resolve()

    server = InferenceServer(("127.0.0.1", args.port), InferenceHandler)
    server.serve_forever()

if __name__ == "__main__":
    main()
