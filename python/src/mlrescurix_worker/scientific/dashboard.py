"""Loopback dashboard over committed Rust receipts/events and M2 evidence files."""
from __future__ import annotations

import argparse
import http.cookies
import json
import os
import re
import secrets
import shutil
import sqlite3
import sys
import subprocess
import tempfile
import time
import zipfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

MAX_BODY = 65_536
ID = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")

PAGE = r'''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>MLRescurix local</title><style>
body{font:15px system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#17202a;background:#f5f7fa}
section{background:white;border:1px solid #d9e1ea;border-radius:10px;padding:1rem;margin:1rem 0}button,input{font:inherit;padding:.5rem;margin:.2rem}button{cursor:pointer}
pre{white-space:pre-wrap;overflow-wrap:anywhere}.muted{color:#657487}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:.5rem;border-bottom:1px solid #e2e8f0}
</style><h1>MLRescurix</h1><p class="muted">Local project state is read from committed Rust snapshots and M2 workflow evidence.</p>
<section id="login"><h2>Unlock local session</h2><input id="token" type="password" autocomplete="current-password" placeholder="Session token"><button onclick="login()">Connect</button><span id="loginerr"></span></section>
<section id="app" hidden><label>Project ID <input id="project" value="project-1"></label><button onclick="refresh()">Refresh</button>
<button onclick="command('paused')">Pause</button><button onclick="command('created')">Resume</button><button onclick="command('cancelled')">Cancel project</button>
<a id="download" hidden>Download verified bundle</a>
<p id="state" class="muted"></p><h2>Committed jobs and candidates</h2><pre id="jobs">Waiting for project state</pre><h2>Events</h2><pre id="events"></pre><h2>Recent command receipt</h2><pre id="receipt">No command sent</pre></section>
<script>
const $=x=>document.getElementById(x);let source=null,revision=null,lastCursor='0';
async function login(){let r=await fetch('/session',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({token:$('token').value})});if(r.ok){$('login').hidden=true;$('app').hidden=false;refresh();connectEvents()}else $('loginerr').textContent='Token rejected'}
async function refresh(){let p=$('project').value.trim();if(!/^[A-Za-z0-9_.:-]{1,128}$/.test(p))return;let r=await fetch('/v1/projects/'+encodeURIComponent(p)+'/snapshot');if(!r.ok){$('state').textContent='Snapshot unavailable ('+r.status+')';return}let s=await r.json();revision=s.revision;$('state').textContent='State: '+s.state+' · revision '+s.revision+' · event cursor '+s.event_cursor;$('jobs').textContent=JSON.stringify({tasks:s.tasks,jobs:s.jobs,candidates:s.candidates,finalization:s.finalization,ledger:s.project_ledger},null,2);$('download').hidden=!(s.finalization&&s.finalization.status==='verified_development_only');$('download').href='/v1/projects/'+encodeURIComponent(p)+'/bundle'}
async function command(state){let p=$('project').value.trim();if(revision===null)return;let id='ui-'+crypto.randomUUID();let body={command_id:id,project_id:p,expected_revision:revision,operation:{type:'lifecycle',state}};let r=await fetch('/v1/projects/'+encodeURIComponent(p)+'/commands',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});let data=await r.json();$('receipt').textContent=JSON.stringify(data,null,2);refresh()}
function connectEvents(){if(source)source.close();let p=$('project').value.trim();source=new EventSource('/v1/projects/'+encodeURIComponent(p)+'/events?after='+lastCursor);source.onmessage=e=>{lastCursor=e.lastEventId||lastCursor;let line=JSON.stringify(JSON.parse(e.data));$('events').textContent=(line+'\n'+$('events').textContent).slice(0,10000);refresh()};source.onerror=()=>{$('state').textContent='Event stream reconnecting; snapshot remains available'}}
setInterval(()=>{if(!$('app').hidden)refresh()},5000)
</script>'''


def serve(state: Path, binary: Path, port: int, token: str) -> None:
    state = state.resolve(strict=True)
    binary = binary.resolve(strict=True)

    class Handler(BaseHTTPRequestHandler):
        server_version = "MLRescurixDashboard/1"

        def log_message(self, fmt, *args):
            # Never log headers, cookies, command bodies, or credentials.
            return

        def send_json(self, value, code=200):
            data = json.dumps(value, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def valid_host(self):
            return self.headers.get("Host", "").lower() in {f"127.0.0.1:{port}", f"localhost:{port}"}

        def authenticated(self):
            cookie = http.cookies.SimpleCookie(self.headers.get("Cookie", ""))
            return "mlr_session" in cookie and secrets.compare_digest(cookie["mlr_session"].value, token)

        def read_body(self):
            size = int(self.headers.get("Content-Length", "-1"))
            if size < 0 or size > MAX_BODY:
                raise ValueError("request body exceeds limit")
            raw = self.rfile.read(size)
            return json.loads(raw)

        def rust(self, *args, input_data=None):
            proc = subprocess.run([str(binary), *args], input=input_data, capture_output=True,
                                  timeout=5, check=False, env={"PATH": "/usr/bin:/bin"})
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr.decode(errors="replace")[:2000])
            return json.loads(proc.stdout)

        def do_GET(self):
            try:
                self.handle_get()
            except Exception as exc:
                print(f"dashboard GET failed: {type(exc).__name__}: {str(exc)[:300]}", file=sys.stderr, flush=True)
                try: self.send_json({"error": "dashboard request failed"}, 500)
                except OSError: pass

        def handle_get(self):
            if not self.valid_host():
                return self.send_json({"error": "invalid Host"}, 400)
            path = urlparse(self.path)
            path = path._replace(path=unquote(path.path))
            if path.path == "/":
                data = PAGE.encode()
                self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data))); self.send_header("Cache-Control", "no-store")
                self.end_headers(); self.wfile.write(data); return
            if not self.authenticated():
                return self.send_json({"error": "authentication required"}, 401)
            match = re.fullmatch(r"/v1/projects/([A-Za-z0-9_.:-]{1,128})/snapshot", path.path)
            if match:
                project = match.group(1)
                snapshot = self.rust("snapshot", "--state", str(state), "--project", project)
                with sqlite3.connect(state / "state.sqlite") as db:
                    jobs = db.execute("SELECT id,spec,status,reason,pid,submitted_ms,started_ms,ended_ms,artifacts FROM jobs WHERE project=? ORDER BY submitted_ms,id LIMIT 256", (project,)).fetchall()
                snapshot["jobs"] = [{"job_id": row[0], "spec": json.loads(row[1]), "status": row[2], "reason": row[3],
                                     "pid": row[4], "submitted_ms": row[5], "started_ms": row[6], "ended_ms": row[7],
                                     "artifacts": json.loads(row[8])} for row in jobs]
                workflow = state / "workflows" / project
                for key, filename in (("dataset", "dataset_manifest.json"), ("candidates", "results.json"), ("finalization", "final.json")):
                    file = workflow / filename
                    if file.is_file() and not file.is_symlink() and file.stat().st_size <= 4 * 1024 * 1024:
                        try: snapshot[key] = json.loads(file.read_text())
                        except (ValueError, OSError): snapshot[key] = {"status": "unavailable"}
                return self.send_json(snapshot)
            match = re.fullmatch(r"/v1/commands/([A-Za-z0-9_.:-]{1,128})/([A-Za-z0-9_.:-]{1,128})", path.path)
            if match:
                with sqlite3.connect(state / "state.sqlite") as db:
                    row = db.execute("SELECT receipt FROM commands WHERE project=? AND id=?", match.groups()).fetchone()
                if not row: return self.send_json({"status": "unknown"}, 404)
                return self.send_json(json.loads(row[0]))
            match = re.fullmatch(r"/v1/projects/([A-Za-z0-9_.:-]{1,128})/events", path.path)
            if match:
                project = match.group(1)
                try: cursor = int(parse_qs(path.query).get("after", ["0"])[0])
                except ValueError: return self.send_json({"error": "invalid cursor"}, 400)
                if cursor < 0 or cursor > 9_007_199_254_740_991: return self.send_json({"error": "invalid cursor"}, 400)
                self.send_response(200); self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache"); self.send_header("X-Accel-Buffering", "no")
                self.end_headers()
                deadline = time.monotonic() + 25
                while time.monotonic() < deadline:
                    events = self.rust("events", "--state", str(state), "--project", project, "--after", str(cursor))
                    for event in events:
                        cursor = int(event["cursor"])
                        frame = f"id: {cursor}\ndata: {json.dumps(event, separators=(',', ':'))}\n\n".encode()
                        self.wfile.write(frame); self.wfile.flush()
                    if not events:
                        self.wfile.write(b": keepalive\n\n"); self.wfile.flush()
                    time.sleep(1)
                return
            match = re.fullmatch(r"/v1/projects/([A-Za-z0-9_.:-]{1,128})/bundle", path.path)
            if match:
                bundle = state / "workflows" / match.group(1) / "bundle"
                final_path = (bundle / "final.json")
                if not final_path.is_file() or json.loads(final_path.read_text()).get("status") != "verified_development_only":
                    return self.send_json({"error": "verified bundle is unavailable"}, 404)
                files = [p for p in bundle.rglob("*") if p.is_file()]
                if any(p.is_symlink() for p in files) or sum(p.stat().st_size for p in files) > 256 * 1024 * 1024:
                    return self.send_json({"error": "bundle exceeds safe download bound"}, 413)
                with tempfile.NamedTemporaryFile(prefix="mlr-bundle-", suffix=".zip", dir="/private/tmp", delete=False) as output:
                    archive_path = output.name
                try:
                    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                        for file in files: archive.write(file, file.relative_to(bundle).as_posix())
                    size = os.path.getsize(archive_path)
                    self.send_response(200); self.send_header("Content-Type", "application/zip")
                    self.send_header("Content-Disposition", "attachment; filename=mlrescurix-bundle.zip")
                    self.send_header("Content-Length", str(size)); self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    with open(archive_path, "rb") as archive: shutil.copyfileobj(archive, self.wfile)
                finally: os.unlink(archive_path)
                return
            return self.send_json({"error": "not found"}, 404)

        def do_POST(self):
            try:
                self.handle_post()
            except Exception as exc:
                print(f"dashboard POST failed: {type(exc).__name__}: {str(exc)[:300]}", file=sys.stderr, flush=True)
                try: self.send_json({"error": "dashboard request failed"}, 500)
                except OSError: pass

        def handle_post(self):
            if not self.valid_host(): return self.send_json({"error": "invalid Host"}, 400)
            origin = self.headers.get("Origin", "")
            if origin != f"http://127.0.0.1:{port}" and origin != f"http://localhost:{port}":
                return self.send_json({"error": "Origin rejected"}, 403)
            try: payload = self.read_body()
            except (ValueError, json.JSONDecodeError): return self.send_json({"error": "invalid bounded JSON body"}, 400)
            if self.path == "/session":
                if not isinstance(payload, dict) or not secrets.compare_digest(str(payload.get("token", "")), token):
                    return self.send_json({"error": "invalid token"}, 401)
                data = json.dumps({"authenticated": True}).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json")
                self.send_header("Set-Cookie", f"mlr_session={token}; HttpOnly; SameSite=Strict; Path=/; Max-Age=3600")
                self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data); return
            if not self.authenticated(): return self.send_json({"error": "authentication required"}, 401)
            match = re.fullmatch(r"/v1/projects/([A-Za-z0-9_.:-]{1,128})/commands", self.path)
            if not match or not isinstance(payload, dict) or payload.get("project_id") != match.group(1):
                return self.send_json({"error": "invalid command request"}, 400)
            if not ID.fullmatch(str(payload.get("command_id", ""))): return self.send_json({"error": "invalid command id"}, 400)
            try:
                data = json.dumps(payload, allow_nan=False).encode()
                with tempfile.NamedTemporaryFile(prefix="mlr-command-", suffix=".json", dir="/private/tmp", delete=False) as request:
                    request.write(data); request.flush(); os.fsync(request.fileno()); request_path = request.name
                try: result = self.rust("command", "--state", str(state), "--request", request_path)
                finally: os.unlink(request_path)
            except (ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
                return self.send_json({"error": str(exc)[:1000]}, 400)
            return self.send_json(result)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    try: server.serve_forever(poll_interval=0.25)
    finally: server.server_close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("port must be between 1024 and 65535")
    token = secrets.token_urlsafe(32)
    print(json.dumps({"url": f"http://127.0.0.1:{args.port}/", "session_token": token}), flush=True)
    serve(args.state, args.binary, args.port, token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
