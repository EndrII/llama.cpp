#!/usr/bin/env python3
"""Small persistent web host for OMP's native RPC protocol."""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import re
import signal
import subprocess
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


OMP = "/home/svt/.local/bin/omp"
APP_DIR = Path("/home/svt/.local/share/omp-web")
SESSION_ROOT = Path("/home/svt/.omp/web-sessions")
INDEX = APP_DIR / "index.html"
MAX_BODY = 1_048_576
SESSION_ID = re.compile(r"^[a-f0-9-]{36}$")
IDLE_SECONDS = 12 * 60 * 60

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("omp-web")

CLEANUP_SERVICE_WORKER = b"""\
const CUTOVER = 'omp-cutover-v1';
self.addEventListener('install', event => event.waitUntil(self.skipWaiting()));
self.addEventListener('activate', event => event.waitUntil((async () => {
  for (const key of await caches.keys()) await caches.delete(key);
  await self.registration.unregister();
  const windows = await self.clients.matchAll({type: 'window', includeUncontrolled: true});
  for (const client of windows) await client.navigate('/?omp=1');
})()));
self.addEventListener('fetch', event => event.respondWith(fetch(event.request)));
"""


class OmpSession:
    def __init__(self, session_id: str, cwd: str):
        self.session_id = session_id
        self.cwd = str(Path(cwd).expanduser().resolve())
        if not Path(self.cwd).is_dir():
            raise ValueError(f"cwd is not an existing directory: {self.cwd}")
        self.directory = SESSION_ROOT / session_id
        self.directory.mkdir(parents=True, exist_ok=True)
        self.events: queue.Queue[dict[str, Any]] = queue.Queue()
        self.prompt_lock = threading.Lock()
        self.write_lock = threading.Lock()
        self.last_used = time.monotonic()
        self.closed = False

        command = [
            OMP,
            "--mode",
            "rpc",
            "--thinking",
            "auto",
            "--cwd",
            self.cwd,
            "--auto-approve",
            "--session-dir",
            str(self.directory),
        ]
        if self.cwd == "/home/svt":
            command.append("--allow-home")
        if any(self.directory.rglob("*.jsonl")):
            command.append("--continue")

        env = os.environ.copy()
        env["HOME"] = "/home/svt"
        env["PATH"] = "/home/svt/.local/bin:/usr/local/bin:/usr/bin:/bin"
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
            start_new_session=True,
        )
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()
        try:
            ready = self.events.get(timeout=30)
        except queue.Empty as exc:
            self.close()
            raise RuntimeError("OMP RPC did not become ready within 30 seconds") from exc
        if ready.get("type") != "ready":
            self.close()
            raise RuntimeError(f"Unexpected OMP startup frame: {ready.get('type')}")
        log.info("session=%s started pid=%s cwd=%s", session_id, self.process.pid, self.cwd)

    def _read_stdout(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            try:
                event = json.loads(line)
                if isinstance(event, dict):
                    self.events.put(event)
            except json.JSONDecodeError:
                log.warning("session=%s invalid RPC frame", self.session_id)
        self.events.put({"type": "process_exit", "code": self.process.poll()})

    def _read_stderr(self) -> None:
        assert self.process.stderr is not None
        for line in self.process.stderr:
            log.warning("session=%s omp: %s", self.session_id, line.rstrip())

    def send(self, frame: dict[str, Any]) -> None:
        if self.process.poll() is not None:
            raise RuntimeError("OMP RPC process is not running")
        assert self.process.stdin is not None
        data = json.dumps(frame, separators=(",", ":")) + "\n"
        with self.write_lock:
            self.process.stdin.write(data)
            self.process.stdin.flush()
        self.last_used = time.monotonic()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            if self.process.stdin:
                self.process.stdin.close()
        except Exception:
            pass
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=10)
            except Exception:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        log.info("session=%s stopped", self.session_id)


sessions: dict[str, OmpSession] = {}
sessions_lock = threading.Lock()


def get_session(session_id: str, cwd: str) -> OmpSession:
    if not SESSION_ID.fullmatch(session_id):
        raise ValueError("invalid session id")
    with sessions_lock:
        current = sessions.get(session_id)
        if current and current.process.poll() is None:
            return current
        if current:
            current.close()
        current = OmpSession(session_id, cwd)
        sessions[session_id] = current
        return current


def reap_idle() -> None:
    while True:
        time.sleep(300)
        cutoff = time.monotonic() - IDLE_SECONDS
        with sessions_lock:
            expired = [key for key, value in sessions.items() if value.last_used < cutoff]
            for key in expired:
                sessions.pop(key).close()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "omp-web/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        log.info("%s %s", self.address_string(), fmt % args)

    def _security_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; connect-src 'self'",
        )

    def _json(self, status: int, value: dict[str, Any]) -> None:
        data = json.dumps(value, separators=(",", ":")).encode()
        self.send_response(status)
        self._security_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> dict[str, Any]:
        size = int(self.headers.get("Content-Length", "0"))
        if size <= 0 or size > MAX_BODY:
            raise ValueError("invalid request size")
        value = json.loads(self.rfile.read(size))
        if not isinstance(value, dict):
            raise ValueError("JSON body must be an object")
        return value

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/health":
            self._json(HTTPStatus.OK, {"status": "ok", "sessions": len(sessions)})
            return
        if path == "/":
            data = INDEX.read_bytes()
            self.send_response(HTTPStatus.OK)
            self._security_headers()
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if path == "/sw.js":
            data = CLEANUP_SERVICE_WORKER
            self.send_response(HTTPStatus.OK)
            self._security_headers()
            self.send_header("Content-Type", "application/javascript; charset=utf-8")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Service-Worker-Allowed", "/")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            body = self._body()
            session_id = str(body.get("sessionId", ""))
            cwd = str(body.get("cwd", "/home/svt"))
            session = get_session(session_id, cwd)
            if path == "/api/prompt":
                self._prompt(session, body)
            elif path == "/api/abort":
                session.send({"id": f"abort-{time.time_ns()}", "type": "abort"})
                self._json(HTTPStatus.OK, {"ok": True})
            elif path == "/api/ui-response":
                response = body.get("response")
                if not isinstance(response, dict):
                    raise ValueError("response must be an object")
                session.send(response)
                self._json(HTTPStatus.OK, {"ok": True})
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except (ValueError, json.JSONDecodeError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:
            log.exception("request failed")
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def _prompt(self, session: OmpSession, body: dict[str, Any]) -> None:
        message = body.get("message")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a non-empty string")
        if not session.prompt_lock.acquire(blocking=False):
            self._json(HTTPStatus.CONFLICT, {"error": "session is already running"})
            return

        self.send_response(HTTPStatus.OK)
        self._security_headers()
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        connected = True

        def emit(value: dict[str, Any]) -> None:
            nonlocal connected
            if not connected:
                return
            try:
                self.wfile.write((json.dumps(value, separators=(",", ":")) + "\n").encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                connected = False

        try:
            request_id = f"prompt-{time.time_ns()}"
            session.send({"id": request_id, "type": "prompt", "message": message})
            while True:
                event = session.events.get(timeout=3700)
                kind = event.get("type")
                if kind == "message_update":
                    update = event.get("assistantMessageEvent") or {}
                    update_kind = update.get("type")
                    if update_kind == "text_delta":
                        emit({"type": "delta", "text": update.get("delta", "")})
                    elif update_kind == "thinking_delta":
                        emit({"type": "thinking", "text": update.get("delta", "")})
                elif kind == "tool_execution_start":
                    call = event.get("toolCall") or {}
                    emit({"type": "status", "text": f"Running {call.get('name') or event.get('toolName') or 'tool'}"})
                elif kind == "tool_execution_end":
                    emit({"type": "status", "text": "Tool complete"})
                elif kind == "thinking_level_changed":
                    emit({"type": "status", "text": f"Thinking: {event.get('configured', 'auto')} → {event.get('resolved', event.get('thinkingLevel', 'auto'))}"})
                elif kind == "extension_ui_request":
                    method = event.get("method")
                    if method in {"confirm", "input", "select", "editor"}:
                        emit({"type": "ui_request", "request": event})
                    elif method == "notify":
                        emit({"type": "status", "text": str(event.get("message", ""))})
                elif kind == "response" and event.get("id") == request_id and not event.get("success"):
                    emit({"type": "error", "text": str(event.get("error", "prompt rejected"))})
                    break
                elif kind == "agent_end":
                    emit({"type": "done"})
                    break
                elif kind == "process_exit":
                    emit({"type": "error", "text": "OMP process exited"})
                    break
        except queue.Empty:
            emit({"type": "error", "text": "OMP produced no terminal event within one hour"})
        except Exception as exc:
            emit({"type": "error", "text": str(exc)})
        finally:
            session.last_used = time.monotonic()
            session.prompt_lock.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9120)
    args = parser.parse_args()
    APP_DIR.mkdir(parents=True, exist_ok=True)
    SESSION_ROOT.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=reap_idle, daemon=True).start()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    log.info("listening on http://%s:%s", args.host, args.port)
    try:
        server.serve_forever()
    finally:
        with sessions_lock:
            for session in sessions.values():
                session.close()


if __name__ == "__main__":
    main()
