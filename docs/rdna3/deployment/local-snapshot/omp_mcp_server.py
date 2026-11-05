#!/usr/bin/env python3
"""Dependency-free MCP stdio adapter exposing OMP to llama.cpp's web UI."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any


OMP = "/home/svt/.local/bin/omp"
DEFAULT_CWD = "/home/svt"
MAX_OUTPUT_CHARS = 120_000
RUN_TIMEOUT_SECONDS = 3600
PROTOCOL_VERSION = "2025-03-26"

TOOL = {
    "name": "use_omp",
    "description": (
        "Delegate any coding, file, shell, LSP, diagnostics, refactor, or "
        "web-search task to OMP. Pass the complete objective and relevant "
        "context in task. Use /home/svt when no project directory is known."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "task": {"type": "string", "description": "Complete task for OMP."},
            "cwd": {
                "type": "string",
                "description": "Existing working directory on obsidian.",
                "default": DEFAULT_CWD,
            },
        },
        "required": ["task"],
        "additionalProperties": False,
    },
}


def emit(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def result(request_id: Any, value: dict[str, Any]) -> None:
    emit({"jsonrpc": "2.0", "id": request_id, "result": value})


def error(request_id: Any, code: int, message: str) -> None:
    emit(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }
    )


def run_omp(arguments: dict[str, Any]) -> tuple[str, bool]:
    task = arguments.get("task")
    cwd = arguments.get("cwd", DEFAULT_CWD)
    if not isinstance(task, str) or not task.strip():
        return "OMP_NOT_RUN: task must be a non-empty string", True
    if not isinstance(cwd, str):
        return "OMP_NOT_RUN: cwd must be a string", True

    workdir = Path(cwd).expanduser().resolve()
    if not workdir.is_dir():
        return f"OMP_NOT_RUN: cwd is not an existing directory: {workdir}", True

    command = [
        OMP,
        "--print",
        "--no-session",
        "--auto-approve",
        "--thinking",
        "auto",
        "--cwd",
        str(workdir),
    ]
    if workdir == Path(DEFAULT_CWD):
        command.append("--allow-home")
    command.append(task)

    env = os.environ.copy()
    env["PATH"] = "/home/svt/.local/bin:/usr/local/bin:/usr/bin:/bin"
    process: subprocess.Popen[str] | None = None
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            start_new_session=True,
        )
        stdout, stderr = process.communicate(timeout=RUN_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        if process is not None:
            os.killpg(process.pid, signal.SIGTERM)
            process.communicate()
        return f"OMP_TIMEOUT: exceeded {RUN_TIMEOUT_SECONDS}s", True
    except Exception as exc:
        return f"OMP_FAILED_TO_START: {exc}", True

    out = stdout.strip()
    err = stderr.strip()
    if process.returncode != 0:
        details = err or out or "no output"
        return f"OMP_FAILED ({process.returncode}): {details[-MAX_OUTPUT_CHARS:]}", True
    return (out[-MAX_OUTPUT_CHARS:] if out else "OMP_OK: no textual output"), False


def dispatch(message: dict[str, Any]) -> None:
    if "id" not in message:
        return
    request_id = message["id"]
    method = message.get("method")
    params = message.get("params") or {}

    if method == "initialize":
        requested = params.get("protocolVersion")
        result(
            request_id,
            {
                "protocolVersion": requested or PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "omp-agent", "version": "1.0.0"},
            },
        )
    elif method == "ping":
        result(request_id, {})
    elif method == "tools/list":
        result(request_id, {"tools": [TOOL]})
    elif method == "tools/call":
        if params.get("name") != TOOL["name"]:
            error(request_id, -32602, f"Unknown tool: {params.get('name')}")
            return
        text, failed = run_omp(params.get("arguments") or {})
        result(
            request_id,
            {"content": [{"type": "text", "text": text}], "isError": failed},
        )
    else:
        error(request_id, -32601, f"Method not found: {method}")


def main() -> None:
    for line in sys.stdin:
        try:
            message = json.loads(line)
            if isinstance(message, dict):
                dispatch(message)
        except Exception as exc:
            print(f"omp-mcp: {exc}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
