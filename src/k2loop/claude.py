"""Thin wrapper around Claude Code headless mode (`claude -p`, prompt on stdin)."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

_LIMIT_RE = re.compile(r"usage limit|rate limit|limit reached|quota|overloaded", re.I)
READ_TOOLS = "Read,Grep,Glob"
EDIT_TOOLS = "Read,Edit,Write,Grep,Glob,Bash"


@dataclass
class ClaudeResult:
    ok: bool
    text: str
    limited: bool = False
    error: str = ""


def parse_claude_json(stdout: str) -> tuple[str, bool]:
    """Return (result_text, is_error) from `--output-format json` output."""
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        return stdout.strip(), False
    if isinstance(data, list):
        res = [d for d in data if isinstance(d, dict) and d.get("type") == "result"]
        data = res[-1] if res else (data[-1] if data and isinstance(data[-1], dict) else {})
    if not isinstance(data, dict):
        return "", True
    text = data.get("result") or ""
    is_error = bool(data.get("is_error")) or data.get("subtype", "success") != "success"
    return (text if isinstance(text, str) else json.dumps(text)), is_error


def run_claude(cfg: dict, prompt: str, cwd: Path, *, model: str, edit: bool,
               timeout: float = 1800, max_turns: int | None = None) -> ClaudeResult:
    c = cfg["claude"]
    binary = c.get("binary", "claude")
    if not c.get("enabled", True):
        return ClaudeResult(False, "", error="claude disabled in config")
    if not shutil.which(binary):
        return ClaudeResult(False, "", error=f"{binary!r} not found on PATH")
    cmd = [binary, "-p", "Follow the instructions given on stdin exactly.",
           "--output-format", "json", "--model", model,
           "--max-turns", str(max_turns or c.get("max_turns", 40))]
    if edit:
        cmd += ["--permission-mode", "acceptEdits", "--allowedTools", EDIT_TOOLS]
    else:
        cmd += ["--allowedTools", READ_TOOLS, "--disallowedTools", "Edit,Write,Bash,NotebookEdit"]
    try:
        proc = subprocess.run(cmd, cwd=cwd, input=prompt[:180_000], capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return ClaudeResult(False, "", error="claude timed out")
    text, is_error = parse_claude_json(proc.stdout)
    limited = bool(_LIMIT_RE.search(f"{text}\n{proc.stderr}")) and (proc.returncode != 0 or is_error)
    ok = proc.returncode == 0 and not is_error and bool(text.strip())
    return ClaudeResult(ok, text, limited, proc.stderr[-2000:])


def version(cfg: dict) -> str:
    binary = cfg["claude"].get("binary", "claude")
    if not shutil.which(binary):
        return ""
    try:
        return subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""
