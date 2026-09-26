"""Parse the Bernstein worker prompt and report the terminal outcome back to its task server."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .heartbeat import find_orchestrator_root

_PLACEHOLDERS = {"{{TASK_ID}}", "<task_id>", "{task_id}"}


@dataclass
class PromptInfo:
    task_ids: list[str] = field(default_factory=list)
    base_url: str = "http://127.0.0.1:8052"
    token: str | None = None
    heartbeat_paths: list[Path] = field(default_factory=list)
    root: Path | None = None
    session_id: str | None = None
    shutdown_file: Path | None = None


def parse_task_ids(prompt: str) -> list[str]:
    ids = re.findall(r"^#+\s*Task \d+:.*\(id=([A-Za-z0-9._:-]+)\)\s*$", prompt, re.M)          # Bernstein 3.x
    ids += re.findall(r"bernstein task complete ([A-Za-z0-9._:-]+)", prompt)
    ids += [m for m in re.findall(r"/tasks/([^/\s'\"]+)/complete", prompt) if m not in _PLACEHOLDERS]
    m = re.search(r"one POST per id\):\s*([^\n]+)", prompt)
    if m:
        ids += [x.strip().strip(".") for x in m.group(1).split(",") if x.strip()]
    out: list[str] = []
    for i in ids:
        if i and i not in out and i not in _PLACEHOLDERS:
            out.append(i)
    return out


def parse_prompt(prompt: str, cwd: Path) -> PromptInfo:
    info = PromptInfo(task_ids=parse_task_ids(prompt))
    m = re.search(r"(https?://(?:127\.0\.0\.1|localhost):\d+)/tasks/", prompt)
    if m:
        info.base_url = m.group(1)
    m = re.search(r"Authorization: Bearer ([A-Za-z0-9._~+/=-]+)", prompt)
    info.root = find_orchestrator_root(cwd)
    tok_file = re.search(r"(/[^\s)\"'`]+/agent_tokens/[^\s)\"'`]+\.token)", prompt)
    if tok_file and Path(tok_file.group(1)).exists():
        info.token = Path(tok_file.group(1)).read_text(encoding="utf-8").strip() or None
    elif os.environ.get("BERNSTEIN_AUTH_TOKEN"):
        info.token = os.environ["BERNSTEIN_AUTH_TOKEN"]
    elif m and not m.group(1).startswith("<"):
        info.token = m.group(1)
    elif info.root and (info.root / ".sdd/runtime/auth.token").exists():
        info.token = (info.root / ".sdd/runtime/auth.token").read_text(encoding="utf-8").strip() or None
    if info.root and (info.root / ".sdd/runtime/server.port").exists():
        port = (info.root / ".sdd/runtime/server.port").read_text(encoding="utf-8").strip()
        if port.isdigit() and not m:
            info.base_url = f"http://127.0.0.1:{port}"
    sig = re.search(r"\.sdd/runtime/signals/([^/\s]+)/", prompt)
    if sig:
        info.session_id = sig.group(1)
        info.shutdown_file = cwd / ".sdd" / "runtime" / "signals" / sig.group(1) / "SHUTDOWN"
    hb = [Path(p) for p in re.findall(r"'([^']*?/\.sdd/runtime/heartbeats/[^'/]+\.json)'", prompt)]
    if not hb and info.root and info.session_id:
        hb = [info.root / ".sdd" / "runtime" / "heartbeats" / f"{info.session_id}.json"]
    if not hb and info.root:
        logs = sorted((cwd / ".sdd" / "runtime").glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        if logs:
            hb = [info.root / ".sdd" / "runtime" / "heartbeats" / f"{logs[0].stem}.json"]
    info.heartbeat_paths = hb
    return info


_DROP_SECTION = re.compile(r"^#+\s*(Done signal|Heartbeat|Completion|Auth)", re.I)


def clean_task_text(prompt: str, limit: int = 14_000) -> str:
    """Strip Bernstein plumbing (curl/heartbeat/contract) so the coding model sees only the task."""
    out, skipping, in_code = [], False, False
    for ln in prompt.splitlines():
        if ln.lstrip().startswith("#") and not in_code:
            skipping = bool(_DROP_SECTION.match(ln.lstrip()))
        if ln.strip().startswith("```"):
            in_code = not in_code
        if skipping:
            continue
        if re.search(r"curl |/tasks/|heartbeats/|http_code|Content-Type: application/json", ln):
            continue
        out.append(ln)
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()
    return text[:limit]


def mentioned_files(text: str, root: Path, limit: int = 8) -> list[str]:
    found: list[str] = []
    for tok in re.findall(r"[\w./-]+\.[A-Za-z0-9]{1,6}", text):
        tok = tok.strip("./") if tok.startswith("./") else tok
        if tok not in found and (root / tok).is_file():
            found.append(tok)
        if len(found) >= limit:
            break
    return found


def _post(info: PromptInfo, path: str, body: dict) -> tuple[int, str]:
    headers = {"Content-Type": "application/json"}
    if info.token:
        headers["Authorization"] = f"Bearer {info.token}"
    req = urllib.request.Request(info.base_url + path, data=json.dumps(body).encode(), headers=headers,
                                 method="POST")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, r.read().decode()[:500]
        except urllib.error.HTTPError as exc:  # 4xx: state changed, do not retry
            return exc.code, exc.read().decode(errors="ignore")[:500]
        except (urllib.error.URLError, ConnectionError):
            time.sleep(2 * (attempt + 1))
    return 0, "connection failed"


def _cli_complete(info: PromptInfo, tid: str, summary: str) -> tuple[int, str] | None:
    """Bernstein 3.x first-class completion CLI (resolves port + token itself)."""
    if not shutil.which("bernstein"):
        return None
    env = dict(os.environ, BERNSTEIN_SERVER_URL=info.base_url)
    if info.token:
        env["BERNSTEIN_AUTH_TOKEN"] = info.token
    p = subprocess.run(["bernstein", "task", "complete", tid, "--summary", summary[:2000]],
                       cwd=info.root or Path.cwd(), env=env, capture_output=True, text=True, timeout=120)
    return (200 if p.returncode == 0 else 0), (p.stdout + p.stderr).strip()[:500]


def report_complete(info: PromptInfo, summary: str, files: list[str], verify_cmd: str | None, rc: int | None):
    out = []
    for tid in info.task_ids:
        r = _cli_complete(info, tid, summary)
        if r and r[0] == 200:
            out.append((tid, *r))
            continue
        payload = {"contract": "worker-completion/v1", "summary": summary[:2000], "files_changed": files,
                   "verification": ({"command": verify_cmd, "exit_code": rc} if verify_cmd else None)}
        out.append((tid, *_post(info, f"/tasks/{tid}/complete", {"payload": payload})))
    return out


def report_refusal(info: PromptInfo, question: str):
    payload = {"contract": "worker-completion/v1", "kind": "awaiting_operator", "detail": question[:1500],
               "question": question[:1500]}
    return [(_tid, *_post(info, f"/tasks/{_tid}/complete", {"payload": payload})) for _tid in info.task_ids]


def report_fail(info: PromptInfo, reason: str):
    return [(_tid, *_post(info, f"/tasks/{_tid}/fail", {"reason": reason[:1500]})) for _tid in info.task_ids]
