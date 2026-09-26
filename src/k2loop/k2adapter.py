"""Aider <-> K2 Horizon translator.

K2 Horizon is trained for agentic tool use and often answers Aider's edit requests with its own tool-call
syntax instead of Aider's text SEARCH/REPLACE blocks, e.g.

    <ifm|tool_calls><ifm|tool_call>SEARCH
    <ifm|arg_key>path</ifm|arg_key><ifm|arg_value>calc.py</ifm|arg_value>
    <ifm|arg_key>old_string</ifm|arg_key><ifm|arg_value>...</ifm|arg_value>
    <ifm|arg_key>new_string</ifm|arg_key><ifm|arg_value>...</ifm|arg_value>
    </ifm|tool_call></ifm|tool_calls>

or with whole files in fenced blocks. Aider applies none of these, so every candidate came back empty.
`AdapterProxy` sits between Aider and llama-swap for one Aider run and rewrites each answer into SEARCH/REPLACE
blocks (file names inferred from the worktree when missing); correct SEARCH/REPLACE answers pass through.
"""
from __future__ import annotations

import json
import re
import subprocess
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_CALL = re.compile(r"<ifm\|tool_call>(.*?)</ifm\|tool_call>", re.S)
_ARG = re.compile(r"<ifm\|arg_key>(.*?)</ifm\|arg_key>\s*<ifm\|arg_value>(.*?)</ifm\|arg_value>", re.S)
_CALLS_BLOCK = re.compile(r"<ifm\|tool_calls>.*?(</ifm\|tool_calls>|$)", re.S)
_IFM_TAG = re.compile(r"<\/?ifm\|[^>]*>")
_THINK = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.S | re.I)
_THINK_END = re.compile(r"</ifm\|think[^>]*>|</think>", re.I)
_FENCE = re.compile(r"(?m)^(?P<path>[\w./\\-]+\.[A-Za-z0-9_]+)[ \t]*\n```[^\n]*\n(?P<body>.*?)\n```[ \t]*$", re.S)

PATH_KEYS = ("path", "file_path", "filepath", "filename", "file", "target_file")
OLD_KEYS = ("old_string", "search_block", "search", "old", "old_str", "original", "find")
NEW_KEYS = ("new_string", "replace_block", "replace", "new", "new_str", "updated", "replacement")
CONTENT_KEYS = ("content", "file_text", "text", "code")


def _first(d: dict, keys: tuple[str, ...]) -> str | None:
    for k in keys:
        if k in d and d[k] is not None:
            return str(d[k])
    return None


def _safe(root: Path, rel: str) -> Path | None:
    rel = rel.strip().strip("`'\"").replace("\\", "/").removeprefix("./") if rel else ""
    if not rel:
        return None
    p = (root / rel).resolve()
    try:
        p.relative_to(root.resolve())
    except ValueError:
        return None
    return p


def _tracked(root: Path) -> list[str]:
    r = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True)
    return r.stdout.splitlines() if r.returncode == 0 else []


def _find_file(root: Path, old: str) -> str | None:
    needle = old.strip()
    if not needle:
        return None
    for rel in _tracked(root):
        try:
            if needle in (root / rel).read_text(encoding="utf-8", errors="ignore"):
                return rel
        except OSError:
            continue
    return None


def _block(path: str, old: str, new: str) -> str:
    return f"{path}\n```\n<<<<<<< SEARCH\n{old}\n=======\n{new}\n>>>>>>> REPLACE\n```\n"


def _edit_from_args(root: Path, args: dict) -> str | None:
    path = _first(args, PATH_KEYS)
    old = _first(args, OLD_KEYS)
    new = _first(args, NEW_KEYS)
    content = _first(args, CONTENT_KEYS)
    if old is not None and new is not None:
        path = path or _find_file(root, old)
        return _block(path, old.strip("\n"), new.strip("\n")) if path and _safe(root, path) else None
    if path and content is not None:
        return _whole_file(root, path, content)
    return None


def _whole_file(root: Path, path: str, content: str) -> str | None:
    p = _safe(root, path)
    if p is None:
        return None
    old = p.read_text(encoding="utf-8", errors="ignore").strip("\n") if p.is_file() else ""
    new = content.strip("\n")
    return None if old == new else _block(path.strip(), old, new)


def _tool_calls_from_message(msg: dict) -> list[dict]:
    out = []
    for tc in msg.get("tool_calls") or []:
        fn = (tc or {}).get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            continue
        if isinstance(args, dict):
            out.append(args)
    return out


def normalize(content: str, root: Path, extra_calls: list[dict] | None = None) -> str:
    """Rewrite one K2 answer into text Aider's diff edit format can apply."""
    content = _THINK.sub("", content or "")
    ends = list(_THINK_END.finditer(content))
    if ends:  # drop inline reasoning (it may contain draft code blocks)
        content = content[ends[-1].end():]
    if "<<<<<<< SEARCH" in content and ">>>>>>> REPLACE" in content and "<ifm|tool_call>" not in content:
        return _IFM_TAG.sub("", content)
    blocks: list[str] = []
    for call in _CALL.findall(content):
        args = {k.strip(): v for k, v in _ARG.findall(call)}
        b = _edit_from_args(root, args)
        if b:
            blocks.append(b)
    for args in extra_calls or []:
        b = _edit_from_args(root, args)
        if b:
            blocks.append(b)
    prose = _IFM_TAG.sub("", _CALLS_BLOCK.sub("", content)).strip()
    if not blocks:  # whole files in fenced blocks: the last block per path wins
        latest: dict[str, str] = {}
        for m in _FENCE.finditer(prose):
            latest[m.group("path")] = m.group("body")
        for path, body in latest.items():
            b = _whole_file(root, path, body)
            if b:
                blocks.append(b)
        if blocks:
            prose = _FENCE.sub("", prose).strip()
    if not blocks:
        return prose
    return (prose + "\n\n" if prose else "") + "\n".join(blocks)


class AdapterProxy:
    """OpenAI-compatible pass-through on 127.0.0.1:<random> for the duration of one Aider run."""

    def __init__(self, upstream: str, root: Path):
        self.upstream = upstream.rstrip("/")
        self.root = root
        proxy = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _relay(self, method: str, body: bytes | None):
                headers = {k: v for k, v in self.headers.items()
                           if k.lower() in ("content-type", "authorization", "accept")}
                path = self.path[3:] if self.path.startswith("/v1") else self.path
                req = urllib.request.Request(proxy.upstream + path, data=body, method=method, headers=headers)
                try:
                    with urllib.request.urlopen(req, timeout=3600) as r:
                        return r.status, r.headers.get("Content-Type", "application/json"), r.read()
                except urllib.error.HTTPError as e:
                    return e.code, "application/json", e.read()

            def _send(self, code: int, ctype: str, data: bytes):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):  # noqa: N802
                self._send(*self._relay("GET", None))

            def do_POST(self):  # noqa: N802
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                code, ctype, data = self._relay("POST", body)
                if code == 200 and self.path.rstrip("/").endswith("chat/completions") and "json" in ctype:
                    try:
                        obj = json.loads(data)
                        msg = obj["choices"][0]["message"]
                        msg["content"] = normalize(msg.get("content") or "", proxy.root,
                                                   _tool_calls_from_message(msg))
                        msg.pop("tool_calls", None)
                        data = json.dumps(obj, ensure_ascii=False).encode()
                    except (ValueError, KeyError, IndexError, TypeError):
                        pass
                self._send(code, ctype, data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def __enter__(self) -> "AdapterProxy":
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc) -> None:
        self.server.shutdown()
        self.server.server_close()
