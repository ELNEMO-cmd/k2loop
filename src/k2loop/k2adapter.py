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
import time
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
# the closing fence is a line of its own; the body may be empty (K2 lists new empty files like `pkg/__init__.py`)
_FENCE = re.compile(r"(?m)^(?P<path>[\w./\\-]+\.[A-Za-z0-9_]+)[ \t]*\n```[^\n]*\n(?P<body>.*?)^```[ \t]*$", re.S)

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
    return None if p.is_file() and old == new else _block(path.strip(), old, new)


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


_JSON_FENCE = re.compile(r"```(?:json)?[ \t]*\n(.*?)\n```", re.S)


def _json_dicts(obj) -> list[dict]:
    """Every dict in a JSON answer, outer first: an edit may be {"path","search","replace"} / {"path","content"}
    at any depth, e.g. {"files": [{"path": ..., "search": ..., "replace": ...}]}."""
    if isinstance(obj, list):
        return [d for x in obj for d in _json_dicts(x)]
    if not isinstance(obj, dict):
        return []
    return [obj] + [d for v in obj.values() for d in _json_dicts(v)]


def _json_edits(root: Path, text: str) -> tuple[list[str], str]:
    """K2 sometimes answers with its edits as plain JSON. Returns (SEARCH/REPLACE blocks, text without that JSON)."""
    blocks, rest = [], text
    cands = [(m.group(0), m.group(1)) for m in _JSON_FENCE.finditer(text)]
    if text.strip()[:1] in "{[":
        cands.append((text, text))
    for whole, body in cands:
        try:
            obj = json.loads(body)
        except ValueError:
            continue
        found = [b for b in (_edit_from_args(root, d) for d in _json_dicts(obj)) if b]
        if found:
            blocks += found
            rest = rest.replace(whole, "")
    return blocks, rest.strip()


_ANY_FENCE = re.compile(r"(?m)^```[^\n]*\n(?P<body>.*?)^```[ \t]*$\n?", re.S)


def repair_blocks(content: str) -> str:
    """K2 slips in Aider's SEARCH/REPLACE format and Aider then rejects the WHOLE answer ("did not conform to the edit
    format"), losing every good block with the bad one. Seen live: blocks closed by the fence without
    `>>>>>>> REPLACE` (the marker is added) and a stray `<<<<<<< SEARCH` fence with no divider (dropped)."""
    if "<<<<<<< SEARCH" not in content:
        return content

    def fix(m: re.Match) -> str:
        body = m.group("body")
        if not body.lstrip().startswith("<<<<<<< SEARCH") or ">>>>>>> REPLACE" in body:
            return m.group(0)
        if "\n=======" not in "\n" + body:
            return ""                                  # nothing to apply: an unfinished block
        whole = m.group(0)
        return whole[:m.start("body") - m.start()] + body + ">>>>>>> REPLACE\n" + whole[m.end("body") - m.start():]

    return _ANY_FENCE.sub(fix, content)


# a fence whose first line names the file: ```python\n# pkg/mod.py\n...```
_COMMENT_FENCE = re.compile(r"(?m)^```[^\n]*\n[ \t]*(?:#|//)[ \t]*(?P<path>[\w./-]+\.[A-Za-z0-9_]+)[ \t]*\n(?P<body>.*?)^```[ \t]*$",
                            re.S)


# for NEW files only: the path line may be decorated and followed by a colon and a blank line, as K2 writes while
# thinking ("borsa_agent/signals.py:\n\n```python"). Never used for existing files (a draft must not replace code).
_LOOSE_FENCE = re.compile(r"(?m)^[ \t]*[*`_]*(?P<path>[\w./-]+\.[A-Za-z0-9_]+)[*`_]*:?[ \t]*\n(?:[ \t]*\n)?"
                          r"```[^\n]*\n(?P<body>.*?)^```[ \t]*$", re.S)


def _new_file_blocks(root: Path, text: str) -> list[str]:
    """Blocks creating files that do not exist yet, from fences named by a path line or a `# path` first line; the
    last fence per path wins. Only NEW files: a snippet can never overwrite existing code this way."""
    latest: dict[str, str] = {}
    for rx in (_LOOSE_FENCE, _COMMENT_FENCE):
        for m in rx.finditer(text):
            latest[m.group("path")] = m.group("body")
    out = []
    for path, body in latest.items():
        p = _safe(root, path)
        if p is not None and not p.exists() and body.strip() and "<<<<<<< SEARCH" not in body:
            out.append(_block(path.strip(), "", body.strip("\n")))
    return out


def normalize(content: str, root: Path, extra_calls: list[dict] | None = None, reasoning: str = "") -> str:
    """Rewrite one K2 answer into text Aider's diff edit format can apply. `reasoning` (the server's
    reasoning_content) is only used when the answer itself has no edit: K2 sometimes writes the code while thinking
    and answers with an empty stub; then the NEW files written in the reasoning are created."""
    content = _THINK.sub("", content or "")
    ends = list(_THINK_END.finditer(content))
    if ends:  # drop inline reasoning (it may contain draft code blocks)
        content = content[ends[-1].end():]
    content = repair_blocks(content)
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
    if not blocks:
        blocks, prose = _json_edits(root, prose)
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
        blocks = _new_file_blocks(root, prose)
        if blocks:
            prose = _COMMENT_FENCE.sub("", prose).strip()
    if not blocks and reasoning:
        blocks = _new_file_blocks(root, _THINK.sub("", reasoning))
    if not blocks:
        return prose
    return (prose + "\n\n" if prose else "") + "\n".join(blocks)


# Aider's reply when an answer names a repo file that is not in the chat: it adds the file and asks again, and the
# edits of that answer are dropped without being applied (aider/coders/base_coder.py, check_for_file_mentions).
ADDED_FILES = "I added these files to the chat:"


def replay_edits(body: bytes) -> bytes | None:
    """When Aider re-asks only because it added mentioned files, answer with the previous reply again instead of
    calling the model: with the files now in the chat Aider applies those edits. K2 rarely repeats its edits when
    asked again, so without this the whole answer was lost. None = forward the request to the model."""
    try:
        req = json.loads(body)
    except ValueError:
        return None
    msgs = req.get("messages") if isinstance(req, dict) else None
    if req.get("stream") or not isinstance(msgs, list) or len(msgs) < 2:
        return None
    last, prev = msgs[-1], msgs[-2]
    if not (isinstance(last, dict) and isinstance(prev, dict) and last.get("role") == "user"
            and prev.get("role") == "assistant"):
        return None
    text, old = last.get("content"), prev.get("content")
    if not (isinstance(text, str) and isinstance(old, str) and text.startswith(ADDED_FILES)
            and "<<<<<<< SEARCH" in old and ">>>>>>> REPLACE" in old):
        return None
    return json.dumps({
        "id": "k2adapter-replay", "object": "chat.completion", "created": int(time.time()),
        "model": req.get("model", ""),
        "choices": [{"index": 0, "message": {"role": "assistant", "content": old}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }, ensure_ascii=False).encode()


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
                if self.path.rstrip("/").endswith("chat/completions"):
                    again = replay_edits(body)
                    if again is not None:
                        self._send(200, "application/json", again)
                        return
                code, ctype, data = self._relay("POST", body)
                if code == 200 and self.path.rstrip("/").endswith("chat/completions") and "json" in ctype:
                    try:
                        obj = json.loads(data)
                        msg = obj["choices"][0]["message"]
                        msg["content"] = normalize(msg.get("content") or "", proxy.root,
                                                   _tool_calls_from_message(msg),
                                                   str(msg.get("reasoning_content") or ""))
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
