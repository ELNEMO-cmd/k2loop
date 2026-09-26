"""The three answer styles below are verbatim from the real K2 Horizon server (demo run, 2026-09-25)."""
import json
import subprocess
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from k2loop.k2adapter import AdapterProxy, normalize

ORIG = "def add(a, b):\n    return a + b\n"

CALL_NO_PATH = """I'll add `multiply` and `divide` to calc.py.<ifm|tool_calls>
<ifm|tool_call>SEARCH/REPLACE block
<ifm|arg_key>search_block</ifm|arg_key>
<ifm|arg_value>def add(a, b):
    return a + b
</ifm|arg_value>
<ifm|arg_key>replace_block</ifm|arg_key>
<ifm|arg_value>def add(a, b):
    return a + b


def multiply(a, b):
    return a * b
</ifm|arg_value>
</ifm|tool_call>
</ifm|tool_calls>"""

CALL_WITH_PATH = """<ifm|tool_calls>
<ifm|tool_call>SEARCH
<ifm|arg_key>context_lines</ifm|arg_key>
<ifm|arg_value>3</ifm|arg_value>
<ifm|arg_key>old_string</ifm|arg_key>
<ifm|arg_value>def add(a, b):
    return a + b</ifm|arg_value>
<ifm|arg_key>path</ifm|arg_key>
<ifm|arg_value>calc.py</ifm|arg_value>
<ifm|arg_key>new_string</ifm|arg_key>
<ifm|arg_value>def add(a, b):
    return a + b


def divide(a, b):
    if b == 0:
        raise ValueError("cannot divide by zero")
    return a / b</ifm|arg_value>
</ifm|tool_call>
</ifm|tool_calls>"""

WHOLE_FILES = """calc.py
```python
def add(a, b):
    return a + b


def multiply(a, b):
    return a * b
```

tests/test_new.py
```python
from calc import multiply


def test_multiply():
    assert multiply(2, 3) == 6
```

tests/test_new.py
```python
import pytest
from calc import multiply


def test_multiply():
    assert multiply(2, 3) == 6
```
"""


def _repo(tmp_path):
    r = tmp_path / "r"
    (r / "tests").mkdir(parents=True)
    (r / "calc.py").write_text(ORIG, encoding="utf-8")
    (r / ".k2loop").mkdir()
    (r / ".k2loop" / "check.sh").write_text("echo ok\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=r, check=True)
    subprocess.run(["git", "add", "-A"], cwd=r, check=True)
    return r


def test_tool_call_without_path_finds_the_file(tmp_path):
    out = normalize(CALL_NO_PATH, _repo(tmp_path))
    assert "calc.py\n```\n<<<<<<< SEARCH\ndef add(a, b):\n    return a + b\n=======\n" in out
    assert "def multiply(a, b):" in out and ">>>>>>> REPLACE" in out and "<ifm|" not in out
    assert out.startswith("I'll add")


def test_tool_call_with_path(tmp_path):
    out = normalize(CALL_WITH_PATH, _repo(tmp_path))
    assert out.startswith("calc.py\n```\n<<<<<<< SEARCH\n") and 'raise ValueError("cannot divide by zero")' in out


def test_whole_files_become_search_replace(tmp_path):
    out = normalize(WHOLE_FILES, _repo(tmp_path))
    assert "calc.py\n```\n<<<<<<< SEARCH\ndef add(a, b):\n    return a + b\n=======\n" in out
    new = out.split("tests/test_new.py\n```\n<<<<<<< SEARCH\n", 1)[1]
    assert new.startswith("\n=======\nimport pytest")  # new file (empty SEARCH), last copy wins
    assert out.count("tests/test_new.py") == 1


def test_passthrough_reasoning_and_path_safety(tmp_path):
    r = _repo(tmp_path)
    ok = "calc.py\n```\n<<<<<<< SEARCH\nx\n=======\ny\n>>>>>>> REPLACE\n```"
    assert normalize("</ifm|think_faster>" + ok, r) == ok
    draft = "<ifm|think>calc.py\n```python\nDRAFT\n```\n</ifm|think_faster>" + CALL_WITH_PATH
    assert "DRAFT" not in normalize(draft, r)
    evil = CALL_WITH_PATH.replace("<ifm|arg_value>calc.py</ifm|arg_value>", "<ifm|arg_value>../../etc/passwd</ifm|arg_value>")
    assert "passwd" not in normalize(evil, r)
    assert normalize("I could not do it.", r) == "I could not do it."


def test_proxy_rewrites_upstream_answer(tmp_path):
    r = _repo(tmp_path)

    class Up(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):  # noqa: N802
            self.rfile.read(int(self.headers["Content-Length"]))
            data = json.dumps({"choices": [{"message": {"role": "assistant", "content": CALL_WITH_PATH}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    up = ThreadingHTTPServer(("127.0.0.1", 0), Up)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    with AdapterProxy(f"http://127.0.0.1:{up.server_address[1]}/v1", r) as px:
        req = urllib.request.Request(px.url + "/chat/completions", data=b'{"model":"k2"}',
                                     headers={"Content-Type": "application/json"})
        body = json.loads(urllib.request.urlopen(req, timeout=10).read())
    up.shutdown()
    assert body["choices"][0]["message"]["content"].startswith("calc.py\n```\n<<<<<<< SEARCH")
