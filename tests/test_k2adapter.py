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


# verbatim shape of a real K2 answer (bench date_bugfix, 2026-09-27): the edit as plain JSON in a json fence
JSON_EDIT = '''
```json
{"files":[{"path":"calc.py","search":"def add(a, b):\\n    return a + b","replace":"def add(a, b):\\n    return a + b\\n\\n\\ndef neg(a):\\n    return -a"}]}
```'''


def test_plain_json_edits_become_search_replace(tmp_path):
    r = _repo(tmp_path)
    out = normalize(JSON_EDIT, r)
    assert out.startswith("calc.py\n```\n<<<<<<< SEARCH\ndef add(a, b):") and "def neg(a):" in out
    assert "```json" not in out
    nested = json.dumps({"edits": [{"search": "def add(a, b):\n    return a + b", "replace": "def add(a, b):\n    return b + a"}]})
    assert normalize(nested, r).startswith("calc.py\n```\n<<<<<<< SEARCH")  # no path: found by the SEARCH text
    mapping = '```json\n{"stat.py": "stats.py", "tests.py": "tests/test_stats.py"}\n```'  # not an edit
    assert normalize(mapping, r) == mapping


def test_replay_when_aider_only_added_files():
    from k2loop.k2adapter import replay_edits
    edits = "calc.py\n```\n<<<<<<< SEARCH\na\n=======\nb\n>>>>>>> REPLACE\n```"
    msgs = [{"role": "user", "content": "task"}, {"role": "assistant", "content": edits},
            {"role": "user", "content": "I added these files to the chat: tests/test_calc.py\nLet me know if there are others we should add."}]
    out = json.loads(replay_edits(json.dumps({"model": "k2", "messages": msgs}).encode()))
    assert out["choices"][0]["message"]["content"] == edits
    assert replay_edits(json.dumps({"messages": msgs[:-1] + [{"role": "user", "content": "fix it"}]}).encode()) is None
    assert replay_edits(json.dumps({"messages": [msgs[0], {"role": "assistant", "content": "ok"}, msgs[2]]}).encode()) is None
    assert replay_edits(json.dumps({"stream": True, "messages": msgs}).encode()) is None
    assert replay_edits(b"not json") is None


def test_empty_fences_do_not_swallow_the_next_file(tmp_path):
    """Verbatim from a live run (borsa-agent step 1): two new empty files. The old fence regex took the second
    file's name as the first file's content."""
    r = _repo(tmp_path)
    out = normalize("borsa_agent/__init__.py\n```python\n```\n\ntests/__init__.py\n```python\n```\n", r)
    assert out.count("<<<<<<< SEARCH\n\n=======\n\n>>>>>>> REPLACE") == 2
    assert out.index("borsa_agent/__init__.py") < out.index("tests/__init__.py") and "```python" not in out
    assert normalize("calc.py\n```python\n" + ORIG + "```\n", r) == "calc.py\n```python\n" + ORIG + "```"  # unchanged


# shape of a live K2 answer (borsa-agent, new files): no `>>>>>>> REPLACE`, then a stray unfinished block
UNCLOSED = """Here are the blocks.

pkg/new.py
```python
<<<<<<< SEARCH
=======
def f():
    return 1
```

tests/test_new.py
```python
<<<<<<< SEARCH
=======
from pkg.new import f
```

I'll create two new files.

```python
<<<<<<< SEARCH

```
"""


def test_unclosed_blocks_are_repaired_and_stray_ones_dropped(tmp_path):
    r = _repo(tmp_path)
    out = normalize(UNCLOSED, r)
    assert out.count(">>>>>>> REPLACE") == 2 and out.count("<<<<<<< SEARCH") == 2
    assert "pkg/new.py\n```python\n<<<<<<< SEARCH\n=======\ndef f():\n    return 1\n>>>>>>> REPLACE\n```" in out
    ok = "calc.py\n```\n<<<<<<< SEARCH\na\n=======\nb\n>>>>>>> REPLACE\n```"
    assert normalize(ok, r) == ok                                   # correct answers are untouched


def test_stub_answer_takes_new_files_from_reasoning_only(tmp_path):
    """Live shape (borsa-agent step 2): the answer was only "```python\\n# pkg/ind.py\\n```", the code was in the
    reasoning. New files are taken from there; existing files never (a thought-draft must not overwrite code)."""
    r = _repo(tmp_path)
    stub = "```python\n# pkg/ind.py\n```"
    reasoning = ("Let me write it.\n\n```python\n# pkg/ind.py\ndef sma(xs, n):\n    return sum(xs[-n:]) / n\n```\n\n"
                 "calc.py\n```python\ndef add(a, b):\n    return 0\n```\n")
    out = normalize(stub, r, reasoning=reasoning)
    assert "pkg/ind.py\n```\n<<<<<<< SEARCH\n\n=======\ndef sma(xs, n):" in out
    assert "return 0" not in out                                               # existing calc.py untouched
    assert normalize(stub, r) == stub                                          # no reasoning: nothing invented
    named = "Here it is.\n\n```python\n# pkg/other.py\nX = 1\n```"
    assert "pkg/other.py\n```\n<<<<<<< SEARCH\n\n=======\nX = 1\n>>>>>>> REPLACE" in normalize(named, r)


def test_reasoning_path_with_colon_and_blank_line_creates_new_files(tmp_path):
    """Live shape (borsa-agent signals step): code only in the reasoning, each file introduced as `path:` + blank line;
    drafts come first, the last block per path is the final one."""
    r = _repo(tmp_path)
    reasoning = ("Tests:\n\n```python\ndraft\n```\n\npkg/signals.py:\n\n```python\nOLD = 1\n```\n\nBetter:\n\n"
                 "**`pkg/signals.py`**:\n\n```python\nNEW = 2\n```\n\ncalc.py:\n\n```python\nbroken\n```\n")
    out = normalize("I'll create the two new files now:", r, reasoning=reasoning)
    assert "pkg/signals.py\n```\n<<<<<<< SEARCH\n\n=======\nNEW = 2\n>>>>>>> REPLACE" in out
    assert "OLD = 1" not in out and "broken" not in out and "draft" not in out
