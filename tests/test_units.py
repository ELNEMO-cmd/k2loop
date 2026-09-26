import json
import subprocess
from pathlib import Path

import yaml

from k2loop import gitops
from k2loop.bernstein_io import clean_task_text, parse_prompt, parse_task_ids
from k2loop.claude import parse_claude_json
from k2loop.cli import bernstein_yaml
from k2loop.config import DEFAULTS
from k2loop.decisions import Decider, classify_test_rule, compact_state, retry_escalate_rule
from k2loop.llm import extract_json, strip_reasoning
from k2loop.planner import to_bernstein_yaml, validate_plan
from k2loop.project import detect, render_check_sh
from k2loop.testrun import count_failures, log_tail, run_checks, signature

PROMPT = """You are a backend agent.
## Task
Add a function add(a, b) to src/calc.py and a test in tests/test_calc.py.

## Done signal (completion contract worker-completion/v1)
Report your terminal outcome.
```bash
curl -s -w '\\n%{http_code}' -X POST http://127.0.0.1:8052/tasks/t-0042/complete \\
  -H "Content-Type: application/json" \\
  -d '{"payload": {"contract": "worker-completion/v1"}}'
```
# heartbeat cadence
mkdir -p '/home/u/proj/.sdd/runtime/heartbeats'; (while [ ! -f x ]; do printf '{}' > '/home/u/proj/.sdd/runtime/heartbeats/backend-abc123.json'; done) &
Then exit.
"""


def test_prompt_parsing(tmp_path):
    assert parse_task_ids(PROMPT) == ["t-0042"]
    assert parse_task_ids("report ... (one POST per id): a1, b2\n/tasks/{{TASK_ID}}/complete") == ["a1", "b2"]
    info = parse_prompt(PROMPT, tmp_path)
    assert info.base_url == "http://127.0.0.1:8052"
    assert info.heartbeat_paths == [Path("/home/u/proj/.sdd/runtime/heartbeats/backend-abc123.json")]
    txt = clean_task_text(PROMPT)
    assert "add(a, b)" in txt and "curl" not in txt and "/tasks/" not in txt and "heartbeats" not in txt


def test_json_and_reasoning():
    assert strip_reasoning("<think>hmm</think>  ok") == "ok"
    assert strip_reasoning("long thoughts</think>answer") == "answer"
    assert strip_reasoning("</ifm|think_faster>pong") == "pong"  # seen from the real K2 server
    assert strip_reasoning("<ifm|think>plan the diff</ifm|think_faster>\n{\"a\": 1}") == '{"a": 1}'
    assert extract_json('<think>x</think>Here:\n```json\n{"a": [1,2]}\n```') == {"a": [1, 2]}
    assert extract_json('prefix {"scores": [3, 7]} suffix') == {"scores": [3, 7]}
    assert extract_json("no json") is None


def test_claude_json():
    assert parse_claude_json(json.dumps({"type": "result", "subtype": "success", "result": "hi"})) == ("hi", False)
    assert parse_claude_json(json.dumps([{"type": "system"}, {"type": "result", "result": "x", "is_error": True}])) == ("x", True)


def test_rules():
    assert classify_test_rule("E   ModuleNotFoundError: No module named 'x'") == "import_dependency"
    assert classify_test_rule("SyntaxError: invalid syntax") == "compile_syntax"
    assert classify_test_rule("E  AssertionError: 1 != 2\n1 failed") == "assertion_logic"
    assert classify_test_rule("TIMEOUT: checks exceeded 900s") == "timeout_performance"
    base = {"max_depth": 2, "escalations_left": 1, "time_left_min": 40}
    assert retry_escalate_rule({**base, "level": 1, "error_changed": False}) == "retry_local"
    assert retry_escalate_rule({**base, "level": 2, "error_changed": True}) == "escalate_claude"
    assert retry_escalate_rule({**base, "level": 2, "error_changed": True, "escalations_left": 0}) == "ask_human"
    st = compact_state({"a": "x" * 5000, "b": "short"}, 1800)
    assert len(json.dumps(st)) <= 1800 and st["b"] == "short"


class FakeLaya:
    def __init__(self, choice, conf):
        self.choice, self.conf = choice, conf

    def predict(self, state, questions):
        qid = next(iter(questions))
        return {qid: {"choice": self.choice, "answer_confidence": self.conf, "probabilities": {}}}


def _cfg(tmp_path, mode="shadow"):
    import copy
    c = copy.deepcopy(DEFAULTS)
    c["paths"]["data_dir"] = str(tmp_path)
    c["laya"]["mode"] = {k: mode for k in c["laya"]["mode"]}
    c["semif"]["enabled"] = False  # never call a real model server from unit tests
    return c


def test_decider_shadow_and_active(tmp_path):
    d = Decider(_cfg(tmp_path), "run1", client=FakeLaya("environment_flaky", 0.95))
    used, rid = d.classify_failure("AssertionError", ["python"], "bash .k2loop/check.sh")
    assert used == "assertion_logic"  # shadow: rule wins
    d.outcome(rid, "test_classification", "assertion_logic")
    rec = json.loads((tmp_path / "laya" / "test_classification.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert rec["laya"] == "environment_flaky" and rec["rule"] == "assertion_logic"
    d2 = Decider(_cfg(tmp_path, "active"), "run2", client=FakeLaya("environment_flaky", 0.95))
    assert d2.classify_failure("AssertionError", [], "x")[0] == "environment_flaky"
    d3 = Decider(_cfg(tmp_path, "active"), "run3", client=FakeLaya("environment_flaky", 0.30))
    assert d3.classify_failure("AssertionError", [], "x")[0] == "assertion_logic"  # low confidence -> rule
    hyps = [{"hypothesis": f"h{i}"} for i in range(3)]
    kept = Decider(_cfg(tmp_path), "r", client=FakeLaya("try", 0.9)).prune(hyps, [1, 9, 5], {}, 2, 0.0)
    assert [h["hypothesis"] for h, _ in kept] == ["h1", "h2"]


def test_strong_model_on_disk(tmp_path):
    from k2loop.cli import strong_model_on_disk
    c = _cfg(tmp_path)
    c["strong"]["model"] = "qwen3.6-35b-a3b"
    assert not strong_model_on_disk(c)
    (tmp_path / "models-moe" / "IQ4_XS").mkdir(parents=True)  # hf download keeps the repo's sub-folder
    (tmp_path / "models-moe" / "IQ4_XS" / "Qwen3.6-35B-A3B-IQ4_XS.gguf").write_bytes(b"x")
    assert strong_model_on_disk(c)
    c["strong"]["model"] = "qwen3.8-27b"
    assert not strong_model_on_disk(c)


def test_tool_path_added(tmp_path, monkeypatch):
    from k2loop import config
    home = tmp_path / "home"
    (home / ".local" / "bin").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setattr(config, "TOOL_DIRS", ["~/.local/bin", str(tmp_path / "missing")])
    config.ensure_tool_path()
    import os
    assert os.environ["PATH"].split(os.pathsep)[0] == str(home / ".local" / "bin")
    config.ensure_tool_path()  # idempotent
    assert os.environ["PATH"].count(".local") == 1


def test_semif_option_probs_and_prompt():
    from k2loop.semif import build_messages, option_probs
    msgs, ids = build_messages({"log_tail": "ModuleNotFoundError"}, "Which category?",
                               {"compile_syntax": "does not compile", "import_dependency": "module missing"})
    assert ids == ["compile_syntax", "import_dependency"]
    assert "A) compile_syntax" in msgs[1]["content"] and "B) import_dependency" in msgs[1]["content"]
    import math
    top = [{"token": "B", "logprob": math.log(0.6)}, {"token": " A", "logprob": math.log(0.2)},
           {"token": "The", "logprob": math.log(0.1)}, {"token": "Z", "logprob": math.log(0.1)}]
    p = option_probs(top, ids)
    assert abs(p["import_dependency"] - 0.75) < 1e-6 and abs(p["compile_syntax"] - 0.25) < 1e-6
    assert option_probs([{"token": "hello", "logprob": 0}], ids) == {}


class FakeSemIf:
    def __init__(self, choice, conf):
        self.choice, self.conf = choice, conf

    def predict(self, state, question, options):
        assert self.choice in options and question
        return self.choice, self.conf, {self.choice: self.conf}


def test_decider_semif_mode_and_eval(tmp_path):
    c = _cfg(tmp_path, "semif")
    d = Decider(c, "r", client=FakeLaya("environment_flaky", 0.99), semif=FakeSemIf("import_dependency", 0.9))
    used, rid = d.classify_failure("AssertionError", [], "x")
    assert used == "import_dependency"  # semif mode: SemIf decides, Laya only logged
    rec = json.loads((tmp_path / "laya" / "test_classification.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert rec["laya"] == "environment_flaky" and rec["semif"] == "import_dependency" and rec["semif_conf"] == 0.9
    low = Decider(c, "r", client=FakeLaya("try", 0.1), semif=FakeSemIf("import_dependency", 0.2))
    assert low.classify_failure("AssertionError", [], "x")[0] == "assertion_logic"  # low confidence -> rule
    from k2loop.cli import threshold_scan
    rows = [({"rule": "a", "semif": "b", "semif_conf": 0.9}, "b")] * 8 + [({"rule": "a", "semif": "b", "semif_conf": 0.55}, "a")] * 2
    rule_acc, pure, t, acc = threshold_scan(rows, "semif")
    assert rule_acc == 0.2 and pure == 0.8 and t is not None and 0.55 < t <= 0.9 and acc == 1.0


def test_testrun_helpers(tmp_path):
    out = "E  AssertionError\n=== 3 failed, 10 passed in 1.2s\nerror: boom at /a/b/c.py:12"
    assert count_failures(out) == 3
    assert signature(out) == signature(out.replace("12", "99"))
    assert "AssertionError" in log_tail(out)
    (tmp_path / ".k2loop").mkdir()
    (tmp_path / ".k2loop" / "check.sh").write_text("echo hi; exit 3", encoding="utf-8")
    r = run_checks(tmp_path, 30)
    assert not r.passed and r.exit_code == 3


def test_project_and_check_sh(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "platformio.ini").write_text("[env:esp32]\n[env:native]\nplatform=native\n", encoding="utf-8")
    info = detect(tmp_path)
    assert "python" in info["types"] and "platformio" in info["types"]
    names = [n for n, _ in info["checks"]]
    assert "tests" in names and "build" in names
    sh = render_check_sh(info["checks"])
    assert "pio test -e native" in sh and "$PYTEST -x -q" in sh and "uv run --quiet --with pytest" in sh
    assert subprocess.run(["bash", "-n"], input=sh, text=True).returncode == 0
    assert "K2LOOP_NO_CHECKS" in render_check_sh([])


def test_git_worktree_roundtrip(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    g = lambda *a: subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True, text=True).stdout  # noqa: E731
    g("init", "-q")
    (repo / "a.txt").write_text("one\n", encoding="utf-8")
    g("add", "-A")
    g("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init")
    wt = gitops.new_worktree(repo, tmp_path / "wt" / "c1")
    (wt / "a.txt").write_text("one\ntwo\n", encoding="utf-8")
    (wt / "b.txt").write_text("new\n", encoding="utf-8")
    (wt / "node_modules").mkdir()
    (wt / "node_modules" / "junk.js").write_text("x", encoding="utf-8")
    (wt / ".aider.chat.history.md").write_text("x", encoding="utf-8")
    diff = gitops.snapshot_diff(wt)
    assert "b.txt" in diff and "node_modules" not in diff and ".aider" not in diff
    assert gitops.diff_lines(diff) == 2
    wt2 = gitops.new_worktree(repo, tmp_path / "wt" / "c2", diff)
    assert (wt2 / "b.txt").read_text(encoding="utf-8") == "new\n"
    gitops.apply_diff(repo, diff)
    files = gitops.commit_all(repo, "k2loop: test")
    assert sorted(files) == ["a.txt", "b.txt"]
    gitops.remove_worktree(repo, wt)
    gitops.remove_worktree(repo, wt2)
    assert not wt.exists()


def test_planner_output_valid():
    alt = {"stages": [
        {"name": "Core Logic", "steps": [{"title": "Add add()", "description": "d", "role": "backend",
                                          "scope": "small", "complexity": "low", "files": ["src/calc.py"]}]},
        {"name": "Docs", "depends_on": ["Core Logic"], "steps": [{"title": "README", "role": "weird"}]}]}
    plan = to_bernstein_yaml("Build a calculator", alt, "k2-horizon-7b")
    assert validate_plan(plan) == []
    assert plan["stages"][1]["depends_on"] == ["core-logic"]
    assert plan["stages"][1]["steps"][0]["role"] == "backend"
    bad = {"stages": [{"name": "a", "depends_on": ["zzz"], "steps": [{"title": "t", "role": "backend", "cli": "x"}]}]}
    errs = validate_plan(bad)
    assert any("unknown keys" in e for e in errs) and any("depends_on" in e for e in errs)


def test_bernstein_yaml_is_yaml():
    data = yaml.safe_load(bernstein_yaml("k2-horizon-7b", 80))
    assert data["cli"] == "generic" and data["merge_strategy"] == "direct"
    assert data["role_model_policy"]["backend"] == {"cli": "generic", "model": "k2-horizon-7b"}
    assert data["tuning"]["task"]["scope_timeout_s"]["medium"] >= 6600


def test_commit_with_gitignored_sdd(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    g = lambda *a: subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True, text=True).stdout  # noqa: E731
    g("init", "-q")
    (repo / ".gitignore").write_text(".sdd/\n", encoding="utf-8")
    g("add", "-A")
    g("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init")
    (repo / ".sdd" / "runtime").mkdir(parents=True)
    (repo / ".sdd" / "runtime" / "x.log").write_text("log", encoding="utf-8")
    (repo / "node_modules").symlink_to(tmp_path)
    (repo / "f.py").write_text("x = 1\n", encoding="utf-8")
    assert gitops.commit_all(repo, "t") == ["f.py"]


REAL_PROMPT_3_20 = """## Tasks
### Task 1: Implement add with tests (id=d1740b1eccfa)
Create src/calc.py with add(a, b).
Files: src/calc.py, tests/test_calc.py

## Instructions
**Step 2: Mark tasks complete on the task server**
```bash
bernstein task complete d1740b1eccfa --summary "Completed: Implement add with tests"
```
## Signal files (check periodically)
cat .sdd/runtime/signals/backend-ac37197d/WAKEUP 2>/dev/null
## Task Server Authentication
-H "Authorization: Bearer $(cat TOKPATH)"
Marking a task complete - use the first-class CLI, NOT curl.
bernstein task complete <TASK_ID> --summary "Done"
"""


def test_real_bernstein_320_prompt(tmp_path):
    root = tmp_path / "proj"
    (root / ".sdd" / "runtime" / "agent_tokens").mkdir(parents=True)
    (root / ".sdd" / "backlog").mkdir()
    tok = root / ".sdd" / "runtime" / "agent_tokens" / "backend-ac37197d.token"
    tok.write_text("secret-token\n", encoding="utf-8")
    wt = root / ".sdd" / "worktrees" / "backend-ac37197d"
    wt.mkdir(parents=True)
    info = parse_prompt(REAL_PROMPT_3_20.replace("TOKPATH", str(tok)), wt)
    assert info.task_ids == ["d1740b1eccfa"]
    assert info.token == "secret-token"
    assert info.root == root
    assert info.session_id == "backend-ac37197d"
    assert info.heartbeat_paths == [root / ".sdd" / "runtime" / "heartbeats" / "backend-ac37197d.json"]
    txt = clean_task_text(REAL_PROMPT_3_20)
    assert "Create src/calc.py" in txt


def test_commit_staged_ignores_untracked_junk(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    g = lambda *a: subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True, text=True).stdout  # noqa: E731
    g("init", "-q")
    (repo / "a.py").write_text("a = 1\n", encoding="utf-8")
    g("add", "-A")
    g("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init")
    wt = gitops.new_worktree(repo, tmp_path / "wt")
    (wt / "a.py").write_text("a = 2\n", encoding="utf-8")
    (wt / "__pycache__").mkdir()
    (wt / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"x")
    diff = gitops.snapshot_diff(wt)
    assert "__pycache__" not in diff
    (repo / "AGENTS.md").write_text("orchestrator junk", encoding="utf-8")   # untracked file dropped by a tool
    gitops.apply_diff(repo, diff)
    assert gitops.commit_staged(repo, "k2loop: t") == ["a.py"]
    assert "AGENTS.md" in g("status", "--porcelain")


def test_repo_context_and_local_config(tmp_path):
    from k2loop.planner import repo_context
    repo = tmp_path / "r"
    (repo / "src").mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "README.md").write_text("# Calc\nA calculator.", encoding="utf-8")
    (repo / "src" / "calc.py").write_text("def add(a, b):\n    return a + b\n\nclass Calc:\n    pass\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    ctx = repo_context(repo)
    assert "src/calc.py" in ctx and "def add(a, b):" in ctx and "A calculator." in ctx
    from k2loop.config import load_config
    p = tmp_path / "c.yaml"
    p.write_text(f"mode: local\npaths: {{data_dir: {tmp_path}}}\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg["mode"] == "local" and cfg["tot"]["task_time_budget_min"] >= 200
    from k2loop.llm import LLM
    assert LLM(cfg, "strong").base == LLM(cfg, "llm").base == "http://127.0.0.1:8080/v1"  # llama-swap
    p.write_text("strong: {base_url: 'http://127.0.0.1:8081/v1'}\n", encoding="utf-8")  # 0.1 config files name the old ports
    assert load_config(p)["strong"]["base_url"] == "http://127.0.0.1:8080/v1"


def test_local_plan_prompt_survives_braces(tmp_path, monkeypatch):
    import contextlib
    from k2loop import planner
    repo = tmp_path / "r"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "check.sh").write_text('x="${A:-1}"; { echo hi; }\n', encoding="utf-8")
    (repo / "package.json").write_text('{"scripts": {"test": "jest"}}', encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    seen = []

    class FakeLLM:
        def __init__(self, *a, **k):
            pass

        def ask_json(self, prompt, effort, **kw):
            seen.append(prompt)
            if "candidate plans" in prompt:
                return {"scores": [1, 8], "best_index": 1}
            return {"alternatives": [{"stages": [{"name": "s", "steps": [
                {"title": f"t{len(seen)}", "description": "d", "role": "backend"}]}]}]}

    monkeypatch.setattr("k2loop.llm.LLM", FakeLLM)
    monkeypatch.setattr("k2loop.swap.strong_model", lambda cfg, log=print: contextlib.nullcontext(True))
    import copy
    from k2loop.config import DEFAULTS
    cfg = copy.deepcopy(DEFAULTS)
    cfg["strong"]["plan_samples"] = 2
    alts, best, _ = planner._local_alternatives(cfg, "goal {with braces}", repo, log=lambda *_: None)
    assert len(alts) == 2 and best == 1
    assert '{"scripts": {"test": "jest"}}' in seen[0] and "goal {with braces}" in seen[0]


def _mini_repo(tmp_path):
    repo = tmp_path / "r"
    (repo / "src").mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "src" / "auth.py").write_text("SECRET_KEY = 'x'\n\ndef login(user):\n    return check_password(user)\n" + "# pad\n" * 5000, encoding="utf-8")
    (repo / "src" / "db.py").write_text("def check_password(user):\n    return True\n", encoding="utf-8")
    (repo / "README.md").write_text("# App", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    return repo


def test_read_files_and_grep_are_safe_and_bounded(tmp_path):
    from k2loop.planner import git_grep, read_files
    repo = _mini_repo(tmp_path)
    (tmp_path / "outside.txt").write_text("nope", encoding="utf-8")
    out = read_files(repo, ["src/auth.py", "../outside.txt", "/etc/passwd", "src/db.py"], budget=3000, per_file=1000)
    assert "=== FILE: src/auth.py" in out and "truncated" in out and "outside" not in out and "root:" not in out
    assert len(out) <= 3200
    g = git_grep(repo, ["check_password", "nothing_here_xyz"], 5000)
    assert "src/db.py:1:" in g and "(no matches)" in g


def test_explore_two_rounds(tmp_path):
    import copy
    from k2loop.config import DEFAULTS
    from k2loop.planner import explore
    repo = _mini_repo(tmp_path)
    prompts = []

    class FakeLLM:
        def ask_json(self, prompt, effort, **kw):
            prompts.append(prompt)
            if len(prompts) == 1:
                return {"files": ["src/auth.py", "does/not/exist.py"], "search": ["check_password"], "done": False}
            return {"files": ["src/db.py"], "search": [], "done": True}

    cfg = copy.deepcopy(DEFAULTS)
    block = explore(FakeLLM(), cfg, "add 2FA", repo, "OVERVIEW", 20000, log=lambda *_: None)
    assert "=== FILE: src/auth.py" in block and "=== FILE: src/db.py" in block and "=== SEARCH: check_password" in block
    assert "does/not/exist" not in block
    assert "YOU ALREADY READ (src/auth.py)" in prompts[1]
    cfg["strong"]["read_files"] = False
    assert explore(FakeLLM(), cfg, "g", repo, "O", 20000, log=lambda *_: None) == ""


def test_input_budget_scales_with_context():
    import copy
    from k2loop.config import DEFAULTS
    from k2loop.planner import input_budget_chars
    cfg = copy.deepcopy(DEFAULTS)
    cfg["strong"]["context"] = 32768
    small = input_budget_chars(cfg)
    cfg["strong"]["context"] = 65536
    assert input_budget_chars(cfg) > small >= 12000
