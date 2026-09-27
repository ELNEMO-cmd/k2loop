"""Regressions: agent loop survives SystemExit, malformed model JSON, protected check.sh, process-group
timeouts, cancellation never commits."""
import subprocess
import threading
import time
from pathlib import Path
from unittest import mock

from k2loop import gitops
from k2loop.agent import Agent
from k2loop.config import load_config
from k2loop.engine import Engine
from k2loop.planner import _dict_alts, validate_plan
from k2loop.testrun import run_checks, run_group


def _cfg(tmp_path):
    cfg = load_config(tmp_path / "none.yaml")
    cfg["paths"]["data_dir"] = str(tmp_path / "data")
    cfg["trace"]["enabled"] = False
    return cfg


def _repo(tmp_path, check_sh="exit 1\n"):
    r = tmp_path / "repo"
    (r / ".k2loop").mkdir(parents=True)
    (r / ".k2loop" / "check.sh").write_text(check_sh)
    (r / "a.py").write_text("x = 1\n")
    g = lambda *a: subprocess.run(["git", *a], cwd=r, check=True, capture_output=True, text=True)  # noqa: E731
    g("init", "-q")
    g("add", "-A")
    g("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init")
    return r


def test_agent_loop_survives_systemexit(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr("k2loop.agent.load_config", lambda: cfg)
    ag = Agent(cfg)

    def boom(g):
        raise SystemExit("planner: model açılamadı")

    with mock.patch("k2loop.agent.Agent.intent", side_effect=SystemExit("x")):
        ag.submit("hedef 1")
        ag.handle(ag.q.next(timeout=0.1))  # handle itself must not let SystemExit out
    assert [g.status for g in ag.q.snapshot()] == ["failed"]

    ag.handle = boom  # even if handle lets something through, the loop keeps going
    t = threading.Thread(target=ag.run_forever, daemon=True)
    t.start()
    ag.submit("hedef 2")
    for _ in range(50):
        if ag.q.snapshot()[-1].status != "queued" and ag.q.snapshot()[-1].status != "running":
            break
        time.sleep(0.1)
    assert t.is_alive()
    assert ag.q.snapshot()[-1].status == "failed"
    ag.stopped.set()


def test_intent_tolerates_json_list(tmp_path):
    ag = Agent(_cfg(tmp_path))
    with mock.patch("k2loop.agent.LLM") as L:
        L.return_value.ask_json.return_value = ["code"]
        assert ag.intent("almanca çalışma planı yap")["kind"] == "plan"


def _bare_engine(reviewer_answers):
    eng = Engine.__new__(Engine)
    eng.tot, eng.task, eng.eff, eng.local = {"review_votes": len(reviewer_answers)}, "t", {"review": "low"}, False
    eng.cfg = {"claude": {"enabled": False}}
    eng.left, eng.phase, eng.emit, eng.notes = (lambda: 9999), (lambda *a: None), (lambda *a, **k: None), []
    eng.reviewer = mock.Mock(ask_json=mock.Mock(side_effect=reviewer_answers))
    eng._fix_round = mock.Mock(side_effect=lambda win, issues, tag: win)
    return eng


def test_review_ignores_malformed_answers():
    win = mock.Mock(diff="+x")
    eng = _bare_engine([[{"issue": "x"}], None, {"verdict": "changes", "issues": "not a list"}])
    assert eng.review(win) is win
    assert "K2 review votes: ['changes']" in eng.notes  # list / None are not votes, not silent approvals
    eng._fix_round.assert_called_once()


def test_candidate_cannot_edit_check_sh(tmp_path):
    repo = _repo(tmp_path)
    wt = gitops.new_worktree(repo, tmp_path / "wt")
    (wt / ".k2loop" / "check.sh").write_text("exit 0\n")  # the model "fixes" the judge
    (wt / "a.py").write_text("x = 2\n")
    diff = gitops.snapshot_diff(wt)
    assert "check.sh" not in diff and "a.py" in diff
    assert gitops.restore_protected(wt)
    assert not run_checks(wt, 30).passed
    gitops.remove_worktree(repo, wt)


def test_timeout_kills_process_group(tmp_path):
    marker = f"k2loop_orphan_{time.time_ns()}"
    script = f"exec -a {marker} sleep 300 & sleep 300; wait\n"
    rc, _, _, timed_out = run_group(["bash", "-c", script], tmp_path, 1)
    assert rc == 124 and timed_out
    time.sleep(0.5)
    assert marker not in subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout


def test_cancel_never_commits(tmp_path):
    repo = _repo(tmp_path, "exit 0\n")
    cfg = _cfg(tmp_path)
    hb = mock.Mock(shutdown=False)
    eng = Engine(cfg, "task", repo, "r1", hb=hb, log=lambda m: None, emit=lambda *a, **k: None)
    head = gitops.git(["rev-parse", "HEAD"], repo)

    def fake_aider(cfg, d, *a, **k):
        (Path(d) / "a.py").write_text("x = 3\n")
        hb.shutdown = True  # /iptal arrives while the first candidate is written
        return True, "ok"

    with mock.patch("k2loop.engine.run_aider", fake_aider):
        res = eng.run()
    assert res.status == "human" and "cancelled" in res.summary
    assert gitops.git(["rev-parse", "HEAD"], repo) == head


def test_planner_drops_malformed_alternatives():
    assert _dict_alts({"alternatives": ["x", {"stages": []}, {"name": "n"}]}) == [{"stages": []}]
    assert _dict_alts(["x"]) == [] and _dict_alts({"alternatives": {"a": 1}}) == []
    assert validate_plan({"stages": ["x", {"name": "s", "steps": ["y"]}]})  # errors, no crash


def test_secrets_quotes_stripped(tmp_path):
    (tmp_path / "c.yaml").write_text("{}\n")
    (tmp_path / "secrets.env").write_text('LDR_USER="ali"\nLDR_PASSWORD=\'p w\'\n')
    cfg = load_config(tmp_path / "c.yaml")
    assert cfg["research"]["ldr_user"] == "ali" and cfg["research"]["ldr_password"] == "p w"


# ---------------------------------------------------------------- goal loop (rounds until done)
ENGINE_CFGS: list[dict] = []


RESEARCH_QUERIES: list[str] = []


def _loop_agent(tmp_path, monkeypatch, step_results, verdicts, max_rounds=0, check_sh="exit 0\n", on_done=None):
    """step_results: per round, the statuses of that round's steps; verdicts: answers of the goal check."""
    import json

    import yaml

    from k2loop.engine import Result
    cfg = _cfg(tmp_path)
    cfg["agent"].update(research_first=False, max_rounds=max_rounds)
    monkeypatch.setattr("k2loop.agent.load_config", lambda: cfg)
    repo = _repo(tmp_path, check_sh)
    ag = Agent(cfg)
    goals_seen: list[str] = []
    rounds = iter(step_results)

    def fake_plan(cfg, goal, root, log=print, research=""):
        goals_seen.append(goal)
        statuses = next(rounds)
        p = tmp_path / f"plan{len(goals_seen)}.yaml"
        p.write_text(yaml.safe_dump({"name": "p", "stages": [{"name": "s", "steps": [
            {"title": f"r{len(goals_seen)}-step{i}", "description": "d", "role": "backend", "_st": st}
            for i, st in enumerate(statuses)]}]}))
        return p

    class FakeEngine:
        def __init__(self, cfg, task, root, run_id, **kw):
            self.title = kw["title"]
            ENGINE_CFGS.append(cfg)

        def run(self):
            plan = yaml.safe_load((tmp_path / f"plan{len(goals_seen)}.yaml").read_text())
            st = next(s["_st"] for s in plan["stages"][0]["steps"] if s["title"] == self.title)
            if st == "done":
                if on_done:
                    on_done(repo)
                return Result(st, f"summary of {self.title}")
            patch = tmp_path / f"{self.title}.patch"
            patch.write_text(f"+ attempt of {self.title}\n")
            return Result(st, f"summary of {self.title}", patch_path=str(patch),
                          check_tail="E   AssertionError: 3 != 4\nFAILED tests/test_x.py::test_rows")

    answers = iter(verdicts)
    monkeypatch.setattr("k2loop.planner.make_plan", fake_plan)
    monkeypatch.setattr("k2loop.planner.summarize", lambda p: "plan")
    monkeypatch.setattr("k2loop.engine.Engine", FakeEngine)
    monkeypatch.setattr(ag, "_model_call", lambda profile, prompt, **kw: (
        a if isinstance(a := next(answers), str) else json.dumps(a)))
    ENGINE_CFGS.clear()
    RESEARCH_QUERIES.clear()

    class FakeResearcher:
        def run(self, q, iterations=None):
            RESEARCH_QUERIES.append(q)
            return {"summary": "use csv.Sniffer [1]", "sources": []}

    monkeypatch.setattr(ag, "_researcher", lambda: FakeResearcher())
    g = ag.submit(f"{repo} içine özellik ekle")
    ag.q.next(0.1)
    status, text = ag.code_goal(g, {"kind": "code", "project": str(repo), "topic": "x"})
    return status, text, goals_seen


def test_goal_loop_replans_after_failed_step(tmp_path, monkeypatch):
    status, text, goals = _loop_agent(tmp_path, monkeypatch, [["done", "human", "done"], ["done"]],
                                      [{"done": True, "remaining": [], "why": "hepsi var"}])
    assert status == "done" and "Hedef tamamlandı" in text and len(goals) == 2
    assert "r1-step0" in goals[1] and "FAILED ATTEMPTS" in goals[1] and "r1-step1" in goals[1]
    assert "⏸ r1-step2" in text  # skipped after the failure, re-planned in round 2


def test_goal_loop_continues_when_verifier_finds_missing_work(tmp_path, monkeypatch):
    status, text, goals = _loop_agent(tmp_path, monkeypatch, [["done"], ["done"]], [
        {"done": False, "remaining": ["CLI flag --out yok"], "why": "eksik"},
        {"done": True, "remaining": [], "why": "tamam"}])
    assert status == "done" and len(goals) == 2
    assert "STILL MISSING" in goals[1] and "--out" in goals[1]


def test_goal_loop_keeps_going_with_more_effort(tmp_path, monkeypatch):
    status, text, goals = _loop_agent(tmp_path, monkeypatch, [["human"], ["failed"], ["done"]],
                                      [{"done": True, "remaining": [], "why": "tamam"}])
    assert status == "done" and len(goals) == 3  # two failed rounds are not a reason to give up
    tots = [c["tot"] for c in ENGINE_CFGS]
    assert [t["candidates"] for t in tots] == [3, 4, 5]
    assert [t["debug_depth"] for t in tots] == [2, 3, 4]
    assert tots[2]["task_time_budget_min"] == 3 * tots[0]["task_time_budget_min"]
    assert ENGINE_CFGS[2]["strong"]["max_escalations_per_task"] == 3


def test_goal_loop_stops_only_when_really_stuck(tmp_path, monkeypatch):
    status, text, goals = _loop_agent(tmp_path, monkeypatch, [["human"]] * 5, [])
    assert status == "failed" and len(goals) == 3 and "insan gerekli" in text


def test_unclear_verifier_is_asked_again_and_never_means_done(tmp_path, monkeypatch):
    status, text, goals = _loop_agent(tmp_path, monkeypatch, [["done"]], ["???", "hmm", {"x": 1}])
    assert status == "failed" and "doğrulanamadı" in text
    (tmp_path / "b").mkdir()
    status, text, goals = _loop_agent(tmp_path / "b", monkeypatch, [["done"]],
                                      ["???", {"done": True, "remaining": [], "why": "tamam"}])
    assert status == "done"


def test_goal_loop_round_limit(tmp_path, monkeypatch):
    status, text, goals = _loop_agent(tmp_path, monkeypatch, [["done"], ["done"]],
                                      [{"done": False, "remaining": ["a"], "why": ""}] * 2, max_rounds=2)
    assert status == "failed" and "tur sınırı (2)" in text and "Eksik kalanlar" in text


def test_goal_loop_verifier_ping_pong_is_stopped(tmp_path, monkeypatch):
    same = [{"done": False, "remaining": ["--out bayrağı yok"], "why": ""},
            {"done": False, "remaining": ["--out bayrağı eksik"], "why": ""}] * 5
    status, text, goals = _loop_agent(tmp_path, monkeypatch, [["done"]] * 10, same)
    assert status == "failed" and len(goals) == 4 and "ilerleme olmadı" in text  # 1 new + 3 repeats
    assert ENGINE_CFGS[-1]["tot"].get("codegen_profile") == "strong"   # repeated item: the strong model codes
    assert any("--out" in q for q in RESEARCH_QUERIES)                  # ...and the missing piece is researched



def test_goal_loop_stops_when_verifier_keeps_inventing_new_items(tmp_path, monkeypatch):
    new_each_round = [{"done": False, "remaining": [w], "why": ""} for w in (
        "logging", "docstrings", "type hints", "README section", "CLI --verbose", "retry policy", "dark mode", "man page")]
    status, text, goals = _loop_agent(tmp_path, monkeypatch, [["done"]] * 8, new_each_round)
    assert status == "failed" and len(goals) == 6 and "yeni eksik buldu" in text  # agent.max_verify_rounds (6)

def test_goal_loop_researches_failure_and_carries_closest_attempt(tmp_path, monkeypatch):
    status, text, goals = _loop_agent(tmp_path, monkeypatch, [["done", "human"], ["done"]],
                                      [{"done": True, "remaining": [], "why": "tamam"}])
    assert status == "done"
    assert "AssertionError: 3 != 4" in RESEARCH_QUERIES[0] and "r1-step1" in RESEARCH_QUERIES[0]
    assert "CLOSEST FAILED ATTEMPT" in goals[1] and "attempt of r1-step1" in goals[1]
    assert "use csv.Sniffer" in goals[1]


def test_goal_loop_without_tests_asks_for_a_test_setup_first(tmp_path, monkeypatch):
    status, text, goals = _loop_agent(tmp_path, monkeypatch, [["done"]],
                                      [{"done": True, "remaining": [], "why": "tamam"}])
    assert "NO automated tests" in goals[0]


def _add_tests(repo):
    (repo / "tests").mkdir(exist_ok=True)
    (repo / "tests" / "test_a.py").write_text("def test_a():\n    assert True\n")


def test_generated_check_sh_follows_new_tests(tmp_path, monkeypatch):
    from k2loop.project import detect, render_check_sh
    empty = tmp_path / "e"
    empty.mkdir()
    auto = render_check_sh(detect(empty)["checks"])  # what k2loop writes for a project without tests
    _loop_agent(tmp_path, monkeypatch, [["done"]], [{"done": True, "remaining": [], "why": "tamam"}],
                check_sh=auto, on_done=_add_tests)
    chk = (tmp_path / "repo" / ".k2loop" / "check.sh").read_text()
    assert "pytest" in chk and "K2LOOP_NO_CHECKS" not in chk
    log = subprocess.run(["git", "log", "--format=%s"], cwd=tmp_path / "repo", capture_output=True, text=True).stdout
    assert "regenerate .k2loop/check.sh" in log


def test_user_edited_check_sh_is_never_regenerated(tmp_path, monkeypatch):
    _loop_agent(tmp_path, monkeypatch, [["done"]], [{"done": True, "remaining": [], "why": "tamam"}],
                check_sh="make my-own-checks\n", on_done=_add_tests)
    assert (tmp_path / "repo" / ".k2loop" / "check.sh").read_text() == "make my-own-checks\n"


def test_engine_lets_strong_model_write_code_when_boosted(tmp_path, monkeypatch):
    import contextlib
    repo = _repo(tmp_path, "exit 0\n")
    cfg = _cfg(tmp_path)
    cfg["tot"]["candidates"] = 1
    cfg["tot"]["codegen_profile"] = "strong"
    cfg["strong"]["final_review"] = "none"
    cfg["tot"]["review_votes"] = 0
    used = []
    monkeypatch.setattr("k2loop.swap.strong_model", lambda cfg, log=print: contextlib.nullcontext(True))

    def fake_aider(cfg, d, msg, effort, timeout, files=None, profile="llm"):
        used.append(profile)
        (Path(d) / "a.py").write_text("x = 5\n")
        return True, "ok"

    monkeypatch.setattr("k2loop.engine.run_aider", fake_aider)
    res = Engine(cfg, "task", repo, "r2", log=lambda m: None, emit=lambda *a, **k: None).run()
    assert res.status == "done" and used == ["strong"]



def test_without_project_checks_candidates_are_judged_by_their_own_tests(tmp_path, monkeypatch):
    """First step of a project without tests: a candidate adding a passing test must win over a smaller one that
    adds none, and one adding a failing test must lose (before: all three 'passed' and the smallest won)."""
    repo = _repo(tmp_path, 'echo "K2LOOP_NO_CHECKS"\nexit 0\n')
    cfg = _cfg(tmp_path)
    cfg["tot"]["candidates"] = 3
    cfg["tot"]["codegen_profile"] = "llm"
    cfg["tot"]["overlap_checks"] = False
    cfg["strong"]["final_review"] = "none"
    cfg["tot"]["review_votes"] = 0
    n = iter(range(3))

    def fake_aider(cfg, d, msg, effort, timeout, files=None, profile="llm"):
        d, i = Path(d), next(n)
        (d / "a.py").write_text("x = 2\n")
        if i < 2:
            (d / "pyproject.toml").write_text('[project]\nname = "a"\nversion = "0"\n\n'
                                              '[tool.pytest.ini_options]\npythonpath = ["."]\n')
            (d / "tests").mkdir()
            (d / "tests" / "test_a.py").write_text(f"import a\n\n\ndef test_a():\n    assert a.x == {3 if i == 0 else 2}\n")
        return True, "ok"

    monkeypatch.setattr("k2loop.engine.run_aider", fake_aider)
    events = []
    res = Engine(cfg, "set up tests", repo, "r3", log=lambda m: None, emit=lambda k, **kw: events.append((k, kw))).run()
    cands = {kw["name"]: kw for k, kw in events if k == "cand"}
    assert not cands["gen1"]["passed"]                                         # its own test fails
    assert cands["gen2"]["passed"] and not cands["gen2"]["no_checks"]          # its own test passes
    assert cands["gen3"]["passed"] and cands["gen3"]["no_checks"]              # nothing to run
    assert res.status == "done" and "tests/test_a.py" in res.files             # the tested one won



def test_candidate_that_only_creates_an_empty_file_is_not_a_winner(tmp_path, monkeypatch):
    repo = _repo(tmp_path, "exit 0\n")
    cfg = _cfg(tmp_path)
    cfg["tot"]["candidates"] = 1
    cfg["tot"]["codegen_profile"] = "llm"
    cfg["tot"]["debug_depth"] = 0
    cfg["strong"]["final_review"] = "none"
    cfg["strong"]["max_escalations_per_task"] = 0
    cfg["claude"]["enabled"] = False
    cfg["tot"]["review_votes"] = 0

    def fake_aider(cfg, d, msg, effort, timeout, files=None, profile="llm"):
        (Path(d) / "agent.py").write_text("")          # the stub answer: an empty new module
        return False, "timed out"

    monkeypatch.setattr("k2loop.engine.run_aider", fake_aider)
    res = Engine(cfg, "implement the agent", repo, "r4", log=lambda m: None, emit=lambda *a, **k: None).run()
    assert res.status != "done" and not (repo / "agent.py").exists()


# ---------------------------------------------------------------- review round 2
def test_round_goal_fits_the_planner_context(tmp_path, monkeypatch):
    import contextlib

    from k2loop import planner
    repo = tmp_path / "r"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    for i in range(60):
        (repo / f"m{i}.py").write_text("\n".join(f"def f{j}(x):\n    return x + {j}" for j in range(40)))
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    sizes = []

    class FakeLLM:
        def __init__(self, *a, **k):
            pass

        def ask_json(self, prompt, effort, **kw):
            sizes.append(len(prompt))
            if "candidate plans" in prompt:
                return {"scores": [1, 2, 3], "best_index": 0}
            if "look at the code" in prompt:
                return {"files": [f"m{i}.py" for i in range(12)], "search": ["f1"], "done": True}
            return {"alternatives": [{"name": "a" * 3000, "stages": [{"name": "s", "steps": [
                {"title": "t", "description": "d" * 4000, "role": "backend"}]}]}]}

    monkeypatch.setattr("k2loop.llm.LLM", FakeLLM)
    monkeypatch.setattr("k2loop.swap.strong_model", lambda cfg, log=print: contextlib.nullcontext(True))
    cfg = _cfg(tmp_path)
    goal = "hedef\n" + "- step\n" * 200 + "CLOSEST FAILED ATTEMPT:\n" + "+x\n" * 2500 + "RESEARCH:\n" + "r" * 5000
    planner._local_alternatives(cfg, goal, repo, log=lambda *_: None, research="R" * 20000)
    assert max(sizes) <= planner.input_budget_chars(cfg) + 3000, sizes


def test_fit_text_keeps_head_and_tail():
    from k2loop.planner import fit_text
    t = fit_text("GOAL " + "x" * 1000 + " Plan ONLY the remaining work.", 200)
    assert len(t) <= 200 and t.startswith("GOAL") and t.endswith("remaining work.")


def test_hybrid_mode_verifies_with_claude(tmp_path, monkeypatch):
    import json

    from k2loop.claude import ClaudeResult
    cfg = _cfg(tmp_path)
    cfg["mode"] = "hybrid"
    ag = Agent(cfg)
    repo = _repo(tmp_path, "exit 0\n")
    monkeypatch.setattr(ag, "_model_call", mock.Mock(side_effect=AssertionError("local model must not be used")))
    monkeypatch.setattr("k2loop.claude.run_claude", lambda *a, **k: ClaudeResult(
        True, json.dumps({"done": True, "remaining": [], "why": "ok"})))
    g = ag.submit("x")
    head = gitops.git(["rev-parse", "HEAD"], repo).strip()
    assert ag._verify_goal(g, repo, head, ["a"])["done"] is True


def test_similar_items():
    from k2loop.agent import _similar
    assert _similar("--out bayrağı yok", "--out bayrağı eksik")
    assert _similar("Add tests for the parser", "add tests for the parser.")
    assert not _similar("add tests", "add tests for the --out flag and the csv export")


def test_escalations_do_not_count_against_the_strong_model(tmp_path):
    from k2loop import modelstats
    cfg = _cfg(tmp_path)
    cfg["mode"] = "local"
    for i in range(25):
        modelstats.record(cfg, "strong", kind="codegen", types=["python"], usable=True, passed=i < 15,
                          failures=0, secs=1)
        modelstats.record(cfg, "strong", kind="escalate", types=["python"], usable=True, passed=False,
                          failures=3, secs=1)  # handed K2's hopeless cases
        modelstats.record(cfg, "llm", kind="codegen", types=["python"], usable=True, passed=i < 10,
                          failures=0, secs=1)
    assert modelstats.choose_coder(cfg, ["python"])[0] == "strong"


def test_strong_coder_also_debugs_and_reviews(tmp_path):
    cfg = _cfg(tmp_path)
    cfg["mode"] = "local"
    cfg["tot"]["codegen_profile"] = "strong"
    repo = _repo(tmp_path, "exit 0\n")
    eng = Engine(cfg, "t", repo, "r9", log=lambda m: None, emit=lambda *a, **k: None)
    assert eng.llm.profile == "strong" and eng.reviewer.profile == "strong"
    assert eng.scorer.profile == "helper"  # CPU model stays: no GPU swap
