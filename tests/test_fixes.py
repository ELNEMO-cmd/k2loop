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
