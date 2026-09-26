"""The benchmark must measure something: each task's reference solution passes the hidden tests and the starting
code does not; the runner compares models fairly and records the results."""
import contextlib

import pytest

from k2loop import modelstats
from k2loop.bench import judge, make_repo, pytest_cmd, report, run_bench, wilson
from k2loop.bench_tasks import TASKS
from k2loop.config import load_config


@pytest.mark.parametrize("task", TASKS, ids=[t.name for t in TASKS])
def test_task_is_meaningful(task, tmp_path):
    py = pytest_cmd()
    start = make_repo(task, tmp_path / "start")
    passed, _, _, out = judge(task, start, py)
    assert not passed, f"{task.name}: starting code already passes the hidden tests\n{out}"
    solved = make_repo(task, tmp_path / "solved")
    for rel, text in task.solution.items():
        (solved / rel).write_text(text, encoding="utf-8")
    passed, vis_ok, fails, out = judge(task, solved, py)
    assert passed and vis_ok and fails == 0, f"{task.name}: reference solution fails\n{out}"


def test_run_bench_compares_and_records(tmp_path, monkeypatch):
    cfg = load_config(tmp_path / "none.yaml")
    cfg["paths"]["data_dir"] = str(tmp_path / "data")
    cfg["mode"] = "local"
    monkeypatch.setattr("k2loop.swap.strong_model", lambda cfg, log=print: contextlib.nullcontext(True))
    tasks = {t.name: t for t in TASKS}

    def fake_aider(cfg, repo, msg, effort, timeout, files=None, profile="llm"):
        task = next(t for t in TASKS if repo.name.startswith(t.name + "-"))
        if profile == "llm":                       # "K2" solves everything, "35B" changes nothing
            for rel, text in tasks[task.name].solution.items():
                (repo / rel).write_text(text, encoding="utf-8")
        return True, "ok"

    atts, out_dir = run_bench(cfg, ["llm", "strong"], 2, ["csv_stats", "lru_cache"], log=lambda m: None,
                              aider=fake_aider)
    assert len(atts) == 8
    assert all(a.passed for a in atts if a.profile == "llm")
    assert not any(a.passed or a.usable for a in atts if a.profile == "strong")
    text = (out_dir / "report.txt").read_text()
    assert "Fark anlamlı" in text and cfg["llm"]["model"] in text and "4/4" in text
    assert (out_dir / "attempts.json").exists() and not (out_dir / "work").exists()
    s = modelstats.summary(cfg)
    assert s[cfg["llm"]["model"]]["passed"] == 4 and s[cfg["strong"]["model"]]["passed"] == 0


def test_model_cannot_fake_the_hidden_tests(tmp_path):
    task = next(t for t in TASKS if t.name == "lru_cache")
    repo = make_repo(task, tmp_path / "r")
    for rel in task.hidden:  # the model writes a trivially passing file with the hidden test's name
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text("def test_ok():\n    assert True\n")
    assert not judge(task, repo, pytest_cmd())[0]


def test_wilson_and_report_small_sample():
    lo, hi = wilson(2, 3)
    assert 0 < lo < 0.67 < hi < 1
    assert wilson(0, 0) == (0.0, 0.0)
    from k2loop.bench import Attempt
    atts = [Attempt("t", "easy", p, m, 1, ok, ok, True, True, 0, 60.0)
            for p, m, ok in (("llm", "k2", True), ("strong", "q", False))]
    assert "net değil" in report(atts)
