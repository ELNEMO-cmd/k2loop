"""The coder is chosen from measured pass rates, not from model size."""
import contextlib
import subprocess
from pathlib import Path

from k2loop import modelstats
from k2loop.cli import main as cli_main
from k2loop.config import load_config
from k2loop.engine import Engine


def _cfg(tmp_path, mode="local"):
    cfg = load_config(tmp_path / "none.yaml")
    cfg["paths"]["data_dir"] = str(tmp_path / "data")
    cfg["trace"]["enabled"] = False
    cfg["mode"] = mode
    return cfg


def _fill(cfg, profile, n, passed, types=("python",)):
    for i in range(n):
        modelstats.record(cfg, profile, kind="codegen", types=list(types), usable=True, passed=i < passed,
                          failures=0 if i < passed else 2, secs=10)


def test_without_data_a_different_model_is_tried(tmp_path):
    cfg = _cfg(tmp_path)
    assert modelstats.choose_coder(cfg, ["python"])[0] == "llm"
    assert modelstats.choose_coder(cfg, ["python"], failed="llm")[0] == "strong"
    assert modelstats.choose_coder(cfg, ["python"], failed="strong")[0] == "llm"


def test_clearly_better_measured_model_wins_even_after_its_failure(tmp_path):
    cfg = _cfg(tmp_path)
    _fill(cfg, "llm", 25, 15)       # K2 60 %
    _fill(cfg, "strong", 25, 5)     # "big" model 20 %
    prof, why = modelstats.choose_coder(cfg, ["python"], failed="llm")
    assert prof == "llm" and "ölçüme göre" in why


def test_close_rates_mean_try_the_other_model(tmp_path):
    cfg = _cfg(tmp_path)
    _fill(cfg, "llm", 30, 15)
    _fill(cfg, "strong", 30, 16)
    assert modelstats.choose_coder(cfg, ["python"], failed="strong")[0] == "llm"


def test_same_project_type_is_preferred(tmp_path):
    cfg = _cfg(tmp_path)
    _fill(cfg, "llm", 25, 20, types=("node",))
    _fill(cfg, "strong", 25, 5, types=("node",))
    _fill(cfg, "llm", 25, 5)
    _fill(cfg, "strong", 25, 20)
    assert modelstats.choose_coder(cfg, ["python"])[0] == "strong"
    assert modelstats.choose_coder(cfg, ["node"])[0] == "llm"


def test_stats_are_per_model_name_not_profile(tmp_path):
    cfg = _cfg(tmp_path)
    _fill(cfg, "strong", 25, 25)                 # the MoE did great...
    cfg["strong"]["model"] = "qwen3.8-27b"       # ...but now the dense model is configured: no data for it
    _fill(cfg, "llm", 25, 10)
    assert modelstats.choose_coder(cfg, ["python"])[0] == "llm"


def test_hybrid_mode_only_uses_k2(tmp_path):
    cfg = _cfg(tmp_path, mode="hybrid")
    assert modelstats.choose_coder(cfg, ["python"], failed="llm")[0] == "llm"


def _repo(tmp_path):
    r = tmp_path / "repo"
    (r / ".k2loop").mkdir(parents=True)
    (r / ".k2loop" / "check.sh").write_text("exit 0\n")
    (r / "a.py").write_text("x = 1\n")
    for args in (["init", "-q"], ["add", "-A"], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "i"]):
        subprocess.run(["git", *args], cwd=r, check=True, capture_output=True)
    return r


def test_engine_records_every_candidate_and_uses_the_measured_best(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg["tot"].update(candidates=1, review_votes=0)
    cfg["strong"]["final_review"] = "none"
    _fill(cfg, "llm", 25, 5)
    _fill(cfg, "strong", 25, 20)
    used = []
    monkeypatch.setattr("k2loop.swap.strong_model", lambda cfg, log=print: contextlib.nullcontext(True))

    def fake_aider(cfg, d, msg, effort, timeout, files=None, profile="llm"):
        used.append(profile)
        (Path(d) / "a.py").write_text("x = 7\n")
        return True, "ok"

    monkeypatch.setattr("k2loop.engine.run_aider", fake_aider)
    res = Engine(cfg, "task", _repo(tmp_path), "r1", log=lambda m: None, emit=lambda *a, **k: None,
                 project_types=["python"]).run()
    assert used == ["strong"] and res.coder == "strong" and res.status == "done"
    s = modelstats.summary(cfg)[cfg["strong"]["model"]]
    assert s["n"] == 26 and s["passed"] == 21   # this run's candidate was recorded


def test_model_stats_command(tmp_path, monkeypatch, capsys):
    cfg = _cfg(tmp_path)
    _fill(cfg, "llm", 3, 2)
    monkeypatch.setattr("k2loop.cli.load_config", lambda: cfg)
    assert cli_main(["model-stats"]) == 0
    out = capsys.readouterr().out
    assert cfg["llm"]["model"] in out and "67%" in out and "Şu anki seçim" in out
