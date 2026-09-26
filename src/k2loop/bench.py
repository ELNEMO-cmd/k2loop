"""`k2loop bench`: K2 vs the strong model on the same coding tasks, judged by hidden tests.

Measures raw code-writing only: one Aider run per attempt in a fresh repo (K2 goes through the real k2adapter), then
the task's hidden tests decide. The full ToT engine is deliberately NOT used: it escalates to the strong model when K2
is stuck, which would mix the two models' results. All attempts of one model run before the other model's, so the
GPU swaps once. Results are printed, saved under <data>/bench/, and recorded in model_stats (kind "bench") so the
automatic coder choice starts from real numbers.
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from . import modelstats
from .aider_runner import run_aider
from .bench_tasks import TASKS, BenchTask
from .config import data_dir
from .engine import CODEGEN_RULES
from .gitops import IDENT, git, snapshot_diff
from .project import render_check_sh
from .testrun import count_failures, run_group


@dataclass
class Attempt:
    task: str
    level: str
    profile: str
    model: str
    rep: int
    passed: bool
    visible_passed: bool
    usable: bool
    aider_ok: bool
    failures: int
    secs: float
    note: str = ""


def pytest_cmd() -> list[str]:
    """A pytest that works on this machine (k2loop's own tool venv usually has none)."""
    for cand in ([sys.executable, "-m", "pytest"], ["python3", "-m", "pytest"]):
        if subprocess.run([*cand, "--version"], capture_output=True).returncode == 0:
            return cand
    if shutil.which("uv"):
        return ["uv", "run", "--quiet", "--no-project", "--with", "pytest", "python", "-m", "pytest"]
    raise SystemExit("pytest bulunamadı: `pip install pytest` ya da uv kurulu olmalı")


def make_repo(task: BenchTask, dest: Path) -> Path:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for rel, text in task.files.items():
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        (dest / rel).write_text(text, encoding="utf-8")
    (dest / ".k2loop").mkdir(exist_ok=True)
    (dest / ".k2loop" / "check.sh").write_text(render_check_sh([("tests", "$PYTEST -x -q")]), encoding="utf-8")
    (dest / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n.aider*\n", encoding="utf-8")
    git(["init", "-q"], dest)
    git(["add", "-A"], dest)
    git([*IDENT, "commit", "-q", "-m", "bench: start"], dest)
    return dest


def run_tests(repo: Path, pytest: list[str], files: list[str] | None, timeout: float) -> tuple[bool, int, str]:
    rc, out, err, timed_out = run_group([*pytest, "-q", "-p", "no:cacheprovider", *(files or [])], repo, timeout)
    text = out + "\n" + err
    return rc == 0 and not timed_out, (0 if rc == 0 else max(1, count_failures(text))), text[-3000:]


def judge(task: BenchTask, repo: Path, pytest: list[str], timeout: float = 300) -> tuple[bool, bool, int, str]:
    """(hidden passed, visible passed, hidden failures, output). Hidden tests are copied in only now, and they
    replace any same-named file the model may have created."""
    vis_ok, _, _ = run_tests(repo, pytest, None, timeout)
    for rel, text in task.hidden.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text, encoding="utf-8")
    ok, fails, out = run_tests(repo, pytest, list(task.hidden), timeout)
    return ok, vis_ok, fails, out


def attempt(cfg: dict, task: BenchTask, profile: str, rep: int, work: Path, pytest: list[str],
            timeout: float, aider: Callable = run_aider) -> Attempt:
    repo = make_repo(task, work / f"{task.name}-{profile}-{rep}")
    msg = CODEGEN_RULES + task.prompt
    t0 = time.time()
    ok, out = aider(cfg, repo, msg, "high", timeout, task.hint_files, profile=profile)
    usable = bool(snapshot_diff(repo).strip())
    passed, vis, fails, test_out = judge(task, repo, pytest)
    secs = time.time() - t0
    (repo / ".bench-aider.log").write_text(out, encoding="utf-8")
    (repo / ".bench-tests.log").write_text(test_out, encoding="utf-8")
    note = "" if usable else ("aider timed out" if "timed out" in out[-200:] else "no change")
    return Attempt(task.name, task.level, profile, cfg[profile]["model"], rep, passed and usable, vis, usable, ok,
                   fails, round(secs, 1), note)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def report(atts: list[Attempt]) -> str:
    models = list(dict.fromkeys(a.model for a in atts))
    tasks = list(dict.fromkeys(a.task for a in atts))
    w = max([len(t) for t in tasks] + [10])
    lines = ["Gizli testleri geçen denemeler (geçen / deneme · ort. süre):", ""]
    lines.append(f"  {'görev':<{w}}  " + "  ".join(f"{m[:24]:>26}" for m in models))
    for t in tasks:
        cells = []
        for m in models:
            xs = [a for a in atts if a.task == t and a.model == m]
            k = sum(a.passed for a in xs)
            secs = sum(a.secs for a in xs) / len(xs) if xs else 0
            cells.append(f"{f'{k}/{len(xs)} · {secs / 60:.1f} dk':>26}")
        level = next(a.level for a in atts if a.task == t)
        lines.append(f"  {t:<{w}}  " + "  ".join(cells) + f"   ({level})")
    lines.append("")
    for m in models:
        xs = [a for a in atts if a.model == m]
        k, n = sum(a.passed for a in xs), len(xs)
        lo, hi = wilson(k, n)
        vis = sum(a.visible_passed and a.usable for a in xs)
        empty = sum(not a.usable for a in xs)
        mins = sum(a.secs for a in xs) / 60
        lines.append(f"  {m}: {k}/{n} = %{100 * k / n if n else 0:.0f}  (%95 aralık %{100 * lo:.0f}–%{100 * hi:.0f})"
                     f" · kendi testleri geçti {vis} · boş cevap {empty} · toplam {mins:.0f} dk")
    if len(models) == 2:
        a, b = ([x for x in atts if x.model == m] for m in models)
        ka, kb = sum(x.passed for x in a), sum(x.passed for x in b)
        la, ha = wilson(ka, len(a))
        lb, hb = wilson(kb, len(b))
        if ha < lb or hb < la:
            win = models[0] if ka / max(1, len(a)) > kb / max(1, len(b)) else models[1]
            lines.append(f"\n→ Fark anlamlı: {win} daha iyi kod yazıyor (bu görevlerde).")
        else:
            lines.append("\n→ Fark istatistiksel olarak net değil (aralıklar örtüşüyor): daha çok tekrar gerekir "
                         "(--repeats) ya da iki model bu görevlerde benzer.")
    return "\n".join(lines)


def run_bench(cfg: dict, profiles: list[str], repeats: int, task_names: list[str] | None = None,
              timeout_min: float | None = None, record: bool = True, keep: bool = False,
              log: Callable[[str], None] = print, aider: Callable = run_aider) -> tuple[list[Attempt], Path]:
    tasks = [t for t in TASKS if not task_names or t.name in task_names]
    if not tasks:
        raise SystemExit(f"görev yok; seçenekler: {', '.join(t.name for t in TASKS)}")
    pytest = pytest_cmd()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = data_dir(cfg) / "bench" / stamp
    work = out_dir / "work"
    timeout = float(timeout_min or cfg["tot"].get("aider_timeout_min", 25)) * 60
    atts: list[Attempt] = []
    total = len(profiles) * len(tasks) * repeats
    for profile in profiles:                 # one model at a time: a single GPU swap per model
        ctx = _model_ctx(cfg, profile, log)
        with ctx as up:
            if not up:
                log(f"✗ {cfg[profile]['model']} açılamadı, atlanıyor")
                continue
            for rep in range(1, repeats + 1):
                for task in tasks:
                    log(f"[{len(atts) + 1}/{total}] {cfg[profile]['model']} · {task.name} · tekrar {rep}")
                    a = attempt(cfg, task, profile, rep, work, pytest, timeout, aider)
                    atts.append(a)
                    log(f"    {'✅ geçti' if a.passed else '❌ kaldı'} ({a.secs / 60:.1f} dk)"
                        + (f" — {a.note}" if a.note else ""))
                    if record:
                        modelstats.record(cfg, profile, kind="bench", types=["python"], usable=a.usable,
                                          passed=a.passed, failures=a.failures, secs=a.secs)
                    _save(out_dir, atts)
    text = report(atts) if atts else "Hiç deneme yapılamadı."
    (out_dir / "report.txt").write_text(text + "\n", encoding="utf-8")
    if not keep:
        shutil.rmtree(work, ignore_errors=True)
    return atts, out_dir


def _save(out_dir: Path, atts: list[Attempt]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "attempts.json").write_text(json.dumps([asdict(a) for a in atts], ensure_ascii=False, indent=1),
                                           encoding="utf-8")


def _model_ctx(cfg: dict, profile: str, log):
    import contextlib
    if profile == "strong":
        from .swap import strong_model
        return strong_model(cfg, log)
    return contextlib.nullcontext(True)
