"""The ToT / Self-Consistency pipeline for ONE task.

1. Code generation (Self-Consistency): N candidates from K2, each in its own git worktree, checks run on each.
   Winner = passing candidate with the smallest diff.
2. Debugging (Tree of Thoughts): if nothing passes, classify the failure (Laya #1), let K2 propose fix
   hypotheses, prune them (Laya #2), try the survivors; repeat up to `debug_depth`, deciding between
   levels whether to continue locally or escalate (Laya #3). Claude gets at most `max_escalations_per_task`.
3. Review (Self-Consistency + Claude): K2 reviews the winning diff `review_votes` times (majority vote),
   one fix round if the majority requests changes, then an optional Claude final review.
4. The winning diff is applied to the Bernstein worktree and committed.

GPU/CPU overlap: while K2 (GPU) writes candidate N+1, candidate N's checks run on the CPU; scoring runs on the
helper model (CPU) when `roles.scoring: helper`. Every step is emitted to the event log (live UI) and traced
(Phoenix).
"""
from __future__ import annotations

import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import gitops, modelstats
from .aider_runner import run_aider
from .claude import run_claude
from .config import data_dir
from .decisions import TEST_LABELS, Decider
from .events import event_log
from .llm import LLM
from .testrun import CheckResult, log_tail, run_checks
from .trace import current_context, set_attrs, setup as trace_setup, span

CHECK_CMD = "bash .k2loop/check.sh"

CODEGEN_RULES = (
    "You are implementing exactly ONE task in this repository.\n"
    "Rules: make a minimal but complete change; keep the existing style; add or update tests when the task "
    "changes behaviour; do NOT run shell commands, do NOT call HTTP endpoints, do NOT commit. "
    f"Correctness is judged by `{CHECK_CMD}`.\n\nTASK:\n"
)


def _str_list(v) -> list[str]:
    return [str(x) for x in v] if isinstance(v, list) else ([str(v)] if v else [])


@dataclass
class Cand:
    name: str
    path: Path
    diff: str
    lines: int
    chk: CheckResult
    source: str = "k2"

    @property
    def usable(self) -> bool:
        return bool(self.diff.strip())


@dataclass
class Result:
    status: str  # done | human | failed
    summary: str
    files: list[str] = field(default_factory=list)
    verified: bool = False
    patch_path: str | None = None
    sha: str | None = None
    check_tail: str = ""          # failing check output of the closest attempt (for re-planning / research)
    coder: str = "llm"            # config profile that wrote the candidates


class Engine:
    def __init__(self, cfg: dict, task_text: str, base: Path, run_id: str, hb=None,
                 log: Callable[[str], None] = print, hint_files: list[str] | None = None,
                 project_types: list[str] | None = None, title: str | None = None,
                 emit: Callable[..., object] | None = None):
        self.cfg, self.tot = cfg, cfg["tot"]
        self.task = task_text
        self.title = (title or next((ln.strip("# ").strip() for ln in task_text.splitlines() if ln.strip()), "task"))[:72]
        self.base = base
        self.run_id = run_id
        self.hb = hb
        self._emit = emit or event_log(cfg).emit
        self._log = log
        self.hint_files = hint_files or []
        self.project_types = project_types or []
        trace_setup(cfg)
        self.llm = LLM(cfg)
        roles = cfg.get("roles") or {}
        self.scorer = LLM(cfg, roles.get("scoring", "llm"))
        self.reviewer = LLM(cfg, roles.get("review", "llm"))
        self.decider = Decider(cfg, run_id)
        self.eff = cfg["llm"]["reasoning_effort"]
        self.run_dir = data_dir(cfg) / "runs" / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        self.deadline = self.t0 + float(self.tot["task_time_budget_min"]) * 60
        self.local = cfg.get("mode") == "local"
        if self.local:
            self.esc_left = int(cfg["strong"]["max_escalations_per_task"])
        else:
            self.esc_left = int(cfg["claude"]["max_escalations_per_task"]) if cfg["claude"].get("enabled") else 0
        self.worktrees: list[Path] = []
        self.notes: list[str] = []
        self.best_fail: Cand | None = None   # closest failing attempt (debug keeps it up to date)
        # who writes the candidates: llm (K2) | strong | auto (the one with the better MEASURED pass rate,
        # modelstats); the goal loop sets it explicitly after failures
        cp = self.tot.get("codegen_profile", "auto") if self.local else "llm"
        self.coder_reason = ""
        if cp == "auto":
            cp, self.coder_reason = modelstats.choose_coder(cfg, self.project_types)
        self.gen_profile = cp if cp in modelstats.CODERS else "llm"
        if self.gen_profile != "llm":
            # the GPU holds one of K2 / strong: hypotheses and review follow the coder instead of swapping back
            self.llm = LLM(cfg, self.gen_profile)
            if roles.get("review", "llm") == "llm":
                self.reviewer = self.llm
            if roles.get("scoring", "llm") == "llm":
                self.scorer = self.llm
        self._gen_profiles: dict[str, str] = {}   # candidate name -> profile that wrote it

    # ------------------------------------------------------------ helpers
    def emit(self, kind: str, **data) -> None:
        try:
            self._emit(kind, run=self.run_id, **data)
        except Exception:  # noqa: BLE001  (the UI must never break a run)
            pass

    def log(self, msg: str) -> None:
        self._log(msg)
        self.emit("log", text=msg)

    def cancelled(self) -> bool:
        return self.hb is not None and bool(getattr(self.hb, "shutdown", False))

    def left(self) -> float:
        if self.cancelled():
            return 0.0
        return self.deadline - time.time()

    def used_frac(self) -> float:
        return (time.time() - self.t0) / max(1.0, self.deadline - self.t0)

    def phase(self, name: str, msg: str) -> None:
        pct = int(self.used_frac() * 100)
        self.emit("phase", stage=name, pct=pct, msg=msg)
        if self.hb:
            self.hb.set(name, pct, msg)
        else:
            self._log(f"[k2loop] {name} {pct}% {msg}")

    def _wt(self, name: str, base_diff: str | None = None) -> Path:
        p = gitops.new_worktree(self.base, self.run_dir / "wt" / name, base_diff)
        self.worktrees.append(p)
        return p

    def _run_checks(self, name: str, d: Path) -> CheckResult:
        if gitops.restore_protected(d):
            self.log(f"[k2loop] {name}: .k2loop/check.sh was modified by the model; the original is used")
        return run_checks(d, self._check_timeout())

    def _check_timeout(self) -> float:
        return min(float(self.tot["check_timeout_min"]) * 60, max(60.0, self.left() - 60))

    def _aider_timeout(self) -> float:
        return min(float(self.tot["aider_timeout_min"]) * 60, max(120.0, self.left() - 180))

    def _generate(self, name: str, message: str, effort: str, base_diff: str | None = None,
                  profile: str | None = None) -> tuple[str, Path, bool, str, str]:
        """GPU part: Aider + K2 (or the strong model) edit a fresh worktree. Returns (name, dir, aider_ok, aider_out,
        diff)."""
        profile = profile or self.gen_profile
        d = self._wt(name, base_diff)
        self.emit("cand_start", name=name, msg=message.removeprefix(CODEGEN_RULES)[-300:])
        with span(f"aider:{name}", "TOOL", tool__name="aider", input__value=message[-4000:],
                  llm__model_name=self.cfg[profile]["model"]) as sp:
            up = True
            if profile == "strong":
                from .swap import strong_model
                with strong_model(self.cfg, self.log) as up:
                    if up:
                        ok, out = run_aider(self.cfg, d, message, effort, self._aider_timeout(), self.hint_files,
                                            profile="strong")
                if not up:
                    self.notes.append(f"{name}: strong model unavailable, K2 wrote it")
            if profile != "strong" or not up:
                profile = "llm"
                ok, out = run_aider(self.cfg, d, message, effort, self._aider_timeout(), self.hint_files)
            set_attrs(sp, output__value=out[-4000:], aider_ok=ok)
        self._gen_profiles[name] = profile
        (self.run_dir / f"{name}.aider.log").write_text(out, encoding="utf-8")
        return name, d, ok, out, gitops.snapshot_diff(d)

    def _check(self, name: str, d: Path, ok: bool, out: str, diff: str, parent=None, source: str = "k2") -> Cand:
        """CPU part: run .k2loop/check.sh in the candidate's worktree."""
        with span(f"check:{name}", "EVALUATOR", parent=parent) as sp:
            chk = self._run_checks(name, d)
            set_attrs(sp, output__value=log_tail(chk.output, 3000), passed=chk.passed, failures=chk.failures)
        (self.run_dir / f"{name}.check.log").write_text(chk.output, encoding="utf-8")
        lines = gitops.diff_lines(diff)
        if name in self._gen_profiles and not out.rstrip().endswith("not found on PATH"):  # not an infra failure
            kind = next((k for pre, k in (("gen", "codegen"), ("fix", "fix"), ("review_fix", "review_fix"),
                                          ("strong_fix", "escalate")) if name.startswith(pre)), "other")
            modelstats.record(self.cfg, self._gen_profiles[name], kind=kind, types=self.project_types,
                              usable=bool(diff.strip()), passed=chk.passed and not chk.no_checks,
                              failures=chk.failures, secs=chk.duration)
        self.log(f"[k2loop] {name}: aider_ok={ok} diff_lines={lines} "
                 f"checks={'PASS' if chk.passed else 'FAIL'} failures={chk.failures}")
        self.emit("cand", name=name, aider_ok=ok, passed=chk.passed, failures=chk.failures, lines=lines,
                  files=gitops.changed_files(diff), secs=round(chk.duration, 1), source=source,
                  aider_tail=out[-1500:], check_tail="" if chk.passed else log_tail(chk.output, 1500))
        return Cand(name, d, diff, lines, chk, source=source)

    def _candidate(self, name: str, message: str, effort: str, base_diff: str | None = None) -> Cand:
        return self._check(*self._generate(name, message, effort, base_diff))

    @staticmethod
    def _passing(cands: list[Cand]) -> list[Cand]:
        return [c for c in cands if c.usable and c.chk.passed]

    # ------------------------------------------------------------ stage 1
    def codegen(self) -> list[Cand]:
        n = int(self.tot["candidates"])
        cands: list[Cand] = []
        pool = ThreadPoolExecutor(max_workers=1) if self.tot.get("overlap_checks", True) else None
        pending = None
        try:
            for i in range(n):
                if self.cancelled():
                    self.notes.append(f"cancelled after {i} candidates")
                    break
                if i > 0 and (self.used_frac() > 0.45 or self.left() < 900):
                    self.notes.append(f"codegen stopped after {i} candidates (time budget)")
                    break
                self.phase("codegen", f"candidate {i + 1}/{n}")
                gen = self._generate(f"gen{i + 1}", CODEGEN_RULES + self.task, self.eff["codegen"])
                if pool is None:
                    cands.append(self._check(*gen))
                    continue
                if pending is not None:          # previous candidate's checks ran while K2 wrote this one
                    cands.append(pending.result())
                pending = pool.submit(self._check, *gen, parent=current_context())
            if pending is not None:
                cands.append(pending.result())
        finally:
            if pool is not None:
                pool.shutdown(wait=True)
        return cands

    # ------------------------------------------------------------ stage 2
    def _hypotheses(self, cur: Cand, category: str, tail: str) -> list[dict]:
        k = int(self.tot["debug_branches"])
        prompt = (
            f"A coding task was attempted but `{CHECK_CMD}` fails.\n\nTASK (abridged):\n{self.task[:3000]}\n\n"
            f"FAILURE CATEGORY (guess): {category}\n\nCHECK OUTPUT (tail):\n{tail}\n\n"
            f"CURRENT DIFF (abridged):\n{cur.diff[:8000]}\n\n"
            f"Propose {k} DIFFERENT, specific hypotheses for what is wrong and how to fix it. "
            f"Categories: {', '.join(TEST_LABELS)}.\n"
            'Answer ONLY JSON: {"hypotheses":[{"hypothesis":"...","category":"...","files":["path"]}]}'
        )
        try:
            data = self.llm.ask_json(prompt, self.eff["debug"])
            hyps = [h for h in (data or {}).get("hypotheses", []) if isinstance(h, dict) and h.get("hypothesis")]
        except Exception as exc:  # noqa: BLE001
            self.log(f"[k2loop] hypothesis generation failed: {exc}")
            hyps = []
        return hyps[:k] or [{"hypothesis": "Fix the failing checks shown in the output.", "category": category}]

    def _k2_scores(self, hyps: list[dict], tail: str) -> list[float]:
        listing = "\n".join(f"{i}. {h['hypothesis']}" for i, h in enumerate(hyps))
        prompt = (f"Check output (tail):\n{tail[:2500]}\n\nFix hypotheses:\n{listing}\n\n"
                  'Score each 0-10 for how likely it fixes the failure. Answer ONLY JSON: {"scores":[...]}')
        try:
            data = self.scorer.ask_json(prompt, self.eff["scoring"], max_tokens=4096)
            scores = [float(s) for s in (data or {}).get("scores", [])]
            if len(scores) == len(hyps):
                return scores
        except Exception:  # noqa: BLE001
            pass
        return [float(len(hyps) - i) for i in range(len(hyps))]

    def debug(self, start: Cand) -> tuple[Cand | None, str]:
        depth = int(self.tot["debug_depth"])
        cur, prev_sig = start, start.chk.signature
        history: list[str] = []
        esc_recs: list[str] = []
        choice = "escalate_claude"
        for level in range(1, depth + 1):
            if self.left() < 600:
                self.notes.append("debug stopped (time budget)")
                break
            tail = log_tail(cur.chk.output)
            cat, crec = self.decider.classify_failure(tail[-1500:], self.project_types, CHECK_CMD)
            self.emit("decision", name="test_classification", used=cat)
            history.append(cat)
            self.phase("debug", f"level {level}/{depth} category={cat}")
            hyps = self._hypotheses(cur, cat, tail)
            scores = self._k2_scores(hyps, tail)
            base_state = {"failure_category": cat, "log_tail": tail[-700:]}
            kept = self.decider.prune(hyps, scores, base_state, int(self.tot["keep_branches"]),
                                      float(self.tot.get("explore_rate", 0.0)))
            kept_txt = [h["hypothesis"] for h, _ in kept]
            self.emit("hypotheses", level=level, items=[
                {"text": str(h.get("hypothesis"))[:300], "score": s, "kept": h["hypothesis"] in kept_txt}
                for h, s in zip(hyps, scores)])
            self.emit("decision", name="branch_pruning", used=f"{len(kept)}/{len(hyps)} dal denenecek")
            best: Cand | None = None
            for j, (h, rid) in enumerate(kept, 1):
                if self.left() < 480:
                    break
                self.phase("debug", f"level {level} branch {j}/{len(kept)}")
                msg = (CODEGEN_RULES + self.task + f"\n\nThe current changes FAIL `{CHECK_CMD}`.\n"
                       f"Failure output (tail):\n{tail}\n\nApply this fix hypothesis:\n{h['hypothesis']}\n")
                b = self._candidate(f"fix{level}_{j}", msg, self.eff["debug"], base_diff=cur.diff)
                improved = b.chk.passed or b.chk.failures < cur.chk.failures
                self.decider.outcome(rid, "branch_pruning", "try" if improved else "skip")
                if b.usable and b.chk.passed:
                    self.decider.outcome(crec, "test_classification", str(h.get("category") or cat))
                    for r in esc_recs:
                        self.decider.outcome(r, "retry_escalate", "retry_local")
                    return b, "local"
                if b.usable and (best is None or b.chk.failures < best.chk.failures):
                    best = b
            if best is not None and best.chk.failures <= cur.chk.failures:
                cur = best
            changed = cur.chk.signature != prev_sig
            prev_sig = cur.chk.signature
            st = {"level": level, "max_depth": depth, "error_changed": changed, "categories": ",".join(history),
                  "failures": cur.chk.failures, "diff_lines": cur.lines, "escalations_left": self.esc_left,
                  "time_left_min": round(self.left() / 60, 1), "project_types": ",".join(self.project_types),
                  "needs_hardware": "platformio" in self.project_types and cat == "environment_flaky"}
            choice, erec = self.decider.retry_or_escalate(st)
            esc_recs.append(erec)
            shown = choice.replace("escalate_claude", "escalate_qwen") if self.local else choice
            self.emit("decision", name="retry_escalate", used=shown)
            self.log(f"[k2loop] after level {level}: next={shown}")
            if choice != "retry_local":
                break
        if choice != "ask_human" and self.esc_left > 0 and self.left() > 300:
            c = self.escalate(cur)
            if c is not None:
                for r in esc_recs:
                    self.decider.outcome(r, "retry_escalate", "escalate_claude")  # label = "escalate to strong model"
                return c, (f"{self.cfg['strong']['model']}-local" if self.local else "claude")
        for r in esc_recs:
            self.decider.outcome(r, "retry_escalate", "ask_human")
        self.best_fail = cur
        self._save_patch(cur, "best-failing")
        return None, "human"

    @staticmethod
    def _handover_note(cur: Cand) -> str:
        """What the stronger model is told about the small model's attempt."""
        if not cur.usable:
            return ("\n\nA smaller model tried this task but did not change any file. Implement the whole task, "
                    f"including the tests, so that `{CHECK_CMD}` passes.")
        return (f"\n\nA smaller model's changes are applied but `{CHECK_CMD}` fails.\n"
                f"Check output (tail):\n{log_tail(cur.chk.output, 6000)}\n\nFind the real cause and fix it.")

    def escalate_local(self, cur: Cand) -> Cand | None:
        """Full-local escalation: the strong local model edits through Aider (GPU swap)."""
        from .swap import strong_model
        self.esc_left -= 1
        self.phase("escalate", f"handing over to {self.cfg['strong']['model']} (local)")
        d = self._wt("strong_fix", cur.diff)
        msg = CODEGEN_RULES + self.task + self._handover_note(cur)
        self.emit("cand_start", name="strong_fix", msg=f"{self.cfg['strong']['model']} eskalasyon (GPU'ya yükleniyor)")
        with strong_model(self.cfg, self.log) as ok:
            if not ok:
                self.notes.append("local escalation skipped: strong model unavailable")
                return None
            with span("aider:strong_fix", "TOOL", tool__name="aider", llm__model_name=self.cfg["strong"]["model"]):
                aok, out = run_aider(self.cfg, d, msg, "high", max(600.0, self.left() - 300), self.hint_files,
                                     profile="strong")
        (self.run_dir / "strong_fix.aider.log").write_text(out, encoding="utf-8")
        self._gen_profiles["strong_fix"] = "strong"
        c = self._check("strong_fix", d, aok, out, gitops.snapshot_diff(d), source=self.cfg["strong"]["model"])
        return c if c.usable and c.chk.passed else None

    def escalate(self, cur: Cand) -> Cand | None:
        if self.local:
            return self.escalate_local(cur)
        self.esc_left -= 1
        self.phase("escalate", "handing over to Claude")
        d = self._wt("claude_fix", cur.diff)
        prompt = (
            "You are fixing a task that a small local model attempted. Its changes are already applied in this "
            f"working tree, but `{CHECK_CMD}` fails.\n\nTASK:\n{self.task}\n\nCHECK OUTPUT (tail):\n"
            f"{log_tail(cur.chk.output, 6000)}\n\nMake `{CHECK_CMD}` pass with a correct, minimal change. "
            "You may run the check script. Do NOT commit, do NOT push, do NOT call any HTTP endpoint."
        )
        res = run_claude(self.cfg, prompt, d, model=self.cfg["claude"]["escalation_model"], edit=True,
                         timeout=min(1800, max(120, self.left() - 120)))
        if not res.ok:
            self.notes.append("claude escalation failed" + (" (usage limit)" if res.limited else f": {res.error[:200]}"))
            return None
        diff = gitops.snapshot_diff(d)
        chk = self._run_checks("claude_fix", d)
        c = Cand("claude_fix", d, diff, gitops.diff_lines(diff), chk, source="claude")
        self.log(f"[k2loop] claude_fix: checks={'PASS' if chk.passed else 'FAIL'}")
        return c if c.usable and chk.passed else None

    # ------------------------------------------------------------ stage 3
    def _fix_round(self, win: Cand, issues: list[str], tag: str) -> Cand:
        if not issues or self.left() < 600:
            return win
        msg = (CODEGEN_RULES + self.task + "\n\nThe current implementation passes the checks, but a review found "
               "these problems. Fix them without breaking the checks:\n- " + "\n- ".join(issues[:8]))
        f = self._candidate(f"review_fix_{tag}", msg, self.eff["codegen"], base_diff=win.diff)
        if f.usable and f.chk.passed:
            f.source = win.source
            return f
        self.notes.append(f"review fix ({tag}) rejected: checks failed")
        return win

    def review(self, win: Cand) -> Cand:
        votes, issues = [], []
        for i in range(int(self.tot["review_votes"])):
            if self.left() < 300:
                break
            self.phase("review", f"K2 vote {i + 1}")
            prompt = (f"Review this change for the task below. Look for bugs, missed requirements and broken "
                      f"edge cases (not style).\n\nTASK:\n{self.task[:4000]}\n\nDIFF:\n{win.diff[:20000]}\n\n"
                      'Answer ONLY JSON: {"verdict":"approve"|"changes","issues":[{"severity":"high"|"low",'
                      '"issue":"..."}]}')
            try:
                data = self.reviewer.ask_json(prompt, self.eff["review"], max_tokens=8192)
            except Exception:  # noqa: BLE001
                continue
            v = str(data.get("verdict", "")).lower() if isinstance(data, dict) else ""
            if v not in ("approve", "changes"):  # unparseable answer: not a vote (and never a silent approval)
                continue
            votes.append(v)
            found = data.get("issues")
            if v == "changes" and isinstance(found, list):
                issues += [str(x.get("issue")) for x in found
                           if isinstance(x, dict) and x.get("severity") == "high" and x.get("issue")]
        self.emit("review", votes=votes, issues=list(dict.fromkeys(issues))[:8])
        if votes and votes.count("changes") * 2 > len(votes):
            win = self._fix_round(win, list(dict.fromkeys(issues)), "k2")
        self.notes.append(f"K2 review votes: {votes}")

        if self.local:
            return self._local_final_review(win)
        c = self.cfg["claude"]
        if c.get("enabled") and c.get("final_review") == "per_task" and self.left() > 240:
            self.phase("review", "Claude final review")
            prompt = (f"Review this diff for the task. Only report real defects.\n\nTASK:\n{self.task[:6000]}\n\n"
                      f"DIFF:\n{win.diff[:60000]}\n\n"
                      'Answer ONLY JSON: {"verdict":"approve"|"reject","issues":["..."]}')
            res = run_claude(self.cfg, prompt, self.base, model=c["review_model"], edit=False, timeout=900,
                             max_turns=8)
            from .llm import extract_json
            data = extract_json(res.text) if res.ok else None
            if isinstance(data, dict) and str(data.get("verdict")).lower() == "reject":
                win = self._fix_round(win, _str_list(data.get("issues")), "claude")
                self.notes.append("Claude review: reject -> fix round")
            elif res.ok:
                self.notes.append("Claude review: approve")
            else:
                self.notes.append("Claude review skipped" + (" (usage limit)" if res.limited else ""))
        return win

    def _local_final_review(self, win: Cand) -> Cand:
        from .llm import LLM
        from .swap import strong_model
        if self.cfg["strong"].get("final_review") != "per_task" or self.left() < 600:
            return win
        self.phase("review", f"{self.cfg['strong']['model']} final review")
        prompt = (f"Review this diff for the task. Only report real defects (bugs, missed requirements).\n\n"
                  f"TASK:\n{self.task[:6000]}\n\nDIFF:\n{win.diff[:30000]}\n\n"
                  'Answer ONLY JSON: {"verdict":"approve"|"reject","issues":["..."]}')
        with strong_model(self.cfg, self.log) as ok:
            if not ok:
                self.notes.append("Qwen review skipped: strong model unavailable")
                return win
            try:
                data = LLM(self.cfg, "strong").ask_json(prompt, "high")
            except Exception as exc:  # noqa: BLE001
                self.notes.append(f"Qwen review failed: {exc}")
                return win
        if isinstance(data, dict) and str(data.get("verdict")).lower() == "reject":
            self.notes.append("Qwen review: reject -> fix round")
            self.emit("review", votes=["qwen: reject"], issues=_str_list(data.get("issues"))[:8])
            return self._fix_round(win, _str_list(data.get("issues")), "qwen")
        if not isinstance(data, dict) or str(data.get("verdict")).lower() != "approve":
            self.notes.append("Qwen review: no usable answer")
            return win
        self.notes.append("Qwen review: approve")
        self.emit("review", votes=["qwen: approve"], issues=[])
        return win

    # ------------------------------------------------------------ main
    def _save_patch(self, c: Cand | None, tag: str) -> str | None:
        if c is None or not c.diff:
            return None
        p = self.run_dir / f"{tag}.patch"
        p.write_text(c.diff, encoding="utf-8")
        return str(p)

    def cleanup(self) -> None:
        for p in self.worktrees:
            gitops.remove_worktree(self.base, p)

    def run(self) -> Result:
        self.emit("task_start", title=self.title, task=self.task[:2000], budget_min=round((self.deadline - self.t0) / 60),
                  coder=self.cfg[self.gen_profile]["model"], coder_reason=self.coder_reason)
        with span(f"task:{self.title}", "AGENT", input__value=self.task[:8000], run_id=self.run_id) as sp:
            res = self._run()
            res.coder = self.gen_profile
            set_attrs(sp, output__value=res.summary, status=res.status)
        self.emit("task_end", title=self.title, status=res.status, summary=res.summary, files=res.files,
                  sha=res.sha, patch=res.patch_path, mins=round((time.time() - self.t0) / 60, 1))
        return res

    def _run(self) -> Result:
        try:
            cands = self.codegen()
            passing = self._passing(cands)
            source = "k2-sc"
            win = min(passing, key=lambda c: c.lines) if passing else None
            if win is None:
                usable = sorted([c for c in cands if c.usable], key=lambda c: (c.chk.failures, c.lines))
                if usable:
                    win, source = self.debug(usable[0])
                elif self.esc_left > 0:
                    self.notes.append("no candidate produced any change")
                    empty = cands[0] if cands else None
                    win = self.escalate(empty) if empty else None
                    source = (f"{self.cfg['strong']['model']}-local" if self.local else "claude") if win else "human"
                else:
                    source = "human"
            if win is None:
                usable = [c for c in cands if c.usable]
                best = self.best_fail or (min(usable, key=lambda c: (c.chk.failures, c.lines)) if usable else None)
                patch = self._save_patch(best, "best-attempt")
                return Result("human", "k2loop could not make the checks pass. " + "; ".join(self.notes),
                              patch_path=patch, check_tail=log_tail(best.chk.output, 2500) if best else "")
            win = self.review(win)
            if self.cancelled():
                patch = self._save_patch(win, "cancelled-winner")
                return Result("human", "cancelled before commit; passing patch saved. " + "; ".join(self.notes),
                              patch_path=patch)
            self.phase("finish", f"applying {win.name}")
            gitops.apply_diff(self.base, win.diff)          # stages exactly the winner's changes
            files = gitops.commit_staged(self.base, f"k2loop: {self.title}")
            sha = gitops.git(["rev-parse", "HEAD"], self.base).strip() if files else None
            verified = not win.chk.no_checks
            summary = (f"k2loop: {len(cands)} candidates, winner={win.name} via {source}, "
                       f"{win.lines} diff lines, checks={'passed' if verified else 'NOT VERIFIED (no checks)'}. "
                       + "; ".join(self.notes))
            return Result("done", summary, files, verified, self._save_patch(win, "winner"), sha)
        except Exception as exc:  # noqa: BLE001
            (self.run_dir / "crash.txt").write_text(traceback.format_exc(), encoding="utf-8")
            return Result("failed", f"k2loop crashed: {type(exc).__name__}: {exc}")
        finally:
            self.cleanup()
