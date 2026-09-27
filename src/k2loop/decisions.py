"""The three Laya decisions, each with a deterministic rule, shadow logging and optional activation.

Shadow mode: the rule decides, Laya only predicts and both are logged.
Active mode: Laya decides when its calibrated confidence >= threshold, otherwise the rule.
Outcomes (what turned out to be right) are logged separately and joined by `laya-eval`.
"""
from __future__ import annotations

import json
import random
import re
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from .config import data_dir
from .jsonl import append_jsonl

# ---------------------------------------------------------------- labels
TEST_LABELS: dict[str, str] = {
    "compile_syntax": "Code does not compile or parse: syntax errors, type errors, undefined symbols at build time.",
    "import_dependency": "A module, package, header or library cannot be found, installed or imported.",
    "assertion_logic": "Code runs but a test assertion fails: wrong result or wrong behaviour.",
    "timeout_performance": "A test or build times out, hangs or is too slow.",
    "environment_flaky": "Failure caused by the environment: ports, permissions, network, disk space, or flakiness.",
    "broken_test": "The test itself is wrong or cannot be collected: bad fixture, bad test setup or test code.",
}
PRUNE_LABELS: dict[str, str] = {
    "try": "This fix hypothesis plausibly addresses the failure shown and is worth trying.",
    "skip": "This fix hypothesis is unlikely to fix the failure shown.",
}
ESC_LABELS: dict[str, str] = {
    "retry_local": "Keep fixing with the local model: the failures are changing or shrinking, progress is being made.",
    "escalate_claude": "Hand the task to the stronger remote model: the local model is stuck on the same failure.",
    "ask_human": "Stop and ask a human: the task is blocked, underspecified or needs hardware/credentials.",
}

_TEST_RULES: list[tuple[str, list[str]]] = [
    ("compile_syntax", [r"SyntaxError", r"IndentationError", r"error TS\d+", r"\berror: expected\b",
                        r"undefined reference", r"fatal error:", r"compilation terminated", r"was not declared",
                        r"cannot find symbol", r"\bNameError\b", r"Unexpected token", r"\*\*\* \[.*\] Error"]),
    ("import_dependency", [r"ModuleNotFoundError", r"ImportError", r"No module named", r"Cannot find module",
                           r"Could not resolve", r"No such file or directory.*\.h", r"Library .* not found",
                           r"ERR_MODULE_NOT_FOUND", r"command not found"]),
    ("timeout_performance", [r"\bTIMEOUT\b", r"timed out", r"TimeoutError", r"Timeout of \d+ms exceeded",
                             r"deadline exceeded"]),
    ("environment_flaky", [r"ECONNREFUSED", r"Address already in use", r"Permission denied",
                           r"No space left", r"Connection refused", r"EADDRINUSE", r"flaky"]),
    ("broken_test", [r"fixture '.*' not found", r"errors? during collection", r"ERROR collecting",
                     r"Test suite failed to run"]),
    ("assertion_logic", [r"AssertionError", r"\bassert\b", r"Expected", r"toBe", r"toEqual", r"FAILED",
                         r"\d+ failed", r"failing"]),
]


_STATIC_FAILED = re.compile(r"=== \[(lint|typecheck)\] FAILED")
_OTHER_FAILED = re.compile(r"=== \[(?!lint\]|typecheck\])[^\]]+\] FAILED")


def classify_test_rule(log: str) -> str:
    # only a static check (ruff F841, eslint, tsc) failed: not a wrong result, whatever "FAILED" line it printed
    if _STATIC_FAILED.search(log) and not _OTHER_FAILED.search(log):
        return "compile_syntax"
    for label, pats in _TEST_RULES:
        if any(re.search(p, log) for p in pats):
            return label
    return "assertion_logic"


def retry_escalate_rule(st: dict[str, Any]) -> str:
    """Deterministic policy used in shadow mode (and as fallback in active mode)."""
    esc_left = int(st.get("escalations_left", 0))
    if st.get("needs_hardware"):
        return "ask_human"
    if float(st.get("time_left_min", 999)) < 8:
        return "escalate_claude" if esc_left > 0 else "ask_human"
    if int(st["level"]) < int(st["max_depth"]) and (st.get("error_changed") or int(st["level"]) == 1):
        return "retry_local"
    return "escalate_claude" if esc_left > 0 else "ask_human"


def _q_choice(instr: str, labels: dict[str, str]) -> dict[str, Any]:
    return {"type": "choice", "instructions": instr, "criteria": labels}


QUESTIONS = {
    "test_classification": {"category": _q_choice(
        "Which category best describes why these checks failed?", TEST_LABELS)},
    "branch_pruning": {"worth": _q_choice(
        "Is this fix hypothesis worth trying for the failure shown?", PRUNE_LABELS)},
    "retry_escalate": {"next": _q_choice(
        "Given this debugging history, what should happen next?", ESC_LABELS)},
}
QID = {"test_classification": "category", "branch_pruning": "worth", "retry_escalate": "next"}


def compact_state(state: dict[str, Any], max_chars: int) -> dict[str, Any]:
    """Shrink long string fields until the JSON fits Laya's small context."""
    st = dict(state)
    for _ in range(12):
        if len(json.dumps(st, ensure_ascii=False)) <= max_chars:
            break
        key = max((k for k, v in st.items() if isinstance(v, str)), key=lambda k: len(st[k]), default=None)
        if key is None:
            break
        v = st[key]
        st[key] = "…" + v[-max(80, int(len(v) * 0.6)):]
    return st


class LayaClient:
    def __init__(self, url: str, timeout: float = 15):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def health(self) -> tuple[bool, str]:
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=5) as r:
                return True, r.read().decode()
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)

    def predict(self, state: Any, questions: dict) -> dict | None:
        req = urllib.request.Request(self.url + "/predict", method="POST",
                                     data=json.dumps({"state": state, "questions": questions}).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode()).get("answers")
        except Exception:  # noqa: BLE001
            return None


class Decider:
    """mode per decision: shadow (rule decides) | active (Laya decides) | semif (SemIf decides);
    Laya and SemIf are always asked (when enabled) and logged, so `laya-eval` can compare all three."""

    def __init__(self, cfg: dict, run_id: str, client: LayaClient | None = None, semif: Any = "auto"):
        self.cfg = cfg
        self.lc = cfg["laya"]
        self.run_id = run_id
        self.dir = data_dir(cfg) / "laya"
        # generous timeout: during a ToT run the CPU is busy (helper model, checks, MoE experts) and Laya answers
        # queued requests one by one; a 15 s timeout silently dropped its predictions
        self.client = client or (LayaClient(self.lc["url"], float(self.lc.get("timeout_s", 60)))
                                 if self.lc.get("enabled", True) else None)
        if semif == "auto":
            from .semif import SemIfScorer
            semif = SemIfScorer(cfg) if cfg.get("semif", {}).get("enabled") else None
        self.semif = semif

    def _semif(self, name: str, state: dict) -> tuple[str | None, float | None, dict | None]:
        if not self.semif:
            return None, None, None
        q = QUESTIONS[name][QID[name]]
        return self.semif.predict(state, q["instructions"], q["criteria"])

    def _laya(self, name: str, state: dict) -> tuple[str | None, float | None, dict | None]:
        if not self.client:
            return None, None, None
        ans = self.client.predict(state, QUESTIONS[name])
        if not ans or QID[name] not in ans:
            return None, None, None
        a = ans[QID[name]]
        conf = a.get("answer_confidence", a.get("confidence"))
        return a.get("choice"), (float(conf) if conf is not None else None), a.get("probabilities")

    def decide(self, name: str, state: dict, rule_label: str) -> tuple[str, str]:
        state = compact_state(state, int(self.lc.get("state_max_chars", 1800)))
        laya_label, conf, probs = self._laya(name, state)
        s_label, s_conf, s_probs = self._semif(name, state)
        mode = self.lc["mode"].get(name, "shadow")
        thr = float(self.lc["thresholds"].get(name, 0.7))
        s_thr = float(self.cfg.get("semif", {}).get("thresholds", {}).get(name, 0.7))
        if mode == "active" and laya_label and conf is not None and conf >= thr:
            used = laya_label
        elif mode == "semif" and s_label and s_conf is not None and s_conf >= s_thr:
            used = s_label
        else:
            used = rule_label
        rec_id = uuid.uuid4().hex[:12]
        append_jsonl(self.dir / f"{name}.jsonl", {
            "id": rec_id, "ts": time.time(), "run": self.run_id, "decision": name, "mode": mode,
            "state": state, "rule": rule_label, "laya": laya_label, "laya_conf": conf,
            "laya_probs": probs, "semif": s_label, "semif_conf": s_conf, "semif_probs": s_probs, "used": used})
        return used, rec_id

    def outcome(self, rec_id: str | None, name: str, label: str) -> None:
        if rec_id:
            append_jsonl(self.dir / f"{name}.outcomes.jsonl",
                         {"id": rec_id, "decision": name, "label": label, "ts": time.time()})

    # -- decision 1
    def classify_failure(self, log_tail: str, project_types: list[str], check_cmd: str) -> tuple[str, str]:
        state = {"project_types": ",".join(project_types), "check_command": check_cmd, "log_tail": log_tail}
        return self.decide("test_classification", state, classify_test_rule(log_tail))

    # -- decision 2
    def prune(self, hyps: list[dict], k2_scores: list[float], base_state: dict, keep: int,
              explore_rate: float) -> list[tuple[dict, str]]:
        """Return the hypotheses to try (ordered) with their decision record ids."""
        order_rule = sorted(range(len(hyps)), key=lambda i: -k2_scores[i] if i < len(k2_scores) else 0)
        rule_keep = set(order_rule[:keep])
        mode = self.lc["mode"].get("branch_pruning", "shadow")
        recs, p_try = [], []
        for i, h in enumerate(hyps):
            st = dict(base_state, hypothesis=str(h.get("hypothesis", ""))[:600])
            used, rid = self.decide("branch_pruning", st, "try" if i in rule_keep else "skip")
            recs.append(rid)
            p_try.append(1.0 if used == "try" else 0.0)
        if random.random() < explore_rate:
            chosen = list(range(len(hyps)))
        elif mode in ("active", "semif"):
            chosen = [i for i in order_rule if p_try[i] > 0][:keep] or order_rule[:1]
        else:
            chosen = order_rule[:keep]
        return [(hyps[i], recs[i]) for i in chosen]

    # -- decision 3
    def retry_or_escalate(self, st: dict) -> tuple[str, str]:
        return self.decide("retry_escalate", st, retry_escalate_rule(st))


def load_logs(cfg: dict, name: str) -> tuple[list[dict], dict[str, str]]:
    from .jsonl import read_jsonl
    d = Path(cfg["paths"]["data_dir"]) / "laya"
    recs = list(read_jsonl(d / f"{name}.jsonl"))
    outs = {o["id"]: o["label"] for o in read_jsonl(d / f"{name}.outcomes.jsonl")}
    return recs, outs
