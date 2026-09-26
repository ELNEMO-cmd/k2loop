from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(os.path.expanduser("~/.config/k2loop"))
CONFIG_PATH = CONFIG_DIR / "config.yaml"
OVERRIDES_PATH = CONFIG_DIR / "overrides.yaml"  # written by `k2loop mode`, `laya-eval --apply`, `laya-mode`

DEFAULTS: dict[str, Any] = {
    "mode": "local",  # local: the strong local model (strong.model) plans/escalates/reviews, nothing leaves the PC | hybrid: Claude does it
    # All local models sit behind llama-swap (one OpenAI endpoint, loads/unloads by model name).
    # manager: llama-swap | systemd (old per-model services, kept for existing installs)
    "models": {"manager": "llama-swap", "swap_url": "http://127.0.0.1:8080"},
    "strong": {
        "base_url": "http://127.0.0.1:8080/v1", "model": "qwen3.6-35b-a3b", "api_key": "none",  # `k2loop strong`
        "temperature": 0.7, "top_p": 0.95, "max_tokens": 24576, "request_timeout_s": 10800,
        "template_kwargs": {"enable_thinking": True},
        "plan_samples": 3, "max_escalations_per_task": 1, "final_review": "per_task",
        "swap": False,  # True only with models.manager: systemd (llama-swap swaps by itself)
        "small_service": "k2-llm.service", "strong_service": "k2-strong.service",
        "load_timeout_s": 1800, "task_time_budget_min": 200,
        # planning context for the tool-less local planner
        "repo_context_chars": 24000,   # repo overview: tree + README + manifests + symbol map
        "read_files": True,            # exploration step: Qwen picks files to read + terms to grep
        "read_rounds": 2,
        "read_files_max": 12,
        "read_file_chars": 8000,       # per-file cap
        "chars_per_token": 3.0,        # rough, used to fit everything into the model context
    },
    "llm": {  # K2 Horizon 7B on the GPU
        "base_url": "http://127.0.0.1:8080/v1", "model": "k2-horizon-7b", "api_key": "none",
        "temperature": 1.0, "top_p": 0.95, "max_tokens": 32768, "request_timeout_s": 1800,
        "reasoning_effort": {"codegen": "high", "debug": "high", "review": "medium", "scoring": "low"},
    },
    "helper": {  # small model on the CPU, loaded next to K2 so the CPU works while the GPU codes
        "base_url": "http://127.0.0.1:8080/v1", "model": "helper", "api_key": "none",
        "temperature": 0.7, "top_p": 0.8, "max_tokens": 4096, "request_timeout_s": 900,
    },
    # which model does which ToT side job: llm (K2, GPU) | helper (CPU)
    "roles": {"scoring": "helper", "review": "llm", "agent": "helper"},
    "tot": {
        "candidates": 3, "debug_depth": 2, "debug_branches": 3, "keep_branches": 2, "explore_rate": 0.15,
        "review_votes": 3, "task_time_budget_min": 80, "aider_timeout_min": 25, "check_timeout_min": 15,
        "overlap_checks": True,  # run candidate N's checks on the CPU while K2 writes candidate N+1
    },
    "agent": {  # `k2loop serve`: continuous goal loop + OpenAI-compatible endpoint for Open WebUI
        "host": "127.0.0.1", "port": 8090, "model_name": "k2agent",
        "project": "",               # default git repo for code goals (a message can name another path)
        "research_first": True,      # look for existing work on the internet before planning
        "research_iterations": 2,
        # a code goal runs in rounds until it is really done: plan → steps → verify → re-plan what is left
        # completing the goal matters more than speed: every failure raises the effort of the next attempt
        "max_rounds": 0,             # planning rounds per goal (0 = until done)
        "goal_time_budget_min": 0,   # no new step/round starts after this (0 = no limit)
        "max_stalled_rounds": 3,     # stop and ask a human after this many rounds in a row without progress
        "research_on_failure": True, # stuck? search the internet for the failing error / the missing piece
        "verify_goal": True,         # when all steps pass, the strong model checks the goal against the diff
    },
    "plan": {  # plan goals on any topic: ToT branches + multi-agent debate + judge + refine
        "branches": 3,             # how many planners (Pragmatik, Kapsamlı, Risk-öncelikli, Yaratıcı)
        "debate_rounds": 1,        # rounds in which the planners read and criticise each other, then revise
        # strong (MoE, ~20 tok/s, good Turkish; one GPU swap for the whole plan) | llm (K2: weak Turkish) | helper
        "branch_model": "strong",
        "judge_model": "strong",   # the judge and the final refinement
        "language": "Turkish",
    },
    "research": {
        "ldr_url": "http://127.0.0.1:5000",   # Local Deep Research (credentials: ~/.config/k2loop/secrets.env)
        "searxng_url": "http://127.0.0.1:8888",
        "timeout_min": 25,
    },
    "trace": {"enabled": True, "endpoint": "http://127.0.0.1:6006/v1/traces", "project": "k2loop"},
    "claude": {
        "enabled": False, "binary": "claude", "plan_model": "opus", "review_model": "opus",
        "escalation_model": "opus", "max_escalations_per_task": 1, "final_review": "per_task", "max_turns": 40,
    },
    "laya": {
        "enabled": True, "url": "http://127.0.0.1:8765", "timeout_s": 60,
        "mode": {"test_classification": "shadow", "branch_pruning": "shadow", "retry_escalate": "shadow"},
        "thresholds": {"test_classification": 0.70, "branch_pruning": 0.60, "retry_escalate": 0.75},
        "state_max_chars": 1800,
    },
    # SemIf-style decisions (option probabilities from one forward pass of the helper model): a second shadow
    # predictor next to Laya. laya.mode.<decision>: semif makes it decide when its confidence >= threshold.
    "semif": {
        "enabled": True, "model_profile": "helper", "timeout_s": 60,
        "thresholds": {"test_classification": 0.70, "branch_pruning": 0.60, "retry_escalate": 0.75},
    },
    "aider": {"binary": "aider", "edit_format": "diff", "map_tokens": 2048, "extra_args": []},
    "paths": {"data_dir": "~/.local/share/k2loop"},
}


def _merge(base: dict, over: dict) -> dict:
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


TOOL_DIRS = ["~/.local/bin", "/usr/local/cuda/bin", "/usr/lib/wsl/lib"]


def ensure_tool_path() -> None:
    """aider, ruff, uv, llama-swap live in ~/.local/bin. Processes started by systemd, Bernstein (which filters
    the environment) or cron often lack it, and then every candidate silently fails with 'aider not found'."""
    parts = os.environ.get("PATH", "").split(os.pathsep)
    extra = [d for d in (os.path.expanduser(p) for p in TOOL_DIRS) if os.path.isdir(d) and d not in parts]
    if extra:
        os.environ["PATH"] = os.pathsep.join(extra[:1] + parts + extra[1:])  # ~/.local/bin first, the rest last


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    ensure_tool_path()
    cfg = copy.deepcopy(DEFAULTS)
    p = Path(path) if path else CONFIG_PATH
    if p.exists():
        _merge(cfg, yaml.safe_load(p.read_text(encoding="utf-8")) or {})
    if path is None and OVERRIDES_PATH.exists():
        _merge(cfg, yaml.safe_load(OVERRIDES_PATH.read_text(encoding="utf-8")) or {})
    cfg["paths"]["data_dir"] = str(Path(os.path.expanduser(cfg["paths"]["data_dir"])))
    for fname, section in (("llm_state.env", "llm"), ("strong_state.env", "strong")):  # CTX chosen by probes
        state = Path(cfg["paths"]["data_dir"]) / fname
        if state.exists():
            for line in state.read_text(encoding="utf-8").splitlines():
                if line.startswith("CTX="):
                    try:
                        cfg[section]["context"] = int(line.split("=", 1)[1].strip())
                    except ValueError:
                        pass
    if cfg.get("mode") == "local":
        cfg["tot"]["task_time_budget_min"] = max(cfg["tot"]["task_time_budget_min"],
                                                 cfg["strong"]["task_time_budget_min"])
    if cfg["models"].get("manager") == "llama-swap":
        # one endpoint for every local model (older config files still name the per-model ports 8080/8081)
        url = cfg["models"]["swap_url"].rstrip("/") + "/v1"
        for section in ("llm", "strong", "helper"):
            cfg[section]["base_url"] = url
        cfg["strong"]["swap"] = False  # llama-swap unloads K2 itself when Qwen is requested
    secrets = CONFIG_DIR / "secrets.env" if path is None else Path(path).with_name("secrets.env")
    if secrets.exists():
        for line in secrets.read_text(encoding="utf-8").splitlines():
            k, _, v = line.partition("=")
            v = v.strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":  # KEY="value" / KEY='value'
                v = v[1:-1]
            if k.strip() == "LDR_USER":
                cfg["research"]["ldr_user"] = v.strip()
            elif k.strip() == "LDR_PASSWORD":
                cfg["research"]["ldr_password"] = v.strip()
            elif k.strip() == "AGENT_API_KEY":
                cfg["agent"]["api_key"] = v.strip()
    return cfg


def data_dir(cfg: dict) -> Path:
    d = Path(cfg["paths"]["data_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def effective_max_tokens(cfg: dict, profile: str = "llm") -> int:
    mt = int(cfg[profile]["max_tokens"])
    ctx = cfg[profile].get("context")
    if ctx:
        mt = min(mt, max(4096, int(ctx) // 2))
    return mt


def write_overrides(updates: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    cur = yaml.safe_load(OVERRIDES_PATH.read_text(encoding="utf-8")) if OVERRIDES_PATH.exists() else {}
    cur = _merge(cur or {}, updates)
    OVERRIDES_PATH.write_text(yaml.safe_dump(cur, sort_keys=False, allow_unicode=True), encoding="utf-8")


def write_laya_overrides(updates: dict) -> None:
    write_overrides({"laya": updates})
