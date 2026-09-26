"""Run Aider non-interactively against the local K2 server."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

from .config import data_dir, effective_max_tokens
from .testrun import run_group

HIST_MAX_BYTES = 20 * 1024 * 1024


def _settings_files(cfg: dict, profile: str = "llm") -> tuple[dict[str, Path], Path]:
    d = data_dir(cfg) / "aider"
    d.mkdir(parents=True, exist_ok=True)
    c = cfg[profile]
    name = f"openai/{c['model']}"
    ctx = int(c.get("context") or 32768)
    mt = effective_max_tokens(cfg, profile)
    files = {}
    for effort in ("high", "medium", "low"):
        ctk = {"reasoning_effort": effort} if profile == "llm" else dict(c.get("template_kwargs") or {})
        p = d / f"{profile}-settings-{effort}.yml"
        p.write_text(yaml.safe_dump([{
            "name": name, "edit_format": cfg["aider"].get("edit_format", "diff"), "use_repo_map": True,
            "examples_as_sys_msg": True, "reasoning_tag": "think",
            "extra_params": {"max_tokens": mt, "temperature": c["temperature"], "top_p": c["top_p"],
                             "extra_body": {"chat_template_kwargs": ctk}},
        }], sort_keys=False), encoding="utf-8")
        files[effort] = p
    meta = d / f"{profile}-model-metadata.json"
    meta.write_text(json.dumps({name: {"max_input_tokens": max(4096, ctx - mt), "max_output_tokens": mt,
                                       "max_tokens": mt, "input_cost_per_token": 0.0, "output_cost_per_token": 0.0,
                                       "litellm_provider": "openai", "mode": "chat"}}), encoding="utf-8")
    return files, meta


def run_aider(cfg: dict, workdir: Path, message: str, effort: str, timeout: float,
              files: list[str] | None = None, profile: str = "llm") -> tuple[bool, str]:
    a = cfg["aider"]
    binary = a.get("binary", "aider")
    if not shutil.which(binary):
        return False, f"{binary} not found on PATH"
    settings, meta = _settings_files(cfg, profile)
    c = cfg[profile]
    if profile == "llm" and a.get("k2_adapter", True):
        # K2 answers with its own tool-call syntax; the adapter rewrites it into Aider's SEARCH/REPLACE blocks
        from .k2adapter import AdapterProxy
        with AdapterProxy(c["base_url"], workdir) as px:
            return _run(cfg, a, binary, settings, meta, c, px.url, workdir, message, effort, timeout, files)
    return _run(cfg, a, binary, settings, meta, c, c["base_url"], workdir, message, effort, timeout, files)


def _run(cfg: dict, a: dict, binary: str, settings: dict, meta: Path, c: dict, base_url: str, workdir: Path,
         message: str, effort: str, timeout: float, files: list[str] | None) -> tuple[bool, str]:
    hist = data_dir(cfg) / "aider" / "hist"
    hist.mkdir(parents=True, exist_ok=True)
    for f in (hist / "chat.md", hist / "input.txt"):  # Aider appends forever; keep one old generation
        if f.exists() and f.stat().st_size > HIST_MAX_BYTES:
            f.replace(f.with_name(f.name + ".1"))
    cmd = [binary, "--model", f"openai/{c['model']}",
           "--openai-api-base", base_url, "--openai-api-key", c.get("api_key") or "none",
           "--model-settings-file", str(settings.get(effort, settings["high"])), "--model-metadata-file", str(meta),
           "--yes-always", "--no-auto-commits", "--no-dirty-commits", "--no-gitignore", "--no-pretty",
           "--no-stream", "--no-show-model-warnings", "--no-check-update", "--analytics-disable",
           "--no-suggest-shell-commands", "--no-restore-chat-history",
           "--input-history-file", str(hist / "input.txt"), "--chat-history-file", str(hist / "chat.md"),
           "--map-tokens", str(a.get("map_tokens", 2048)), *a.get("extra_args", []),
           "--message", message, *[f for f in (files or []) if (workdir / f).exists()]]
    rc, out, err, timed_out = run_group(cmd, workdir, max(60, timeout))
    if timed_out:
        return False, ((out + err)[-19_000:] + "\naider timed out")
    return rc == 0, (out + err)[-20_000:]
