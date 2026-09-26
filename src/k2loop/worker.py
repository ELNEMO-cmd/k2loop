"""`generic-cli`: the command Bernstein's generic adapter spawns for every task.

Bernstein runs:  generic-cli --model <model> --prompt "<task prompt>"   (cwd = task worktree)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .bernstein_io import clean_task_text, mentioned_files, parse_prompt, report_complete, report_fail, report_refusal
from .config import load_config
from .engine import CHECK_CMD, Engine
from .heartbeat import Heartbeat
from .project import detect


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="generic-cli", description="k2loop worker for Bernstein")
    ap.add_argument("--model", default=None)
    ap.add_argument("--prompt", required=True)
    args, _unknown = ap.parse_known_args(argv)

    cfg = load_config()
    if args.model and args.model not in ("auto", "opus", "sonnet", "haiku"):
        cfg["llm"]["model"] = args.model
    cwd = Path.cwd()
    info = parse_prompt(args.prompt, cwd)
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + (info.task_ids[0] if info.task_ids else "manual")
    log(f"[k2loop] run {run_id} tasks={info.task_ids} server={info.base_url} heartbeats={len(info.heartbeat_paths)}")

    from .config import data_dir
    rd = data_dir(cfg) / "runs" / run_id
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "prompt.txt").write_text(args.prompt, encoding="utf-8")  # for diagnostics
    hb = Heartbeat(info.heartbeat_paths, echo=log, shutdown_file=info.shutdown_file)
    hb.start()
    task = clean_task_text(args.prompt)
    import re as _re
    tm = _re.search(r"^#+\s*Task \d+:\s*(.+?)\s*\(id=", args.prompt, _re.M)
    try:
        eng = Engine(cfg, task, cwd, run_id, hb=hb, log=log, hint_files=mentioned_files(task, cwd),
                     project_types=detect(cwd)["types"], title=tm.group(1) if tm else None)
        res = eng.run()
    finally:
        hb.stop()

    log(f"[k2loop] result: {res.status} — {res.summary}")
    if res.sha:  # registry for `k2loop collect` (Bernstein's own merge can fail on untracked copies)
        from .gitops import git
        from .jsonl import append_jsonl
        common = Path(git(["rev-parse", "--git-common-dir"], cwd).strip())
        common = (cwd / common).resolve() if not common.is_absolute() else common.resolve()
        git(["update-ref", f"refs/k2loop/{run_id}", res.sha], cwd, check=False)  # never garbage-collected
        append_jsonl(data_dir(cfg) / "commits.jsonl", {"repo": str(common), "sha": res.sha, "task": info.task_ids,
                                                         "files": res.files, "ts": time.time()})
        log(f"[k2loop] commit {res.sha[:10]} pinned as refs/k2loop/{run_id}")
    if not info.task_ids:
        log("[k2loop] no task id in prompt (manual run) — nothing reported to Bernstein")
        return 0 if res.status == "done" else 1
    if res.status == "done":
        out = report_complete(info, res.summary, res.files, CHECK_CMD if res.verified else None,
                              0 if res.verified else None)
    elif res.status == "human":
        q = res.summary + (f" Best attempt saved at {res.patch_path}." if res.patch_path else "")
        out = report_refusal(info, q)
    else:
        out = report_fail(info, res.summary)
    for tid, code, body in out:
        log(f"[k2loop] reported {res.status} for {tid}: HTTP {code} {body[:200]}")
    return 0 if res.status in ("done", "human") else 1


if __name__ == "__main__":
    sys.exit(main())
