from __future__ import annotations

import hashlib
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

_ERR = re.compile(r"error|fail|assert|exception|traceback|expected|undefined|not found|timeout", re.I)


@dataclass
class CheckResult:
    passed: bool
    exit_code: int
    output: str
    no_checks: bool
    failures: int
    signature: str
    duration: float


def count_failures(out: str) -> int:
    nums = [int(m) for pat in (r"(\d+) failed", r"(\d+) errors?\b", r"Tests:\s+(\d+) failed", r"(\d+) failing")
            for m in re.findall(pat, out)]
    err_lines = sum(1 for ln in out.splitlines() if re.search(r"\berror\b", ln, re.I))
    return max(nums + [min(err_lines, 999)] + [0])


def signature(out: str) -> str:
    lines = sorted({re.sub(r"0x[0-9a-f]+|\d+|/[^\s:]+/", "#", ln.strip())
                    for ln in out.splitlines() if _ERR.search(ln)})
    return hashlib.sha1("\n".join(lines).encode()).hexdigest()[:12]


def log_tail(out: str, max_chars: int = 4000) -> str:
    lines = out.splitlines()
    keep = [ln for ln in lines if _ERR.search(ln)][-40:] + ["..."] + lines[-40:]
    seen, res = set(), []
    for ln in keep:
        if ln not in seen:
            seen.add(ln)
            res.append(ln[:300])
    txt = "\n".join(res)
    return txt[-max_chars:]


def run_group(cmd: list[str], cwd: Path, timeout: float) -> tuple[int, str, str, bool]:
    """Run cmd in its own process group; on timeout kill the whole group (test servers, pytest workers, npm
    children), not only the direct child. Returns (returncode, stdout, stderr, timed_out)."""
    p = subprocess.Popen(cmd, cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, errors="replace", start_new_session=True)
    try:
        out, err = p.communicate(timeout=timeout)
        return p.returncode, out, err, False
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            out, err = p.communicate(timeout=10)
        except subprocess.TimeoutExpired:  # a grandchild escaped the group and still holds the pipes
            p.kill()
            out, err = "", ""
        return 124, out or "", err or "", True


def run_checks(workdir: Path, timeout: float) -> CheckResult:
    script = workdir / ".k2loop" / "check.sh"
    t0 = time.monotonic()  # a duration: the wall clock jumps (WSL2 resyncs it), shown as "-0.9 sn"
    if not script.exists():
        return CheckResult(True, 0, "K2LOOP_NO_CHECKS (no .k2loop/check.sh)", True, 0, "none", 0.0)
    rc, so, se, timed_out = run_group(["bash", str(script)], workdir, max(30, timeout))
    out = (so + "\n" + se)[-200_000:]
    if timed_out:
        out = f"{out[-100_000:]}\nTIMEOUT: checks exceeded {int(timeout)}s"
    no_checks = "K2LOOP_NO_CHECKS" in out
    passed = rc == 0
    return CheckResult(passed, rc, out, no_checks, 0 if passed else max(1, count_failures(out)),
                       "ok" if passed else signature(out), time.monotonic() - t0)
