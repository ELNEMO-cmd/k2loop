"""GPU swap: 12 GB VRAM holds either K2 (small, fast) or Qwen3.8 27B (strong, partially in RAM).
`with strong_model(cfg):` stops K2, starts the strong server, yields, then restores K2."""
from __future__ import annotations

import contextlib
import fcntl
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Iterator

from .config import data_dir


def _env() -> dict[str, str]:
    env = dict(os.environ)
    rt = f"/run/user/{os.getuid()}"
    env.setdefault("XDG_RUNTIME_DIR", rt)  # Bernstein filters env; systemctl --user needs these
    env.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path={rt}/bus")
    return env


def systemctl(action: str, unit: str) -> bool:
    if not shutil.which("systemctl"):
        return False
    return subprocess.run(["systemctl", "--user", action, unit], env=_env(), capture_output=True).returncode == 0


def _healthy(base_url: str) -> bool:
    root = base_url.rstrip("/").removesuffix("/v1")
    for path in ("/health", "/v1/models"):
        try:
            with urllib.request.urlopen(root + path, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:  # noqa: BLE001
            continue
    return False


def wait_healthy(base_url: str, timeout: float, log=print) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if _healthy(base_url):
            return True
        time.sleep(5)
    log(f"[k2loop] {base_url} did not become healthy in {int(timeout)}s")
    return False


@contextlib.contextmanager
def strong_model(cfg: dict, log=print) -> Iterator[bool]:
    """Yields True when the strong model is reachable."""
    s = cfg["strong"]
    if not s.get("swap", True):          # both servers managed elsewhere (or enough VRAM)
        yield wait_healthy(s["base_url"], 60, log)
        return
    lock_path = data_dir(cfg) / "gpu.lock"
    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        log(f"[k2loop] GPU swap: K2 -> {s['model']} (loading, a few minutes)")
        systemctl("stop", s["small_service"])
        time.sleep(3)
        started = systemctl("start", s["strong_service"])
        ok = started and wait_healthy(s["base_url"], float(s.get("load_timeout_s", 1800)), log)
        try:
            yield ok
        finally:
            log(f"[k2loop] GPU swap: {s['model']} -> K2")
            systemctl("stop", s["strong_service"])
            time.sleep(3)
            systemctl("start", s["small_service"])
            wait_healthy(cfg["llm"]["base_url"], 900, log)
            fcntl.flock(lock, fcntl.LOCK_UN)
