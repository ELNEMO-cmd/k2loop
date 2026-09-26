"""Heartbeats so Bernstein does not reap a long-running ToT worker."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path


def find_orchestrator_root(cwd: Path) -> Path | None:
    for p in cwd.resolve().parents:
        sdd = p / ".sdd"
        if (sdd / "runtime" / "server.port").exists() or (sdd / "backlog").is_dir():
            return p
    return None


class Heartbeat(threading.Thread):
    def __init__(self, paths: list[Path], interval: float = 15.0, echo=print, shutdown_file: Path | None = None):
        super().__init__(daemon=True)
        self.paths = paths
        self.shutdown_file = shutdown_file
        self.shutdown = False
        self.interval = interval
        self.echo = echo
        self.phase, self.pct, self.msg = "starting", 0, "starting"
        self._stop = threading.Event()

    def set(self, phase: str, pct: int, msg: str) -> None:
        self.phase, self.pct, self.msg = phase, max(0, min(100, int(pct))), msg[:200]
        self.echo(f"[k2loop] {phase} {self.pct}% {msg}")
        self._write()

    def _write(self) -> None:
        payload = json.dumps({"timestamp": int(time.time()), "phase": "implementing", "progress_pct": self.pct,
                              "current_file": "", "message": f"{self.phase}: {self.msg}"})
        for p in self.paths:
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(payload, encoding="utf-8")
            except OSError:
                pass

    def run(self) -> None:
        while not self._stop.wait(self.interval):
            self._write()
            if self.shutdown_file is not None and self.shutdown_file.exists() and not self.shutdown:
                self.shutdown = True
                self.echo("[k2loop] SHUTDOWN signal from Bernstein: finishing as soon as possible")
            self.echo(f"[k2loop] alive {self.phase} {self.pct}%")

    def stop(self) -> None:
        self._stop.set()
