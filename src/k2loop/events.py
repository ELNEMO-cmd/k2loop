"""Append-only event log that the UI tails (agent steps, tool calls, ToT progress, live model output).

One JSON object per line in <data>/agent/events.jsonl. Several processes may write (the agent daemon, a
Bernstein worker, `k2loop solve`), so ids come from a counter file guarded by an flock.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover  (Windows: single process only)
    fcntl = None  # type: ignore[assignment]

ROTATE_BYTES = 50 * 1024 * 1024
_context: dict[str, Any] = {}  # e.g. {"goal": "g-..."} while the agent works on a goal (one goal at a time)


def set_context(**fields: Any) -> None:
    _context.clear()
    _context.update({k: v for k, v in fields.items() if v is not None})


class EventLog:
    def __init__(self, directory: Path):
        self.dir = directory
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "events.jsonl"
        self.seq_path = self.dir / "events.seq"
        self._lock = threading.Lock()

    def emit(self, kind: str, **data: Any) -> dict:
        with self._lock, open(self.seq_path, "a+") as seqf:
            if fcntl:
                fcntl.flock(seqf, fcntl.LOCK_EX)
            seqf.seek(0)
            try:
                seq = int(seqf.read().strip() or 0) + 1
            except ValueError:
                seq = 1
            seqf.seek(0)
            seqf.truncate()
            seqf.write(str(seq))
            seqf.flush()
            ev = {"id": seq, "ts": round(time.time(), 3), "kind": kind, **_context, **data}
            if self.path.exists() and self.path.stat().st_size > ROTATE_BYTES:
                os.replace(self.path, self.path.with_suffix(".jsonl.1"))
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(ev, ensure_ascii=False, default=str) + "\n")
            if fcntl:
                fcntl.flock(seqf, fcntl.LOCK_UN)
        return ev

    def last_id(self) -> int:
        try:
            return int(self.seq_path.read_text(encoding="utf-8").strip() or 0)
        except (OSError, ValueError):
            return 0

    def history(self, after: int = 0, limit: int = 1500, tail_bytes: int = 4_000_000) -> list[dict]:
        """Recent events with id > after. Old live-output chunks are dropped to keep the page light."""
        if not self.path.exists():
            return []
        size = self.path.stat().st_size
        with open(self.path, "rb") as f:
            f.seek(max(0, size - tail_bytes))
            raw = f.read().decode("utf-8", errors="ignore").splitlines()
        if size > tail_bytes and raw:
            raw = raw[1:]  # first line is probably cut
        evs = []
        for ln in raw:
            try:
                ev = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if ev.get("id", 0) > after:
                evs.append(ev)
        evs = evs[-limit:]
        keep_from = max(0, len(evs) - 400)
        return [e for i, e in enumerate(evs) if i >= keep_from or e.get("kind") not in ("delta", "stream")]

    def follow(self, after: int, stop: threading.Event, poll: float = 0.3,
               backlog_bytes: int = 2_000_000) -> Iterator[dict | None]:
        """Yield events with id > after, then new ones forever (None = idle tick, for keep-alives).
        Only the last `backlog_bytes` of the file are scanned for older events."""
        f = None
        ino = None
        first = True
        try:
            while not stop.is_set():
                if f is None:
                    if not self.path.exists():
                        time.sleep(poll)
                        yield None
                        continue
                    f = open(self.path, "rb")
                    ino = os.fstat(f.fileno()).st_ino
                    size = os.fstat(f.fileno()).st_size
                    if first and size > backlog_bytes:
                        f.seek(size - backlog_bytes)
                        f.readline()  # drop the cut line
                    first = False
                line = f.readline()
                if line.endswith(b"\n"):
                    try:
                        ev = json.loads(line.decode("utf-8", errors="ignore"))
                    except json.JSONDecodeError:
                        continue
                    if ev.get("id", 0) > after:
                        after = ev["id"]
                        yield ev
                    continue
                if line:  # partial line (writer is mid-write): rewind and wait
                    f.seek(-len(line), os.SEEK_CUR)
                try:
                    if os.stat(self.path).st_ino != ino:  # rotated
                        f.close()
                        f = None
                        continue
                except FileNotFoundError:
                    pass
                time.sleep(poll)
                yield None
        finally:
            if f:
                f.close()


_LOGS: dict[str, EventLog] = {}


def event_log(cfg: dict) -> EventLog:
    d = str(Path(cfg["paths"]["data_dir"]) / "agent")
    if d not in _LOGS:
        _LOGS[d] = EventLog(Path(d))
    return _LOGS[d]
