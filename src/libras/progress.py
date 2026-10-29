"""Progress reporting for long tasks, read by the `task-progress` Claude Code mod.

Writes $CLAUDE_PROGRESS_DIR/<task_id>.json (default ~/.cache/claude-progress), atomically and
at most once per second, plus once at the end:

    {"name": ..., "done": 12, "total": 100, "unit": "videos",
     "started_at": <ms>, "updated_at": <ms>, "status": "running" | "done" | "error"}

Any program may write the same file; this module is only a convenience. Stdlib only, so the
host-side scripts can use it too. Reporting never raises: a task must not fail because its
progress could not be shown.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Self

MIN_INTERVAL_S = 1.0


def progress_dir() -> Path:
    return Path(os.environ.get("CLAUDE_PROGRESS_DIR") or Path.home() / ".cache" / "claude-progress")


class Progress:
    def __init__(self, task_id: str, name: str, total: int = 0, unit: str = ""):
        self.path = progress_dir() / f"{task_id}.json"
        self.name, self.total, self.unit = name, total, unit
        self.done = 0
        self.started_at = time.time()
        self._last_write = 0.0
        self._write("running", force=True)

    def advance(self, n: int = 1) -> None:
        self.done += n
        self._write("running")

    def set(self, done: int, total: int | None = None) -> None:
        self.done = done
        if total is not None:
            self.total = total
        self._write("running")

    def finish(self, ok: bool = True) -> None:
        self._write("done" if ok else "error", force=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, *_) -> None:
        self.finish(ok=exc_type is None)

    def _write(self, status: str, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_write < MIN_INTERVAL_S:
            return
        self._last_write = now
        data = {"name": self.name, "done": self.done, "total": self.total, "unit": self.unit,
                "started_at": int(self.started_at * 1000), "updated_at": int(now * 1000),
                "status": status}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False))
            tmp.replace(self.path)
        except OSError:
            pass
