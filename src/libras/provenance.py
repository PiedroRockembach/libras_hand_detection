"""Append-only provenance log for every operation that changes data in LIBRAS_DATA_ROOT.

Each entry records when, from which commit (and whether the tree was dirty), which
command, and what was read and written, so the data state can be replayed from scratch.
Stdlib only, so the host-side scripts can use it too.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_NAME = "PROVENANCE.jsonl"


def git_state() -> dict:
    def git(*args: str) -> str:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), *args],
                             capture_output=True, text=True, check=False)
        return out.stdout.strip()

    return {"commit": git("rev-parse", "HEAD") or None,
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def log_event(data_root: Path, step: str, inputs: list[str], outputs: list[str],
              result: dict | None = None, command: list[str] | None = None) -> None:
    entry = {
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
        "step": step,
        "command": command if command is not None else sys.argv,
        **git_state(),
        "host": platform.node(),
        "inputs": inputs,
        "outputs": outputs,
        "result": result or {},
    }
    with (Path(data_root) / LOG_NAME).open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
