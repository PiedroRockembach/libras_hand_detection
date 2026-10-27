"""Video inventory: fps, resolution, frame count and SHA-256 via ffprobe, with a cache.

The cache (registry/cache/probe.jsonl) is keyed by path, size and mtime, so re-running a
build only probes new or changed files.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CHUNK = 8 << 20


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _fps(rate: str) -> float | None:
    num, _, den = rate.partition("/")
    try:
        return float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        return None


def probe_video(path: Path) -> dict:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
           "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,nb_read_packets",
           "-of", "json", str(path)]
    out = subprocess.run(cmd, capture_output=True, text=True, check=False)
    streams = json.loads(out.stdout or "{}").get("streams") or []
    if out.returncode != 0 or not streams:
        return {"error": (out.stderr.strip().splitlines() or ["no video stream"])[-1]}
    s = streams[0]
    return {
        "width": s.get("width"), "height": s.get("height"),
        "fps": _fps(s.get("avg_frame_rate", "")) or _fps(s.get("r_frame_rate", "")),
        "n_frames": int(s["nb_read_packets"]) if s.get("nb_read_packets") else None,
        "sha256": sha256(path),
    }


def probe_all(paths: list[Path], cache_file: Path, workers: int = 8) -> dict[str, dict]:
    """Return {str(path): info}; info has 'error' when the file is unreadable."""
    cache: dict[str, dict] = {}
    if cache_file.exists():
        for line in cache_file.read_text().splitlines():
            entry = json.loads(line)
            cache[entry["key"]] = entry["info"]

    def key(p: Path) -> str:
        st = p.stat()
        return f"{p}|{st.st_size}|{int(st.st_mtime)}"

    todo = [p for p in paths if key(p) not in cache]
    if todo:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(workers) as pool, cache_file.open("a") as f:
            for p, info in zip(todo, pool.map(probe_video, todo)):
                cache[key(p)] = info
                f.write(json.dumps({"key": key(p), "info": info}) + "\n")
    return {str(p): cache[key(p)] for p in paths}
