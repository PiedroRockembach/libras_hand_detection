#!/usr/bin/env python3
"""Download the openly available Libras datasets into LIBRAS_DATA_ROOT/raw/.

Stdlib only, so it runs natively on the host (no container needed). If aria2c is
on PATH it is used for the transfers, with up to 16 connections per file (Zenodo
throttles each connection to a few MB/s); otherwise urllib is used.
Downloads resume from partial files, are verified (MD5 for Zenodo, SHA-256 for
Hugging Face LFS, size otherwise) and are skipped on re-runs once verified.
Each dataset gets a MANIFEST.json with source URLs, sizes, checksums and date.

Usage:
    python3 scripts/download_libras_datasets.py --root /mnt/hd1tb/libras-data [--only minds_rgb ...]

Progress goes to <root>/raw/download.log; stdout only gets one line per dataset.
See docs/datasets/ for licenses and provenance of each source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

CHUNK = 1 << 20
USER_AGENT = "libras-hand-detection-downloader/1.0"
HF_REPO = "MALTA-Lab/MALTA_LIBRAS"
ARIA2C = shutil.which("aria2c")
ARIA2_MAX_CONNECTIONS = 16  # hard upper limit of aria2's --max-connection-per-server

# name -> (kind, source). Only Libras datasets with open, login-free access.
DATASETS: dict[str, tuple[str, str]] = {
    "minds_libras_rgb": ("zenodo", "2667329"),
    "minds_libras_rgbd": ("zenodo", "4322983"),
    "libras_eqt_uece_v4": ("zenodo", "21398895"),
    "lswh100": ("zenodo", "10635628"),
    "malta_libras": ("huggingface", HF_REPO),
    "libras_movement_uci": ("url", "https://archive.ics.uci.edu/static/public/181/libras+movement.zip"),
    "alfabeto_passos_2020": ("url", "https://data.mendeley.com/public-api/zip/k4gs3bmx5k/download/3"),
}

log = logging.getLogger("download")


def open_url(url: str, headers: dict[str, str] | None = None, retries: int = 6):
    """urlopen with retries and exponential backoff (handles 429/5xx)."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    for attempt in range(retries):
        try:
            return urllib.request.urlopen(req, timeout=60)
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                raise
            wait = int(e.headers.get("Retry-After", 0) or 0) or 2 ** (attempt + 2)
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
            wait = 2 ** (attempt + 2)
        log.info("retry %s in %ss", url, wait)
        time.sleep(wait)
    raise RuntimeError("unreachable")


def get_json(url: str):
    with open_url(url) as r:
        return json.load(r), r.headers


def file_hash(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        while chunk := f.read(CHUNK * 8):
            h.update(chunk)
    return h.hexdigest()


def fetch_aria2(url: str, part: Path, size: int | None, connections: int) -> None:
    """Fetch url into part with aria2c, resuming from part (and its .aria2 control file)."""
    control = part.with_name(part.name + ".aria2")
    if part.exists() and not control.exists():
        if size is not None and part.stat().st_size > size:
            part.unlink()
        elif size is not None and part.stat().st_size == size:
            return
    cmd = [
        ARIA2C, url,
        "--dir", str(part.parent), "--out", part.name,
        "--continue=true", "--auto-file-renaming=false", "--allow-overwrite=false",
        f"--max-connection-per-server={connections}", f"--split={connections}",
        "--min-split-size=8M", "--file-allocation=none",
        "--max-tries=30", "--retry-wait=10", "--timeout=60", "--connect-timeout=30",
        f"--user-agent={USER_AGENT}",
        "--console-log-level=warn", "--summary-interval=0", "--download-result=hide",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        tail = " | ".join((r.stdout + r.stderr).strip().splitlines()[-3:])
        raise RuntimeError(f"{part.name}: aria2c exit {r.returncode}: {tail}")


def fetch_urllib(url: str, part: Path, size: int | None) -> None:
    """Fetch url into part with urllib, resuming from the size of part."""
    if part.with_name(part.name + ".aria2").exists():
        # aria2 writes segments out of order, so the size of part is not a resume offset.
        raise RuntimeError(f"{part.name}: partial aria2 download; install aria2c to resume it")
    attempts = 30
    for attempt in range(attempts):
        offset = part.stat().st_size if part.exists() else 0
        if size is not None and offset > size:
            part.unlink()
            offset = 0
        if size is not None and offset == size:
            break
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with open_url(url, headers) as r:
                if offset and r.status != 206:
                    offset = 0  # server ignored Range: restart
                with part.open("ab" if offset else "wb") as f:
                    while chunk := r.read(CHUNK):
                        f.write(chunk)
            if size is None or part.stat().st_size >= size:
                break
            # Server closed the stream early without raising: resume.
            log.info("resume %s at %d/%d bytes", part.name, part.stat().st_size, size)
        except urllib.error.HTTPError as e:
            if e.code == 416 and size is not None and offset == size:
                break  # already complete
            if attempt == attempts - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt == attempts - 1:
                raise
            log.info("resume %s after %s", part.name, e)
        time.sleep(10)


def download(url: str, dest: Path, size: int | None, checksum: str | None, connections: int) -> dict:
    """Download url to dest with resume and verification. checksum is 'algo:hex'."""
    done_marker = dest.with_name(dest.name + ".ok")
    entry = {"file": dest.name, "url": url, "size": size, "checksum": checksum}
    if dest.exists() and done_marker.exists():
        return entry
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")

    if ARIA2C:
        fetch_aria2(url, part, size, connections)
    else:
        fetch_urllib(url, part, size)

    actual = part.stat().st_size
    if size is not None and actual != size:
        raise RuntimeError(f"{dest.name}: size {actual} != expected {size}")
    if checksum:
        algo, expected = checksum.split(":", 1)
        got = file_hash(part, algo)
        if got != expected:
            part.unlink()
            raise RuntimeError(f"{dest.name}: {algo} mismatch ({got} != {expected})")
    part.replace(dest)
    done_marker.touch()
    entry["size"] = actual
    log.info("ok %s (%.1f MB)", dest, actual / 1e6)
    return entry


def zenodo_files(record_id: str) -> tuple[list[tuple[str, str, int, str]], dict]:
    data, _ = get_json(f"https://zenodo.org/api/records/{record_id}")
    meta = data["metadata"]
    if meta.get("access_right") != "open":
        raise RuntimeError(f"Zenodo {record_id} is not open access ({meta.get('access_right')})")
    files = [(f["key"], f["links"]["self"], f["size"], f["checksum"]) for f in data["files"]]
    info = {
        "title": meta.get("title"),
        "doi": data.get("doi"),
        "version": meta.get("version"),
        "license": (meta.get("license") or {}).get("id"),
        "record": f"https://zenodo.org/records/{record_id}",
    }
    return files, info


def hf_files(repo: str) -> tuple[list[tuple[str, str, int, str | None]], dict]:
    url = f"https://huggingface.co/api/datasets/{repo}/tree/main?recursive=true&limit=1000"
    files = []
    while url:
        items, headers = get_json(url)
        for it in items:
            if it["type"] != "file":
                continue
            lfs = it.get("lfs")
            files.append((
                it["path"],
                f"https://huggingface.co/datasets/{repo}/resolve/main/{urllib.parse.quote(it['path'])}",
                lfs["size"] if lfs else it["size"],
                f"sha256:{lfs['oid']}" if lfs else None,
            ))
        m = re.search(r'<([^>]+)>;\s*rel="next"', headers.get("Link", ""))
        url = m.group(1) if m else None
    info_json, _ = get_json(f"https://huggingface.co/api/datasets/{repo}")
    info = {"repo": f"https://huggingface.co/datasets/{repo}", "revision": info_json.get("sha")}
    return files, info


def run_dataset(name: str, kind: str, source: str, raw: Path, workers: int, connections: int) -> str:
    out = raw / name
    if kind == "zenodo":
        files, info = zenodo_files(source)
    elif kind == "huggingface":
        files, info = hf_files(source)
    else:
        fname = "libras_movement.zip" if "uci" in source else f"{name}.zip"
        files, info = [(fname, source, None, None)], {"url": source}

    total = sum(s or 0 for _, _, s, _ in files)
    log.info("%s: %d files, %.1f GB", name, len(files), total / 1e9)
    entries, errors = [], []
    with ThreadPoolExecutor(workers) as pool:
        futs = {
            pool.submit(download, url, out / key, size, ck, connections): key
            for key, url, size, ck in files
        }
        for fut in as_completed(futs):
            try:
                entries.append(fut.result())
            except Exception as e:  # keep going; report at the end
                errors.append(f"{futs[fut]}: {e}")
                log.error("%s: %s", futs[fut], e)

    manifest = {
        "dataset": name,
        "source": info,
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": sorted(entries, key=lambda e: e["file"]),
        "errors": errors,
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    size_gb = sum(e["size"] or 0 for e in entries) / 1e9
    status = "OK" if not errors else f"{len(errors)} ERRORS"
    return f"{name}: {status} ({len(entries)}/{len(files)} files, {size_gb:.1f} GB)"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=os.environ.get("LIBRAS_DATA_ROOT"), help="data root (default: $LIBRAS_DATA_ROOT)")
    p.add_argument("--only", nargs="+", choices=sorted(DATASETS), help="subset of datasets")
    p.add_argument("--workers", type=int, default=8, help="parallel files per dataset")
    p.add_argument(
        "--connections", type=int, default=ARIA2_MAX_CONNECTIONS,
        help=f"aria2c connections per file (1-{ARIA2_MAX_CONNECTIONS}; ignored without aria2c)",
    )
    args = p.parse_args()
    if not args.root:
        p.error("set --root or LIBRAS_DATA_ROOT")
    if not 1 <= args.connections <= ARIA2_MAX_CONNECTIONS:
        p.error(f"--connections must be between 1 and {ARIA2_MAX_CONNECTIONS}")

    raw = Path(args.root).expanduser() / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=raw / "download.log", level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    log.info("transfer: %s", f"aria2c x{args.connections}" if ARIA2C else "urllib")
    failed = False
    for name in args.only or DATASETS:
        kind, source = DATASETS[name]
        hf_workers = max(args.workers, 8) if kind == "huggingface" else args.workers
        try:
            line = run_dataset(name, kind, source, raw, hf_workers, args.connections)
        except Exception as e:
            line = f"{name}: FAILED ({e})"
            log.exception(name)
        failed |= "OK" not in line.split(":", 1)[1][:4]
        print(line, flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
