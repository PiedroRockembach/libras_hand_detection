#!/usr/bin/env python3
"""Package the demo model as a GitHub Release asset, and fetch it on another machine (ADR-0018).

The demo model is a trained run under LIBRAS_DATA_ROOT/experiments/ and stays out of git
(ADR-0005). It is distributed as a .tar.gz attached to a GitHub Release. The run, the release
tag, the asset name and the SHA-256 hashes are versioned in configs/demo/<version>.json.
Stdlib only and platform-independent, so it also runs on the Windows demo machine:

    # training machine: build the asset and write its hashes into the config
    python3 scripts/demo_model.py pack --config configs/demo/v0.json --root /mnt/hd1tb/libras-data

    # demo machine (Windows or Linux), from the cloned repo
    uv run python scripts/demo_model.py fetch --config configs/demo/v0.json --root C:\\libras-data

pack is deterministic (fixed mtimes and owners in the tar and gzip headers): the same run files
always give the same SHA-256. fetch checks the archive and every file before writing, writes
only the files named in the config (never paths taken from the archive), skips files already
present with the right hash, and logs to PROVENANCE.jsonl.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import sys
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from libras.provenance import log_event

USER_AGENT = "libras-demo-model-fetch"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_dir(cfg: dict, root: Path) -> Path:
    return root.joinpath(*PurePosixPath(cfg["run"]).parts)


def member_name(cfg: dict, file: str) -> str:
    return f"{cfg['run']}/{file}"


def asset_url(cfg: dict) -> str:
    r = cfg["release"]
    return f"https://github.com/{r['repo']}/releases/download/{r['tag']}/{r['asset']}"


def build_archive(cfg: dict, root: Path) -> tuple[bytes, dict[str, str]]:
    src = run_dir(cfg, root)
    hashes: dict[str, str] = {}
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0) as gz, \
            tarfile.open(fileobj=gz, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for file in sorted(cfg["files"]):
            data = (src / file).read_bytes()
            info = tarfile.TarInfo(member_name(cfg, file))
            info.size, info.mode, info.mtime = len(data), 0o644, 0
            tar.addfile(info, io.BytesIO(data))
            hashes[file] = sha256_bytes(data)
    return buf.getvalue(), hashes


def cmd_pack(cfg_path: Path, cfg: dict, root: Path, out_dir: Path) -> int:
    archive, hashes = build_archive(cfg, root)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / cfg["release"]["asset"]
    out.write_bytes(archive)
    cfg["sha256"], cfg["files_sha256"] = sha256_bytes(archive), hashes
    shown = str(out.relative_to(root)) if out.is_relative_to(root) else str(out)
    # logged before the config is rewritten, so the recorded git state is the clean commit
    log_event(root, "demo_model.pack", inputs=[member_name(cfg, f) for f in sorted(cfg["files"])],
              outputs=[shown], result={"sha256": cfg["sha256"], "size": len(archive),
                                       "files_sha256": hashes})
    cfg_path.write_text(json.dumps(cfg, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{out}\n  sha256 {cfg['sha256']}  ({len(archive) / 1e6:.1f} MB)\n"
          f"  hashes written to {cfg_path}\n"
          f"Attach it to the release '{cfg['release']['tag']}' of {cfg['release']['repo']}.")
    return 0


def cmd_fetch(cfg: dict, root: Path, url: str) -> int:
    expected: dict[str, str] | None = cfg.get("files_sha256")
    if not expected or not cfg.get("sha256"):
        sys.exit("the config has no hashes yet: run 'pack' on the training machine first")
    dst = run_dir(cfg, root)
    if all((dst / f).is_file() and sha256_file(dst / f) == h for f, h in expected.items()):
        print(f"already present and verified: {dst}")
        return 0

    print(f"downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as r:
        archive = r.read()
    got = sha256_bytes(archive)
    if got != cfg["sha256"]:
        sys.exit(f"archive SHA-256 mismatch: expected {cfg['sha256']}, got {got}")

    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        members = {m.name: m for m in tar.getmembers()}
        wanted = {member_name(cfg, f) for f in expected}
        if set(members) != wanted or not all(m.isfile() for m in members.values()):
            sys.exit(f"unexpected archive contents: {sorted(members)}")
        contents = {f: tar.extractfile(members[member_name(cfg, f)]).read() for f in expected}
    for f, data in contents.items():
        if sha256_bytes(data) != expected[f]:
            sys.exit(f"SHA-256 mismatch for {f}")

    dst.mkdir(parents=True, exist_ok=True)
    for f, data in contents.items():
        tmp = dst / f"{f}.part"
        tmp.write_bytes(data)
        tmp.replace(dst / f)
    log_event(root, "demo_model.fetch", inputs=[url],
              outputs=[member_name(cfg, f) for f in sorted(expected)],
              result={"sha256": got, "release": cfg["release"]})
    print(f"model ready: {dst}\nrun the demo with:\n"
          f"  uv run python -m libras.demo live --run {cfg['run']}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="scripts/demo_model.py")
    p.add_argument("cmd", choices=["pack", "fetch"])
    p.add_argument("--config", type=Path, required=True, help="configs/demo/<version>.json")
    p.add_argument("--root", type=Path, default=os.environ.get("LIBRAS_DATA_ROOT"),
                   help="data root (default: LIBRAS_DATA_ROOT)")
    p.add_argument("--out", type=Path, help="pack: output directory (default: <root>/releases)")
    p.add_argument("--url", help="fetch: download from here instead of the GitHub release")
    args = p.parse_args()
    if not args.root:
        p.error("set --root or LIBRAS_DATA_ROOT")
    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    if args.cmd == "pack":
        return cmd_pack(args.config, cfg, root, args.out or root / "releases")
    return cmd_fetch(cfg, root, args.url or asset_url(cfg))


if __name__ == "__main__":
    sys.exit(main())
