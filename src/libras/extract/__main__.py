"""Landmark extraction CLI (ADR-0012). Runs inside the official container.

    python -m libras.extract models --config configs/extraction/mp_v1.yaml
    python -m libras.extract run --config configs/extraction/mp_v1.yaml \
        --sources minds minds_kinect --signers minds:05 minds_kinect:05 [--malta-like minds]
    python -m libras.extract report --config configs/extraction/mp_v1.yaml

Output: landmarks/<version>/<source_id>/<sample>.npz, MANIFEST.json (config, MediaPipe version,
model SHA-256) and stats.csv (per-sample detection metrics). Every run is logged to
PROVENANCE.jsonl.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
import urllib.request
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version as pkg_version
from pathlib import Path

import numpy as np
import yaml

from libras.extract.mediapipe_tasks import (
    Extractor,
    detection_stats,
    read_frames_tensor,
    read_video,
)
from libras.provenance import log_event
from libras.registry.adapters import malta
from libras.registry.adapters.base import match_key
from libras.registry.schema import TABLES
from libras.registry.store import CURATED_DIR, generated_dir, read_table

STATS_FIELDS = ["sample_id", "source_id", "gloss", "n_frames", "any_hand_rate", "both_hands_rate",
                "pose_rate", "face_rate", "missed_hand_rate", "ms_per_frame", "error"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_paths(cfg: dict, root: Path) -> dict[str, Path]:
    return {k: root / "models" / "mediapipe" / Path(url).name for k, url in cfg["models"].items()}


def out_dir(cfg: dict, root: Path) -> Path:
    return root / "landmarks" / cfg["version"]


def npz_path(cfg: dict, root: Path, sample: dict) -> Path:
    safe = sample["sample_id"].replace(":", "__").replace("/", "_")
    return out_dir(cfg, root) / sample["source_id"] / f"{safe}.npz"


def ensure_manifest(cfg: dict, root: Path) -> dict:
    """Write the version manifest, or refuse to continue if it describes other settings."""
    paths = model_paths(cfg, root)
    manifest = {"config": cfg, "mediapipe": pkg_version("mediapipe"),
                "opencv": pkg_version("opencv-python"),
                "models": {k: {"file": p.name, "sha256": sha256(p)} for k, p in paths.items()}}
    path = out_dir(cfg, root) / "MANIFEST.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old != manifest:
            raise SystemExit(f"{path} describes different settings; use a new version name")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2))
    return manifest


def select_samples(root: Path, sources: list[str], signers: list[str] | None,
                   limit: int | None, malta_like: str | None) -> list[dict]:
    glosses = {g["gloss_id"]: g["gloss"] for g in read_table(CURATED_DIR / "glosses.csv",
                                                            TABLES["glosses"])}
    rows = [s | {"gloss": glosses[s["gloss_id"]]}
            for s in read_table(generated_dir(root) / "samples.csv", TABLES["samples"])
            if s["source_id"] in sources and "label_conflict" not in s["quality_flags"]
            and (not signers or s["signer_uid"] in signers)]
    if malta_like:  # MALTA clips (not in the registry yet) whose label matches SOURCE's glosses
        wanted = {match_key(glosses[s["gloss_id"]]): glosses[s["gloss_id"]]
                  for s in read_table(generated_dir(root) / "samples.csv", TABLES["samples"])
                  if s["source_id"] == malta_like}
        for s in malta.scan(root).samples:
            if (k := match_key(s["source_label"])) in wanted:
                rows.append(s | {"gloss": wanted[k]})
    rows.sort(key=lambda s: s["sample_id"])
    return rows[:limit] if limit else rows


_extractor: Extractor | None = None


def _init_worker(cfg: dict, paths: dict[str, Path]) -> None:
    global _extractor
    _extractor = Extractor(cfg, paths)


def _extract_one(sample: dict, root: str, target: str) -> dict:
    stats = {"sample_id": sample["sample_id"], "source_id": sample["source_id"],
             "gloss": sample["gloss"], "error": ""}
    try:
        src = Path(root) / sample["path"]
        reader = read_frames_tensor if sample["media"] == "frames_tensor" else read_video
        lm = _extractor.run(*reader(src))
        tmp = Path(target).with_suffix(".tmp.npz")
        np.savez_compressed(tmp, sample_id=sample["sample_id"], **lm)
        tmp.replace(target)
        stats |= detection_stats(lm)
    except Exception as e:  # noqa: BLE001 - recorded per sample; one bad file must not stop the run
        stats["error"] = f"{type(e).__name__}: {e}"
    return stats


def cmd_models(cfg: dict, root: Path) -> None:
    fetched = {}
    for name, path in model_paths(cfg, root).items():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(cfg["models"][name], tmp)
            tmp.replace(path)
        fetched[name] = {"file": path.name, "sha256": sha256(path)}
        print(f"{name}: {path} {fetched[name]['sha256'][:16]}")
    log_event(root, "extract.models", inputs=list(cfg["models"].values()),
              outputs=[str(p.relative_to(root)) for p in model_paths(cfg, root).values()],
              result=fetched)


def cmd_run(cfg: dict, root: Path, args: argparse.Namespace) -> None:
    ensure_manifest(cfg, root)
    samples = select_samples(root, args.sources, args.signers, args.limit, args.malta_like)
    todo = [s for s in samples if not npz_path(cfg, root, s).exists()]
    for s in todo:
        npz_path(cfg, root, s).parent.mkdir(parents=True, exist_ok=True)
    stats_file = out_dir(cfg, root) / "stats.csv"
    new_file = not stats_file.exists()
    errors = 0
    print(f"{len(samples)} selected, {len(todo)} to extract, {args.workers} workers", flush=True)
    with ProcessPoolExecutor(args.workers, initializer=_init_worker,
                             initargs=(cfg, model_paths(cfg, root))) as pool, \
            stats_file.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=STATS_FIELDS, extrasaction="ignore")
        if new_file:
            w.writeheader()
        futures = [pool.submit(_extract_one, s, str(root), str(npz_path(cfg, root, s)))
                   for s in todo]
        for n, fut in enumerate(as_completed(futures), start=1):
            row = fut.result()
            errors += bool(row["error"])
            w.writerow(row)
            f.flush()
            if n % 50 == 0 or n == len(futures):
                print(f"  {n}/{len(futures)} done, {errors} errors", flush=True)
    log_event(root, "extract.run", inputs=[f"registry/current/samples.csv ({len(samples)})"],
              outputs=[str(out_dir(cfg, root).relative_to(root))],
              result={"selected": len(samples), "extracted": len(todo) - errors,
                      "errors": errors, "sources": args.sources, "signers": args.signers,
                      "malta_like": args.malta_like})


def cmd_report(cfg: dict, root: Path) -> None:
    rows = list(csv.DictReader((out_dir(cfg, root) / "stats.csv").open()))
    by_source = defaultdict(list)
    for r in rows:
        by_source[r["source_id"]].append(r)
    metrics = ["any_hand_rate", "both_hands_rate", "pose_rate", "face_rate",
               "missed_hand_rate", "ms_per_frame"]
    print(f"{'source':14} {'n':>5} {'err':>4} " + " ".join(f"{m:>16}" for m in metrics))
    for src, rs in sorted(by_source.items()):
        ok = [r for r in rs if not r["error"]]
        vals = []
        for m in metrics:
            xs = [float(r[m]) for r in ok if r[m] not in ("", "nan")]
            vals.append(f"{statistics.median(xs):7.3f} ({statistics.mean(xs):6.3f})" if xs
                        else f"{'-':>16}")
        print(f"{src:14} {len(rs):5d} {len(rs) - len(ok):4d} " + " ".join(f"{v:>16}" for v in vals))
    print("values: median (mean) per sample")


def main() -> int:
    p = argparse.ArgumentParser(prog="python -m libras.extract")
    p.add_argument("cmd", choices=["models", "run", "report"])
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=os.environ.get("LIBRAS_DATA_ROOT"))
    p.add_argument("--sources", nargs="+", default=["minds"])
    p.add_argument("--signers", nargs="+")
    p.add_argument("--malta-like", metavar="SOURCE",
                   help="also extract MALTA clips whose label matches a gloss of SOURCE")
    p.add_argument("--limit", type=int)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = p.parse_args()
    if not args.data_root:
        p.error("set --data-root or LIBRAS_DATA_ROOT")
    cfg = yaml.safe_load(args.config.read_text())
    root = args.data_root.resolve()
    {"models": lambda: cmd_models(cfg, root), "run": lambda: cmd_run(cfg, root, args),
     "report": lambda: cmd_report(cfg, root)}[args.cmd]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
