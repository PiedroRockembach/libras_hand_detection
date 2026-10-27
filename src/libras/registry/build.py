"""Build the generated registry tables from the source adapters (ADR-0010, ADR-0011)."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

from libras.registry.adapters import eqt_uece, malta, minds
from libras.registry.adapters.base import ScanResult, gloss_text, match_key
from libras.registry.probe import probe_all
from libras.registry.schema import TABLES
from libras.registry.store import CURATED_DIR, Row, generated_dir, read_table, write_table

ADAPTERS = {"minds": minds.scan, "eqt_uece": eqt_uece.scan, "malta": malta.scan}


def scan(data_root: Path, names: list[str]) -> list[ScanResult]:
    return [ADAPTERS[n](data_root) for n in names]


def propose_labels(results: list[ScanResult], curated_dir: Path = CURATED_DIR) -> dict[str, int]:
    """Append unmapped source labels to label_map, reusing or creating glosses.

    Glosses are matched across sources by an accent/punctuation-insensitive key. New
    glosses are marked "auto" in notes so a fluent signer can review spelling and unit.
    """
    glosses = read_table(curated_dir / "glosses.csv", TABLES["glosses"])
    label_map = read_table(curated_dir / "label_map.csv", TABLES["label_map"])
    by_key = {match_key(g["gloss"]): g["gloss_id"] for g in glosses}
    mapped = {(r["source_id"], r["source_label"]) for r in label_map}
    next_id = max((int(g["gloss_id"][1:]) for g in glosses), default=0) + 1
    added = Counter()

    for res in results:
        for label_key in sorted({(s["source_id"], s["source_label"]) for s in res.samples}):
            if label_key in mapped:
                continue
            source_id, label = label_key
            k = match_key(label)
            if k not in by_key:
                text = gloss_text(label)
                by_key[k] = f"G{next_id:05d}"
                glosses.append({"gloss_id": by_key[k], "gloss": text,
                                "unit": "phrase" if " " in text else "sign",
                                "notes": f"auto from {source_id}; review"})
                next_id += 1
                added["glosses"] += 1
            label_map.append({"source_id": source_id, "source_label": label,
                              "gloss_id": by_key[k]})
            mapped.add(label_key)
            added["label_map"] += 1

    write_table(curated_dir / "glosses.csv", TABLES["glosses"], glosses)
    write_table(curated_dir / "label_map.csv", TABLES["label_map"], label_map)
    return dict(added)


def build(data_root: Path, names: list[str], curated_dir: Path = CURATED_DIR,
          probe: bool = True) -> dict:
    """Scan sources, probe videos, resolve labels and write signers/samples tables."""
    results = scan(data_root, names)
    label_map = {(r["source_id"], r["source_label"]): r["gloss_id"]
                 for r in read_table(curated_dir / "label_map.csv", TABLES["label_map"])}
    out_dir = generated_dir(data_root)

    # Keep rows of sources that were not rebuilt this time.
    rebuilt = {sid for res in results for sid in res.source_ids}
    signers = [r for r in read_table(out_dir / "signers.csv", TABLES["signers"])
               if r["source_id"] not in rebuilt]
    samples = [r for r in read_table(out_dir / "samples.csv", TABLES["samples"])
               if r["source_id"] not in rebuilt]

    videos = [data_root / s["path"] for res in results for s in res.samples
              if s["media"] == "video"]
    info = probe_all(videos, data_root / "registry" / "cache" / "probe.jsonl") if probe else {}

    unmapped: Counter = Counter()
    unreadable = 0
    for res in results:
        signers += res.signers
        for s in res.samples:
            gloss_id = label_map.get((s["source_id"], s["source_label"]))
            if gloss_id is None:
                unmapped[(s["source_id"], s["source_label"])] += 1
                continue
            row = {k: v for k, v in s.items() if k != "source_label"} | {"gloss_id": gloss_id}
            meta = info.get(str(data_root / s["path"]))
            if meta and "error" in meta:
                row["quality_flags"] = ";".join(filter(None, [row.get("quality_flags"),
                                                              "unreadable"]))
                unreadable += 1
            elif meta:
                row |= {"fps": f"{meta['fps']:.3f}" if meta["fps"] else "",
                        "width": str(meta["width"] or ""), "height": str(meta["height"] or ""),
                        "n_frames": str(meta["n_frames"] or ""),
                        "video_sha256": meta["sha256"]}
            samples.append(row)

    flag_duplicates(samples)
    write_table(out_dir / "signers.csv", TABLES["signers"], signers)
    write_table(out_dir / "samples.csv", TABLES["samples"], samples)
    with (out_dir / "unmapped_labels.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["source_id", "source_label", "n_samples"])
        w.writerows([*k, n] for k, n in sorted(unmapped.items()))

    return {"samples": Counter(s["source_id"] for s in samples),
            "unmapped_labels": len(unmapped), "unmapped_samples": sum(unmapped.values()),
            "unreadable": unreadable, "duplicate_videos": duplicates(samples)}


def _duplicate_groups(samples: list[Row]) -> list[list[Row]]:
    by_hash: dict[str, list[Row]] = defaultdict(list)
    for s in samples:
        if s.get("video_sha256"):
            by_hash[s["video_sha256"]].append(s)
    return [v for v in by_hash.values() if len(v) > 1]


def flag_duplicates(samples: list[Row]) -> None:
    """Flag byte-identical videos; `label_conflict` when copies carry different glosses."""
    for group in _duplicate_groups(samples):
        conflict = len({s["gloss_id"] for s in group}) > 1
        for s in group:
            flags = set(filter(None, s.get("quality_flags", "").split(";")))
            flags |= {"duplicate_video", "label_conflict"} if conflict else {"duplicate_video"}
            s["quality_flags"] = ";".join(sorted(flags))


def duplicates(samples: list[Row]) -> int:
    """Number of samples whose video bytes also appear in another sample."""
    return sum(len(g) for g in _duplicate_groups(samples))
