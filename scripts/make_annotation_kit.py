#!/usr/bin/env python3
"""Build an annotation kit for a fluent Libras annotator (ADR-0014).

Creates <data-root>/annotation/<name>/ with:
  revisao_glosas.csv    one row per gloss: current spelling/unit + columns to correct them
  anotacao_frases.csv   one row per phrase: Portuguese text + column for the Libras gloss sequence
  clips/                example videos copied from the registry (different signers, no label_conflict)
  ORIGEM_DOS_CLIPES.csv clip file -> registry sample_id and source path
The guide (docs/anotacao/GUIA_DE_ANOTACAO.md) is copied in as well. CSVs are UTF-8 with BOM so
spreadsheet programs open the accents correctly. Deterministic; logs to PROVENANCE.jsonl.

Usage:
    python3 scripts/make_annotation_kit.py --name kit-v2 [--data-root PATH]
    python3 scripts/make_annotation_kit.py --name kit-v1 --discard "reason"   # undelivered kits only
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from libras.provenance import log_event  # noqa: E402
from libras.registry.store import CURATED_DIR, generated_dir, read_table  # noqa: E402
from libras.registry.schema import TABLES  # noqa: E402

GUIDE = REPO / "docs" / "anotacao" / "GUIA_DE_ANOTACAO.md"
SIGN_EXAMPLES, PHRASE_EXAMPLES = 2, 3
# Prefer the Canon camera for MINDS examples (the Kinect is another view of the same take).
SOURCE_PREFERENCE = {"minds": 0, "eqt_uece": 0, "minds_kinect": 1}


def slug(text: str) -> str:
    return re.sub(r"[^\w]+", "_", text, flags=re.UNICODE).strip("_")[:40]


def pick_examples(samples: list[dict], n: int) -> list[dict]:
    """Deterministic: one sample per signer, preferred sources first, then by sample_id."""
    chosen, seen = [], set()
    for s in sorted(samples, key=lambda s: (SOURCE_PREFERENCE.get(s["source_id"], 9),
                                            s["signer_uid"], s["sample_id"])):
        if s["signer_uid"] not in seen:
            chosen.append(s)
            seen.add(s["signer_uid"])
        if len(chosen) == n:
            break
    return chosen


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--data-root", type=Path, default=os.environ.get("LIBRAS_DATA_ROOT"))
    p.add_argument("--discard", metavar="REASON",
                   help="delete the (undelivered) kit --name instead of building it, logging REASON")
    args = p.parse_args()
    if not args.data_root:
        p.error("set --data-root or LIBRAS_DATA_ROOT")
    root = args.data_root.resolve()
    out = root / "annotation" / args.name
    if args.discard:
        if not out.is_dir():
            p.error(f"{out} does not exist")
        shutil.rmtree(out)
        print(f"discarded {out}")
        log_event(root, "annotation.discard_kit", inputs=[str(out.relative_to(root))], outputs=[],
                  result={"reason": args.discard})
        return 0
    if out.exists():
        p.error(f"{out} already exists; kits are immutable, use a new --name")

    glosses = read_table(CURATED_DIR / "glosses.csv", TABLES["glosses"])
    label_map = read_table(CURATED_DIR / "label_map.csv", TABLES["label_map"])
    samples = read_table(generated_dir(root) / "samples.csv", TABLES["samples"])

    labels = defaultdict(set)
    for r in label_map:
        labels[r["gloss_id"]].add(f'{r["source_id"]}: {r["source_label"]}')
    by_gloss, conflicted = defaultdict(list), defaultdict(list)
    for s in samples:
        if s["media"] != "video":
            continue
        (conflicted if "label_conflict" in s["quality_flags"] else by_gloss)[s["gloss_id"]].append(s)
    conflicts = {gid: len(v) for gid, v in conflicted.items()}

    (out / "clips").mkdir(parents=True)
    origin, review_rows, phrase_rows = [], [], []
    for g in glosses:
        gid, unit = g["gloss_id"], g["unit"]
        n = PHRASE_EXAMPLES if unit == "phrase" else SIGN_EXAMPLES
        clips = []
        # Glosses whose every video is in conflict still need examples so the annotator can judge.
        pool = by_gloss[gid] or conflicted[gid]
        for i, s in enumerate(pick_examples(pool, n), start=1):
            src = root / s["path"]
            name = f"{gid}_{slug(g['gloss'])}_{i}{src.suffix}"
            shutil.copy2(src, out / "clips" / name)
            clips.append(f"clips/{name}")
            origin.append([f"clips/{name}", s["sample_id"], s["path"]])
        clips += [""] * (n - len(clips))
        note = (f"ATENÇÃO: {conflicts[gid]} vídeos desta glosa são idênticos aos de outra glosa; "
                "dizer qual rótulo está correto" if conflicts.get(gid) else "")
        if not by_gloss[gid] and conflicts.get(gid):
            note += " (os exemplos abaixo são desses vídeos em conflito)"
        review_rows.append([gid, g["gloss"], "sinal" if unit == "sign" else "frase",
                            " | ".join(sorted(labels[gid])), clips[0], clips[1], note,
                            "", "", "", ""])
        if unit == "phrase":
            phrase_rows.append([gid, g["gloss"], str(len(pool)), *clips[:3], note, "", ""])

    write_csv(out / "revisao_glosas.csv",
              ["gloss_id", "glosa_atual", "tipo_atual", "rotulos_originais", "video_1", "video_2",
               "aviso", "glosa_corrigida", "tipo_corrigido", "variante_regional", "observacoes"],
              review_rows)
    write_csv(out / "anotacao_frases.csv",
              ["gloss_id", "frase_portugues", "n_videos", "video_1", "video_2", "video_3", "aviso",
               "sequencia_glosas_libras", "observacoes"], phrase_rows)
    write_csv(out / "ORIGEM_DOS_CLIPES.csv", ["clip", "sample_id", "source_path"], origin)
    shutil.copy2(GUIDE, out / GUIDE.name)

    result = {"glosses": len(review_rows), "phrases": len(phrase_rows), "clips": len(origin)}
    print(f"{out}: {result}")
    log_event(root, "annotation.make_kit",
              inputs=["metadata/registry/glosses.csv", "metadata/registry/label_map.csv",
                      "registry/current/samples.csv"],
              outputs=[str(out.relative_to(root))], result=result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
