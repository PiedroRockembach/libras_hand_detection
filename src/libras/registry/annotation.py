"""Import a filled annotation kit (scripts/make_annotation_kit.py) into the curated tables.

Task 1 (revisao_glosas.csv) corrects gloss spelling, unit and regional variant.
Task 2 (anotacao_frases.csv) gives each phrase its sequence of Libras glosses; tokens that are
not yet glosses (e.g. fingerspelling P-I-X, IX(...), CL(...)) become new sign glosses.
Nothing is written unless apply=True; the report lists every change and every problem.
"""

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from libras.registry.adapters.base import match_key
from libras.registry.schema import TABLES
from libras.registry.store import CURATED_DIR, Row, read_table, write_table

TOKEN_RE = re.compile(r"^(\?|IX\([^()]+\)|CL\([^()]+\)|[A-ZÀ-Ý0-9]+(?:-[A-ZÀ-Ý0-9]+)*)(\+?)$")
UNIT = {"sinal": "sign", "frase": "phrase", "": ""}


@dataclass
class ImportReport:
    changes: list[str] = field(default_factory=list)
    new_glosses: list[str] = field(default_factory=list)
    follow_up: list[str] = field(default_factory=list)  # needs a human decision (e.g. label conflicts)
    errors: list[str] = field(default_factory=list)


def read_kit_csv(path: Path) -> list[Row]:
    """Read a CSV saved by LibreOffice/Excel: UTF-8 (with or without BOM), ',' or ';'."""
    text = path.read_text(encoding="utf-8-sig")
    dialect = csv.Sniffer().sniff(text.splitlines()[0], delimiters=",;")
    return [{k: (v or "").strip() for k, v in r.items()} for r in csv.DictReader(
        text.splitlines(), dialect=dialect)]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def import_kit(kit: Path, annotator: str, curated_dir: Path = CURATED_DIR,
               apply: bool = False) -> ImportReport:
    rep = ImportReport()
    glosses = read_table(curated_dir / "glosses.csv", TABLES["glosses"])
    phrases = read_table(curated_dir / "phrase_glosses.csv", TABLES["phrase_glosses"])
    by_id = {g["gloss_id"]: g for g in glosses}
    tag = f"revisado em {kit.name} por {annotator}"

    # Task 1: gloss review.
    for i, r in enumerate(read_kit_csv(kit / "revisao_glosas.csv"), start=2):
        g = by_id.get(r["gloss_id"])
        if g is None:
            rep.errors.append(f"revisao_glosas:{i}: unknown gloss_id {r['gloss_id']!r}")
            continue
        if r["glosa_atual"] != g["gloss"]:
            rep.errors.append(f"revisao_glosas:{i}: {g['gloss_id']} changed since the kit was made "
                              f"({r['glosa_atual']!r} -> now {g['gloss']!r}); re-check by hand")
            continue
        answers = [r["glosa_corrigida"], r["tipo_corrigido"], r["variante_regional"],
                   r["observacoes"]]
        if not any(answers):
            continue  # not reviewed yet (a correct gloss is marked "OK" in observacoes)
        unit = UNIT.get(r["tipo_corrigido"].lower())
        if unit is None:
            rep.errors.append(f"revisao_glosas:{i}: tipo_corrigido must be sinal or frase")
            continue
        new_text = r["glosa_corrigida"].upper()
        if new_text and new_text != g["gloss"]:
            clash = [o["gloss_id"] for o in glosses
                     if o is not g and match_key(o["gloss"]) == match_key(new_text)]
            if clash:
                rep.errors.append(f"revisao_glosas:{i}: {new_text!r} would duplicate {clash}; "
                                  "merging glosses needs a manual decision")
                continue
            rep.changes.append(f"{g['gloss_id']}: gloss {g['gloss']!r} -> {new_text!r}")
            g["gloss"] = new_text
        if unit and unit != g["unit"]:
            rep.changes.append(f"{g['gloss_id']}: unit {g['unit']} -> {unit}")
            g["unit"] = unit
        if r["variante_regional"] and r["variante_regional"] != g["variant"]:
            rep.changes.append(f"{g['gloss_id']}: variant -> {r['variante_regional']!r}")
            g["variant"] = r["variante_regional"]
        if r["aviso"]:
            rep.follow_up.append(f"{g['gloss_id']} ({g['gloss']}): {r['aviso']} | "
                                 f"resposta: {r['observacoes'] or '(vazia)'}")
        obs = "" if r["observacoes"].upper() == "OK" else r["observacoes"]
        g["notes"] = "; ".join(filter(None, [g["notes"].replace("; review", ""), tag, obs]))
        rep.changes.append(f"{g['gloss_id']}: reviewed")

    # Task 2: phrase gloss sequences (matched against the corrected glosses).
    sign_by_key = {match_key(g["gloss"]): g["gloss_id"] for g in glosses if g["unit"] == "sign"}
    next_id = max(int(g["gloss_id"][1:]) for g in glosses) + 1
    done = {p["gloss_id"] for p in phrases}
    for i, r in enumerate(read_kit_csv(kit / "anotacao_frases.csv"), start=2):
        gid, text = r["gloss_id"], " ".join(r["sequencia_glosas_libras"].upper().split())
        if not text:
            continue
        if gid not in by_id or by_id[gid]["unit"] != "phrase":
            rep.errors.append(f"anotacao_frases:{i}: {gid} is not a phrase (after task 1)")
            continue
        if gid in done:
            rep.errors.append(f"anotacao_frases:{i}: {gid} already annotated; edit "
                              "phrase_glosses.csv by hand to replace it")
            continue
        ids, bad, complete = [], [], "yes"
        for tok in text.split():
            m = TOKEN_RE.match(tok)
            if not m:
                bad.append(tok)
                continue
            base = m.group(1)
            if base == "?":
                complete = "no"
                continue
            key = match_key(base) if not base.startswith(("IX(", "CL(")) else base
            if key not in sign_by_key:
                new = {"gloss_id": f"G{next_id:05d}", "gloss": base, "unit": "sign",
                       "notes": f"from phrase annotation ({kit.name}); {tag}"}
                glosses.append(new)
                by_id[new["gloss_id"]] = new
                sign_by_key[key] = new["gloss_id"]
                rep.new_glosses.append(f"{new['gloss_id']} {base} (first seen in {gid})")
                next_id += 1
            ids.append(sign_by_key[key])
        if bad:
            rep.errors.append(f"anotacao_frases:{i}: {gid}: tokens outside the conventions: {bad}")
            continue
        phrases.append({"gloss_id": gid, "sequence": " ".join(ids), "sequence_text": text,
                        "complete": complete, "annotator": annotator, "kit": kit.name,
                        "notes": r["observacoes"]})
        done.add(gid)
        rep.changes.append(f"{gid}: sequence {text}")

    if apply and not rep.errors:
        write_table(curated_dir / "glosses.csv", TABLES["glosses"], glosses)
        write_table(curated_dir / "phrase_glosses.csv", TABLES["phrase_glosses"], phrases)
    return rep
