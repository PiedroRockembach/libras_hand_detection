"""Read, write, validate and snapshot the registry tables (ADR-0010, ADR-0011)."""

from __future__ import annotations

import csv
import hashlib
import shutil
from collections import Counter
from pathlib import Path

from libras.registry.schema import CURATED, GENERATED, TABLES, Table

Row = dict[str, str]
Registry = dict[str, list[Row]]

REPO_ROOT = Path(__file__).resolve().parents[3]
CURATED_DIR = REPO_ROOT / "metadata" / "registry"


def generated_dir(data_root: Path) -> Path:
    return Path(data_root) / "registry" / "current"


def read_table(path: Path, table: Table) -> list[Row]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        unknown = set(reader.fieldnames or []) - set(table.column_names)
        if unknown:
            raise ValueError(f"{path}: unknown columns {sorted(unknown)}")
        return [{c: (row.get(c) or "").strip() for c in table.column_names} for row in reader]


def write_table(path: Path, table: Table, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: tuple(r.get(k, "") for k in table.key))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=table.column_names, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in table.column_names})


def load_registry(data_root: Path | None, curated_dir: Path = CURATED_DIR) -> Registry:
    reg = {name: read_table(curated_dir / f"{name}.csv", TABLES[name]) for name in CURATED}
    for name in GENERATED:
        rows = read_table(generated_dir(data_root) / f"{name}.csv", TABLES[name]) if data_root else []
        reg[name] = rows
    return reg


def _check_kind(value: str, kind: str) -> bool:
    try:
        {"int": int, "float": float}.get(kind, str)(value)
    except ValueError:
        return False
    return True


def validate(reg: Registry) -> list[str]:
    """Return a list of human-readable errors; empty means valid."""
    errors: list[str] = []
    for name, table in TABLES.items():
        rows = reg.get(name, [])
        for i, row in enumerate(rows, start=2):  # CSV line numbers (header is line 1)
            for col in table.columns:
                v = row.get(col.name, "")
                if not v:
                    if col.required:
                        errors.append(f"{name}:{i}: missing {col.name}")
                    continue
                if col.choices and v not in col.choices:
                    errors.append(f"{name}:{i}: {col.name}={v!r} not in {col.choices}")
                if not _check_kind(v, col.kind):
                    errors.append(f"{name}:{i}: {col.name}={v!r} is not {col.kind}")
        keys = Counter(tuple(r.get(k, "") for k in table.key) for r in rows)
        errors += [f"{name}: duplicate key {k}" for k, n in keys.items() if n > 1]
        for fk in table.foreign_keys:
            targets = {r.get(fk.target, "") for r in reg.get(fk.table, [])}
            for i, row in enumerate(rows, start=2):
                v = row.get(fk.column, "")
                if v and v not in targets:
                    errors.append(f"{name}:{i}: {fk.column}={v!r} not found in {fk.table}")
    units = {g["gloss_id"]: g.get("unit") for g in reg.get("glosses", [])}
    for i, row in enumerate(reg.get("phrase_glosses", []), start=2):
        for gid in row.get("sequence", "").split():
            if units.get(gid) != "sign":
                errors.append(f"phrase_glosses:{i}: sequence item {gid!r} is not a sign gloss")
    return errors


def split_group(reg: Registry) -> dict[str, str]:
    """Map signer_uid -> canonical person id, following same_person_as chains (ADR-0004)."""
    link = {r["signer_uid"]: r.get("same_person_as", "") for r in reg.get("signers", [])}
    out = {}
    for uid in link:
        seen, cur = {uid}, uid
        while link.get(cur) and link[cur] not in seen:
            cur = link[cur]
            seen.add(cur)
        out[uid] = min(seen) if link.get(cur) in seen else cur  # cycles: stable representative
    return out


def snapshot(reg: Registry, data_root: Path, version: str) -> str:
    """Freeze all tables into registry/<version>/ and return the snapshot hash."""
    errors = validate(reg)
    if errors:
        raise ValueError(f"registry invalid ({len(errors)} errors), first: {errors[0]}")
    out = Path(data_root) / "registry" / version
    if out.exists():
        raise FileExistsError(f"snapshot {version} already exists: {out}")
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    lines = []
    for name, table in TABLES.items():
        path = tmp / f"{name}.csv"
        write_table(path, table, reg.get(name, []))
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {name}.csv")
    sums = "\n".join(lines) + "\n"
    (tmp / "SHA256SUMS").write_text(sums)
    tmp.rename(out)
    return hashlib.sha256(sums.encode()).hexdigest()
