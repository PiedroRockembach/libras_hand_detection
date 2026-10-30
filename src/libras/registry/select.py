"""Sample selection shared by landmark extraction and training (registry + MALTA look-alikes)."""

from __future__ import annotations

from pathlib import Path

from libras.registry.adapters import malta
from libras.registry.adapters.base import match_key
from libras.registry.schema import TABLES
from libras.registry.store import CURATED_DIR, generated_dir, read_table


def landmark_path(root: Path, version: str, sample: dict) -> Path:
    """Cached landmarks of one sample: landmarks/<version>/<source_id>/<sample>.npz (ADR-0012)."""
    safe = sample["sample_id"].replace(":", "__").replace("/", "_")
    return root / "landmarks" / version / sample["source_id"] / f"{safe}.npz"


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
