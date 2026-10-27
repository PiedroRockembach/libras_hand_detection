"""Common pieces for source adapters (ADR-0010).

An adapter reads one downloaded dataset in its original layout and returns signer and
sample rows. Samples carry `source_label` (the dataset's own label) instead of `gloss_id`;
the build step resolves it through the curated label_map.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from libras.registry.store import Row


@dataclass
class ScanResult:
    source_ids: tuple[str, ...]
    signers: list[Row] = field(default_factory=list)
    samples: list[Row] = field(default_factory=list)


def rel(path: Path, data_root: Path) -> str:
    return path.relative_to(data_root).as_posix()


def clean_label(text: str) -> str:
    """Collapse whitespace/newlines and strip; keeps case and punctuation."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def gloss_text(label: str) -> str:
    """Canonical gloss spelling: upper case, no trailing punctuation or ellipsis."""
    return clean_label(label).rstrip(" .…").upper()


def match_key(text: str) -> str:
    """Accent- and punctuation-insensitive key used to merge glosses across sources."""
    decomposed = unicodedata.normalize("NFKD", gloss_text(text))
    ascii_text = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[^A-Z0-9]+", " ", ascii_text).strip()
