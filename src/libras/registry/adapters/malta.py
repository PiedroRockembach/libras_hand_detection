"""MALTA-LIBRAS adapter (Hugging Face MALTA-Lab/MALTA_LIBRAS), `malta_libras_complete` only.

Each .pt file is a torch uint8 tensor of 32 x 3 x 224 x 224 frames (no original video, no
signer ids). Every sub-collection becomes one "unknown signer", so a whole sub-collection
always lands on the same side of a split (conservative w.r.t. ADR-0004).
`malta_libras_MINDS_subset` is skipped: on the hub it only holds Git LFS pointer files.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

from libras.registry.adapters.base import ScanResult, clean_label, rel

ROOT = Path("raw") / "malta_libras" / "malta_libras_complete"
COMMON = {"license": "MIT (aggregate; upstream unknown)", "consent_scope": "unknown",
          "media": "frames_tensor", "n_frames": "32", "width": "224", "height": "224"}

# Filename decorations per sub-collection, removed to get the label.
YOUTUBE_RE = re.compile(r"^Libras\s*-\s*(.+?)\s*\(\d+p_.*\)$")
SUFFIX_RE = re.compile(r"(Sm_Prog\d+|_?\d+)$")
NUMBER_PREFIX_RE = re.compile(r"^\d+-")


def fix_mojibake(text: str) -> str:
    """Some upstream names are UTF-8 bytes decoded as CP437 ("Divers├úo" -> "Diversão")."""
    try:
        return text.encode("cp437").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def label_from_stem(collection: str, stem: str) -> str:
    stem = fix_mojibake(unquote(stem))  # some names are URL-encoded ("B%C3%81SICO")
    if collection.startswith("youtube"):
        m = YOUTUBE_RE.match(stem)
        stem = m.group(1) if m else stem
    else:
        stem = NUMBER_PREFIX_RE.sub("", SUFFIX_RE.sub("", stem))
    return clean_label(stem.replace("_", " "))


def scan(data_root: Path) -> ScanResult:
    res = ScanResult(source_ids=("malta",))
    root = data_root / ROOT
    for coll_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        coll = coll_dir.name
        res.signers.append({"signer_uid": f"malta:{coll}:unknown", "source_id": "malta",
                            "source_signer_id": f"{coll} (unknown)"})
        for path in sorted(coll_dir.rglob("*.pt")):
            res.samples.append({
                "sample_id": f"malta:{coll}:{path.relative_to(coll_dir).with_suffix('')}",
                "source_id": "malta", "source_label": label_from_stem(coll, path.stem),
                "signer_uid": f"malta:{coll}:unknown", "path": rel(path, data_root),
                "quality_flags": "upstream_signer_unknown", **COMMON,
            })
    return res
