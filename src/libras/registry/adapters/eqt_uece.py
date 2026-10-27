"""LIBRAS-EQT-UECE v4 adapter (Zenodo 21398895), RGB videos only.

Layout: raw/libras_eqt_uece_v4/extracted/Libras-EQT-UECE (RGB)/<id>_<label>/INFORMANTE <n>/<file>.mp4
Labels mix single signs and whole sentences; the curated glosses table marks which is which.
"""

from __future__ import annotations

import re
from pathlib import Path

from libras.registry.adapters.base import ScanResult, clean_label, rel

CLASS_RE = re.compile(r"^(\d+)_(.+)$", re.DOTALL)
INFORMANT_RE = re.compile(r"^INFORMANTE\s*(\d+)\s*$")
COMMON = {"license": "CC-BY-NC-4.0", "consent_scope": "public_license", "media": "video"}


def scan(data_root: Path) -> ScanResult:
    res = ScanResult(source_ids=("eqt_uece",))
    root = data_root / "raw" / "libras_eqt_uece_v4" / "extracted" / "Libras-EQT-UECE (RGB)"
    informants: set[str] = set()
    for path in sorted(root.glob("*/*/*.mp4")):
        class_dir, informant_dir = path.parent.parent.name, path.parent.name
        cm, im = CLASS_RE.match(class_dir), INFORMANT_RE.match(informant_dir)
        if not cm or not im:
            raise ValueError(f"unexpected EQT-UECE path: {path}")
        class_id, informant = cm.group(1), im.group(1)  # "INFORMANTE 5 " == "INFORMANTE 5"
        informants.add(informant)
        res.samples.append({
            "sample_id": f"eqt_uece:{class_id}:{informant}:{path.stem}",
            "source_id": "eqt_uece", "source_label": clean_label(cm.group(2)),
            "signer_uid": f"eqt_uece:{informant}", "path": rel(path, data_root), **COMMON,
        })
    res.signers = [{"signer_uid": f"eqt_uece:{i}", "source_id": "eqt_uece",
                    "source_signer_id": f"INFORMANTE {i}"} for i in sorted(informants, key=int)]
    return res
