"""MINDS-Libras adapter: Canon RGB (Zenodo 2667329) and Kinect v2 RGB (Zenodo 4322983).

Both records show the same 12 signers, so Kinect signers are linked to the Canon ones via
same_person_as and share a split group. Expects the zips extracted into
raw/minds_libras_rgb/extracted/ and raw/minds_libras_rgbd/extracted/.
"""

from __future__ import annotations

import re
from pathlib import Path

from libras.registry.adapters.base import ScanResult, rel

CANON_RE = re.compile(r"^(\d\d)([^\W\d_]+)Sinalizador(\d\d)-(\d+)\.mp4$")
KINECT_RE = re.compile(r"^(\d+)-(\d\d)([^\W\d_]+)_(\d+)RGB\.mp4$")
N_SIGNERS = 12

# Recording conditions as described in docs/datasets/minds-libras.md.
COMMON = {"background": "chroma_key", "lighting": "studio", "license": "CC-BY-4.0",
          "consent_scope": "public_license", "media": "video"}


def scan(data_root: Path) -> ScanResult:
    res = ScanResult(source_ids=("minds", "minds_kinect"))
    for s in range(1, N_SIGNERS + 1):
        res.signers.append({"signer_uid": f"minds:{s:02d}", "source_id": "minds",
                            "source_signer_id": f"Sinalizador{s:02d}"})
        res.signers.append({"signer_uid": f"minds_kinect:{s:02d}", "source_id": "minds_kinect",
                            "source_signer_id": f"Sinalizador{s:02d}",
                            "same_person_as": f"minds:{s:02d}"})

    canon_root = data_root / "raw" / "minds_libras_rgb" / "extracted"
    for path in sorted(canon_root.glob("Sinalizador*/Canon/*.mp4")):
        m = CANON_RE.match(path.name)
        if not m:
            raise ValueError(f"unexpected MINDS file name: {path}")
        _, label, signer, rep = m.groups()
        res.samples.append({
            "sample_id": f"minds:{signer}:{label.lower()}:{rep}", "source_id": "minds",
            "source_label": label, "signer_uid": f"minds:{signer}",
            "path": rel(path, data_root), "camera": "canon", **COMMON,
        })

    kinect_root = data_root / "raw" / "minds_libras_rgbd" / "extracted"
    for path in sorted(kinect_root.glob("Sinalizador*/*/*RGB.mp4")):
        m = KINECT_RE.match(path.name)
        if not m:
            raise ValueError(f"unexpected MINDS Kinect file name: {path}")
        signer, _, label, rep = m.groups()
        signer = f"{int(signer):02d}"
        res.samples.append({
            "sample_id": f"minds_kinect:{signer}:{label.lower()}:{rep}",
            "source_id": "minds_kinect", "source_label": label,
            "signer_uid": f"minds_kinect:{signer}", "path": rel(path, data_root),
            "camera": "kinect_v2", **COMMON,
        })
    return res
