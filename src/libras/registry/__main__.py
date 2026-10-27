"""Registry CLI.

    python -m libras.registry propose-labels --sources minds eqt_uece
    python -m libras.registry build [--sources ...] [--no-probe]
    python -m libras.registry validate
    python -m libras.registry snapshot v0.1

--data-root defaults to $LIBRAS_DATA_ROOT.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

from libras.provenance import log_event
from libras.registry.build import ADAPTERS, build, propose_labels, scan
from libras.registry.store import load_registry, snapshot, validate


def main() -> int:
    p = argparse.ArgumentParser(prog="python -m libras.registry")
    p.add_argument("--data-root", default=os.environ.get("LIBRAS_DATA_ROOT"))
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate")
    snap = sub.add_parser("snapshot")
    snap.add_argument("version")
    for name in ("propose-labels", "build"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--sources", nargs="+", choices=sorted(ADAPTERS), default=sorted(ADAPTERS))
    sub.choices["build"].add_argument("--no-probe", action="store_true")
    args = p.parse_args()

    data_root = Path(args.data_root) if args.data_root else None
    if data_root is None and args.cmd != "validate":
        p.error(f"{args.cmd} needs --data-root or LIBRAS_DATA_ROOT")

    if args.cmd == "propose-labels":
        added = propose_labels(scan(data_root, args.sources))
        print(added)
        log_event(data_root, "registry.propose_labels", inputs=[f"raw ({s})" for s in args.sources],
                  outputs=["metadata/registry/glosses.csv", "metadata/registry/label_map.csv"],
                  result=added)
        return 0

    if args.cmd == "build":
        report = build(data_root, args.sources, probe=not args.no_probe)
        report = {k: dict(v) if isinstance(v, Counter) else v for k, v in report.items()}
        for k, v in report.items():
            print(f"{k}: {v}")
        log_event(data_root, "registry.build", inputs=[f"raw ({s})" for s in args.sources]
                  + ["metadata/registry/"],
                  outputs=["registry/current/samples.csv", "registry/current/signers.csv",
                           "registry/current/unmapped_labels.csv"], result=report)
        return 0

    reg = load_registry(data_root)
    counts = ", ".join(f"{k}={len(v)}" for k, v in reg.items())

    if args.cmd == "validate":
        errors = validate(reg)
        print(counts)
        for e in errors[:50]:
            print(e)
        if len(errors) > 50:
            print(f"... {len(errors) - 50} more")
        return 1 if errors else 0

    digest = snapshot(reg, data_root, args.version)
    print(f"snapshot {args.version}: {counts}; sha256={digest}")
    log_event(data_root, "registry.snapshot", inputs=["registry/current/", "metadata/registry/"],
              outputs=[f"registry/{args.version}/"], result={"sha256": digest, "counts": counts})
    return 0


if __name__ == "__main__":
    sys.exit(main())
