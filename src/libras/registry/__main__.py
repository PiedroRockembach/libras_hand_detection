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
        print(propose_labels(scan(data_root, args.sources)))
        return 0

    if args.cmd == "build":
        report = build(data_root, args.sources, probe=not args.no_probe)
        for k, v in report.items():
            print(f"{k}: {dict(v) if isinstance(v, Counter) else v}")
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
