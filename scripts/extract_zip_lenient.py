#!/usr/bin/env python3
"""Extract zips whose >4 GB offsets were written without proper ZIP64 records.

The MINDS-Libras RGB zips (macOS Archive Utility) store member offsets modulo 2**32, so
unzip, bsdtar and zipfile all reject or misread them. This tool takes each member from the
central directory, finds its real local header among offset + k * 2**32 candidates
(checking signature and file name), inflates it and verifies the CRC-32. Members are
written atomically, existing files are skipped, and paths outside dest are refused.

Usage:
    python3 scripts/extract_zip_lenient.py ARCHIVE.zip DEST_DIR [--exclude PREFIX ...]
"""

from __future__ import annotations

import argparse
import struct
import sys
import zipfile
import zlib
from pathlib import Path

LOCAL_SIG = b"PK\x03\x04"
LOCAL_HEADER = struct.Struct("<4s5H3I2H")  # sig, ver, flags, method, time, date, crc, csize, usize, n, m
CHUNK = 8 << 20
WRAP = 1 << 32


def find_data_offset(f, info: zipfile.ZipInfo, file_size: int) -> int:
    base = info.header_offset % WRAP
    name = info.orig_filename.encode("utf-8" if info.flag_bits & 0x800 else "cp437")
    for k in range(file_size // WRAP + 1):
        off = base + k * WRAP
        f.seek(off)
        head = f.read(LOCAL_HEADER.size)
        if len(head) < LOCAL_HEADER.size or head[:4] != LOCAL_SIG:
            continue
        *_, n, m = LOCAL_HEADER.unpack(head)
        if f.read(n) == name:
            return off + LOCAL_HEADER.size + n + m
    raise ValueError(f"local header not found for {info.filename}")


def extract_member(f, info: zipfile.ZipInfo, file_size: int, dest: Path) -> None:
    if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
        raise ValueError(f"{info.filename}: unsupported compression {info.compress_type}")
    f.seek(find_data_offset(f, info, file_size))
    inflater = zlib.decompressobj(-15) if info.compress_type == zipfile.ZIP_DEFLATED else None
    tmp = dest.with_name(dest.name + ".tmp")
    crc, remaining = 0, info.compress_size
    with tmp.open("wb") as out:
        while remaining:
            raw = f.read(min(CHUNK, remaining))
            if not raw:
                raise ValueError(f"{info.filename}: truncated data")
            remaining -= len(raw)
            data = inflater.decompress(raw) if inflater else raw
            crc = zlib.crc32(data, crc)
            out.write(data)
        if inflater:
            tail = inflater.flush()
            crc = zlib.crc32(tail, crc)
            out.write(tail)
    if crc != info.CRC:
        tmp.unlink()
        raise ValueError(f"{info.filename}: CRC mismatch")
    tmp.replace(dest)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("archive", type=Path)
    p.add_argument("dest", type=Path)
    p.add_argument("--exclude", nargs="*", default=["__MACOSX/"], help="name prefixes or suffixes to skip")
    args = p.parse_args()

    dest_root = args.dest.resolve()
    file_size = args.archive.stat().st_size
    done = skipped = 0
    errors = []
    with zipfile.ZipFile(args.archive) as zf, args.archive.open("rb") as f:
        for info in zf.infolist():
            name = info.filename
            if info.is_dir() or any(name.startswith(x) or name.endswith(x) for x in args.exclude):
                continue
            target = (dest_root / name).resolve()
            if not target.is_relative_to(dest_root):
                errors.append(f"{name}: path escapes destination")
                continue
            if target.exists() and target.stat().st_size == info.file_size:
                skipped += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                extract_member(f, info, file_size, target)
                done += 1
            except ValueError as e:
                errors.append(str(e))
    print(f"{args.archive.name}: extracted {done}, skipped {skipped}, errors {len(errors)}")
    for e in errors[:10]:
        print("  " + e)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
