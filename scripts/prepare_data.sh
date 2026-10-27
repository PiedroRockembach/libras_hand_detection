#!/usr/bin/env bash
# Reproduce the local data state from scratch: download -> extract -> registry build.
# Every step is idempotent (verified files are skipped) and appends to
# $LIBRAS_DATA_ROOT/PROVENANCE.jsonl. Curated tables (glosses, label_map) come from git.
#
# Usage: LIBRAS_DATA_ROOT=/mnt/hd1tb/libras-data scripts/prepare_data.sh
set -euo pipefail

: "${LIBRAS_DATA_ROOT:?set LIBRAS_DATA_ROOT}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$LIBRAS_DATA_ROOT/raw"
export PYTHONPATH="$REPO/src"

echo "== download"
python3 "$REPO/scripts/download_libras_datasets.py" --root "$LIBRAS_DATA_ROOT"

echo "== extract (CRC-verified; existing files are re-checked)"
extract() { python3 "$REPO/scripts/extract_zip_lenient.py" "$1" "$2" --exclude __MACOSX/ .DS_Store; }
for d in minds_libras_rgb minds_libras_rgbd; do
    for z in "$RAW/$d"/Sinalizador*.zip; do extract "$z" "$RAW/$d/extracted"; done
done
extract "$RAW/libras_eqt_uece_v4/Libras-EQT-UECE (RGB).zip" "$RAW/libras_eqt_uece_v4/extracted"

echo "== registry"
python3 -m libras.registry build --sources minds eqt_uece malta
python3 -m libras.registry validate
