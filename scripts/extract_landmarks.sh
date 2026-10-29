#!/usr/bin/env bash
# Full landmark extraction with the default configuration (ADR-0015), inside the official
# image. Runs after scripts/prepare_data.sh. Idempotent: samples already in the cache are
# skipped; every run is logged to $LIBRAS_DATA_ROOT/PROVENANCE.jsonl.
#
# Usage: LIBRAS_DATA_ROOT=/mnt/hd1tb/libras-data scripts/extract_landmarks.sh [WORKERS]
set -euo pipefail

: "${LIBRAS_DATA_ROOT:?set LIBRAS_DATA_ROOT}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
WORKERS="${1:-8}"
CONFIG=configs/extraction/mp_v2.yaml

# Progress bars for the task-progress mod: the container writes into the host's folder.
PROGRESS_DIR="${CLAUDE_PROGRESS_DIR:-$HOME/.cache/claude-progress}"
mkdir -p "$PROGRESS_DIR"

run() {
    docker run --rm -v "$REPO":/workspace -v "$LIBRAS_DATA_ROOT":/data \
        -v "$PROGRESS_DIR":/progress -e CLAUDE_PROGRESS_DIR=/progress libras:dev \
        python -m libras.extract "$@" --config "$CONFIG" 2>&1 \
        | grep -v '^W0000\|^I0000\|XNNPACK\|InitGoogle\|inference_feedback\|landmark_projection'
}

run models
# Every registry sample of MINDS (Canon and Kinect) and EQT-UECE, except label conflicts.
run run --sources minds minds_kinect eqt_uece --workers "$WORKERS"
# MALTA clips sharing a gloss with MINDS or EQT-UECE (out-of-domain test, ADR-0014).
run run --sources minds --malta-like minds --workers "$WORKERS"
run run --sources eqt_uece --malta-like eqt_uece --workers "$WORKERS"
run report
