"""From cached raw landmarks (ADR-0012) to model input, configured by the experiment YAML.

Two stages, so augmentation can act on geometry:

* ``prepare``: deterministic, once per sample. Selects points, fixes the aspect ratio,
  reassigns hands to the nearest pose wrist, fills short gaps and normalizes by the shoulders.
  Output: ``points`` (T, K, 2) in shoulder-width units (NaN where missing) and ``hand_valid``
  (T, 2).
* ``to_features``: per draw (after augmentation). Resamples to a fixed length and builds the
  per-frame feature vector: body-normalized coordinates, hand-local coordinates, velocity and
  hand presence. Missing values become 0.

Point layout of ``points``: hand slot 0 (21), hand slot 1 (21), then the configured pose points,
then the configured face points. Slot 0 is the hand nearest pose landmark 15, slot 1 nearest 16.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np

HAND = 21
POSE_WRIST = (15, 16)  # pose landmark of the wrist matching hand slot 0 / 1
SHOULDERS = (11, 12)
HAND_SCALE = (0, 9)  # wrist -> middle-finger MCP


@dataclass
class Prepared:
    points: np.ndarray  # (T, K, 2) float32, NaN where missing
    hand_valid: np.ndarray  # (T, 2) bool


def n_points(cfg: dict) -> int:
    return 2 * HAND + len(cfg["pose_points"]) + len(cfg["face_points"])


def assign_hands(hands: np.ndarray, present: np.ndarray, pose: np.ndarray,
                 pose_present: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Put each detected hand in the slot of the nearest pose wrist (MediaPipe's handedness
    label is unreliable, ADR-0015). Frames without pose keep MediaPipe's slots."""
    hands, present = hands.copy(), present.copy()
    for t in np.flatnonzero(pose_present & present.any(axis=1)):
        wrists = pose[t, list(POSE_WRIST), :2]
        if present[t].all():
            keep = sum(np.linalg.norm(hands[t, s, 0] - wrists[s]) for s in (0, 1))
            swap = sum(np.linalg.norm(hands[t, 1 - s, 0] - wrists[s]) for s in (0, 1))
            if swap < keep:
                hands[t] = hands[t, ::-1]
        else:
            s = int(np.flatnonzero(present[t])[0])
            near = int(np.argmin(np.linalg.norm(wrists - hands[t, s, 0], axis=1)))
            if near != s:
                hands[t, near], present[t, near] = hands[t, s], True
                hands[t, s], present[t, s] = np.nan, False
    return hands, present


def fill_gaps(x: np.ndarray, valid: np.ndarray, max_gap: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation over runs of invalid frames with valid frames on both sides and at
    most ``max_gap`` frames long (None: any length). Returns the filled array and validity."""
    x, valid = x.copy(), valid.copy()
    idx = np.flatnonzero(valid)
    for a, b in pairwise(idx):
        gap = b - a - 1
        if gap == 0 or (max_gap is not None and gap > max_gap):
            continue
        w = (np.arange(1, gap + 1) / (gap + 1)).reshape(-1, *([1] * (x.ndim - 1)))
        x[a + 1:b] = x[a] * (1 - w) + x[b] * w
        valid[a + 1:b] = True
    return x, valid


def prepare(lm: dict, cfg: dict) -> Prepared:
    """Raw landmarks of one sample (the .npz contents) -> normalized points."""
    w, h = (float(v) for v in lm["frame_size"])
    aspect = np.array([w / h, 1.0], dtype=np.float32)
    pose_present = lm["pose_present"].astype(bool)
    hands, hand_present = assign_hands(lm["hands"][..., :2], lm["hand_present"].astype(bool),
                                       lm["pose"][..., :2], pose_present)
    hands = hands * aspect
    pose = lm["pose"][..., :2] * aspect
    face = lm["face"][..., :2] * aspect
    face_present = lm["face_present"].astype(bool)

    slots = []
    for s in (0, 1):
        x, v = fill_gaps(hands[:, s], hand_present[:, s], cfg["max_hand_gap"])
        x[~v] = np.nan
        slots.append((x, v))
    pose, pose_present = fill_gaps(pose, pose_present, None)
    pose[~pose_present] = np.nan
    face, face_present = fill_gaps(face, face_present, None)
    face[~face_present] = np.nan

    # Body frame: origin at the shoulder midpoint, unit = shoulder width (medians over frames).
    sh = pose[:, list(SHOULDERS)]
    ok = ~np.isnan(sh).any(axis=(1, 2))
    if ok.any():
        center = np.median(sh[ok].mean(axis=1), axis=0)
        scale = float(np.median(np.linalg.norm(sh[ok, 0] - sh[ok, 1], axis=1)))
    else:  # no shoulders at all: fall back to the frame
        center, scale = np.array([w / h / 2, 0.5]), 0.25
    scale = scale if scale > 1e-6 else 0.25

    points = np.concatenate([slots[0][0], slots[1][0], pose[:, cfg["pose_points"]],
                             face[:, cfg["face_points"]]], axis=1)
    points = ((points - center) / scale).astype(np.float32)
    valid = np.stack([slots[0][1], slots[1][1]], axis=1)
    return Prepared(points, valid)


def load(path: Path, cfg: dict) -> Prepared:
    with np.load(path, allow_pickle=False) as z:
        return prepare({k: z[k] for k in ("frame_size", "hands", "hand_present", "pose",
                                          "pose_present", "face", "face_present")}, cfg)


def resample(x: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Sample ``x`` (T, ...) at fractional frame ``positions``: linear between neighbours,
    nearest neighbour when one of them is NaN."""
    lo = np.floor(positions).astype(int)
    hi = np.minimum(lo + 1, len(x) - 1)
    w = (positions - lo).reshape(-1, *([1] * (x.ndim - 1)))
    out = x[lo] * (1 - w) + x[hi] * w
    nearest = x[np.where(w.reshape(-1) < 0.5, lo, hi)]
    return np.where(np.isnan(out), nearest, out)


def to_features(seq: Prepared, cfg: dict, positions: np.ndarray | None = None) -> np.ndarray:
    """Prepared sequence -> (frames, F) float32 model input."""
    t = len(seq.points)
    if positions is None:
        positions = np.linspace(0, t - 1, cfg["frames"])
    pts = resample(seq.points, positions)
    valid = resample(seq.hand_valid.astype(np.float32), positions) >= 0.5
    pts[:, :2 * HAND][~np.repeat(valid, HAND, axis=1)] = np.nan

    parts = [pts.reshape(len(pts), -1)]
    if cfg.get("hand_local", True):
        for s in (0, 1):
            hand = pts[:, s * HAND:(s + 1) * HAND]
            size = np.linalg.norm(hand[:, HAND_SCALE[1]] - hand[:, HAND_SCALE[0]], axis=1)
            size = np.nanmedian(size) if np.isfinite(size).any() else 1.0
            parts.append(((hand - hand[:, :1]) / max(size, 1e-6)).reshape(len(pts), -1))
    if cfg.get("velocity", True):
        parts.append(np.diff(parts[0], axis=0, prepend=parts[0][:1]))
    parts.append(valid.astype(np.float32))
    return np.nan_to_num(np.concatenate(parts, axis=1), nan=0.0).astype(np.float32)


def n_features(cfg: dict) -> int:
    k = n_points(cfg)
    f = 2 * k
    if cfg.get("hand_local", True):
        f += 2 * 2 * HAND
    if cfg.get("velocity", True):
        f += 2 * k
    return f + 2
