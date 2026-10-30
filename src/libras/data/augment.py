"""Training-time augmentation on prepared sequences (ADR-0014 §3). All draws come from the
generator passed in, so a run is repeatable from its seed."""

from __future__ import annotations

import numpy as np

from libras.data.features import HAND, Prepared


def mirror_permutation(mean_points: np.ndarray) -> np.ndarray:
    """Index map for a left-right flip: hand slots swap; every body point goes to the point
    nearest its mirror image in the mean pose of the training set (pose and face pairs)."""
    k = len(mean_points)
    perm = np.arange(k)
    perm[:HAND], perm[HAND:2 * HAND] = np.arange(HAND, 2 * HAND), np.arange(HAND)
    body = mean_points[2 * HAND:]
    flipped = body * np.array([-1.0, 1.0])
    dist = np.linalg.norm(flipped[:, None] - body[None], axis=2)
    perm[2 * HAND:] = 2 * HAND + np.nanargmin(np.where(np.isnan(dist), np.inf, dist), axis=1)
    return perm


def time_positions(t: int, frames: int, cfg: dict, rng: np.random.Generator) -> np.ndarray:
    """Random crop of up to ``time_crop`` at each end plus a smooth monotonic warp."""
    crop = cfg.get("time_crop", 0.0)
    start = rng.uniform(0, crop) * (t - 1)
    end = (t - 1) * (1 - rng.uniform(0, crop))
    u = np.linspace(0, 1, frames)
    warp = cfg.get("time_warp", 0.0)
    if warp:
        a = rng.uniform(-warp, warp)
        u = u + a * u * (1 - u)  # stays monotonic for |a| < 1
    return start + u * (end - start)


def augment(seq: Prepared, cfg: dict, rng: np.random.Generator,
            mirror_perm: np.ndarray | None) -> Prepared:
    pts, valid = seq.points.copy(), seq.hand_valid.copy()
    t = len(pts)

    if mirror_perm is not None and rng.random() < cfg.get("mirror", 0.0):
        pts = pts[:, mirror_perm] * np.array([-1.0, 1.0], dtype=np.float32)
        valid = valid[:, ::-1]

    s = 1 + rng.uniform(-cfg.get("scale", 0.0), cfg.get("scale", 0.0))
    a = np.deg2rad(rng.uniform(-cfg.get("rotate_deg", 0.0), cfg.get("rotate_deg", 0.0)))
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], dtype=np.float32)
    pts = (pts @ rot.T) * s
    if cfg.get("noise", 0.0):
        pts = pts + rng.normal(0, cfg["noise"], pts.shape).astype(np.float32)

    # Simulated detection failures: one hand lost over a span, and isolated dropped frames.
    if rng.random() < cfg.get("hand_drop", 0.0):
        slot = int(rng.integers(2))
        span = max(1, int(t * rng.uniform(0.1, 0.3)))
        start = int(rng.integers(0, max(1, t - span)))
        valid[start:start + span, slot] = False
        pts[start:start + span, slot * HAND:(slot + 1) * HAND] = np.nan
    if cfg.get("frame_drop", 0.0):
        drop = rng.random(t) < cfg["frame_drop"]
        pts[drop] = np.nan
        valid[drop] = False
    return Prepared(pts.astype(np.float32), valid)
