"""Raw landmark extraction with MediaPipe Tasks (ADR-0012).

One .npz per sample with every point the models return, at the source frame rate, in
normalized image coordinates (plus world coordinates for hands and pose). Missing
detections are NaN, with explicit presence masks. Nothing is selected or normalized here:
that happens at load time.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

N_HAND, N_POSE, N_FACE = 21, 33, 478
HAND_SLOT = {"Left": 0, "Right": 1}  # MediaPipe handedness label -> slot
POSE_WRIST = {0: 15, 1: 16}  # slot -> pose landmark of the wrist with the same label


def read_video(path: Path) -> tuple[Iterator[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

    def frames() -> Iterator[np.ndarray]:
        try:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    return
                yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        finally:
            cap.release()

    return frames(), fps


def read_frames_tensor(path: Path) -> tuple[Iterator[np.ndarray], float]:
    """MALTA-LIBRAS clip: uint8 tensor T x 3 x H x W, no timing information."""
    import torch

    t = torch.load(path, map_location="cpu", weights_only=True)
    arr = np.ascontiguousarray(t.numpy().transpose(0, 2, 3, 1))
    return iter(arr), 0.0


class Extractor:
    def __init__(self, cfg: dict, model_paths: dict[str, Path]):
        self.cfg = cfg
        mode = vision.RunningMode.VIDEO

        def base(name: str) -> BaseOptions:
            return BaseOptions(model_asset_path=str(model_paths[name]))

        self.options = (
            (vision.HandLandmarker, vision.HandLandmarkerOptions(
                base_options=base("hand"), running_mode=mode, **cfg["hand"])),
            (vision.PoseLandmarker, vision.PoseLandmarkerOptions(
                base_options=base("pose"), running_mode=mode, num_poses=1, **cfg["pose"])),
            (vision.FaceLandmarker, vision.FaceLandmarkerOptions(
                base_options=base("face"), running_mode=mode, num_faces=1, **cfg["face"])),
        )

    def run(self, open_frames: Callable[[], Iterator[np.ndarray]], fps: float
            ) -> dict[str, np.ndarray]:
        """Extract every frame. Landmarkers are created per sample: VIDEO mode needs
        increasing timestamps, and tracking must not leak from one sample to the next.

        With `crop` in the config, a first pass runs pose on the full frame; hands and face
        then run on fixed per-sample crops (signing space, head) derived from it, and their
        coordinates are mapped back to the full frame.
        """
        rows: dict[str, list] = {k: [] for k in (
            "hands", "hands_world", "hand_score", "pose", "pose_world", "face")}
        fps = fps or self.cfg["default_fps"]
        crop = self.cfg.get("crop")
        size = [0, 0]  # width, height
        start = time.perf_counter()
        if not crop:
            self._pass(open_frames(), fps, rows, size, ("hand", "pose", "face"))
        else:
            self._pass(open_frames(), fps, rows, size, ("pose",))
            if not rows["pose"]:
                raise ValueError("no frames decoded")
            pose_arr = np.asarray(rows["pose"], np.float32)
            rois = {"hand": signing_roi(pose_arr, crop["signing_margin"]),
                    "face": head_roi(pose_arr, crop["head_scale"])}
            self._pass(open_frames(), fps, rows, size, ("hand", "face"), rois)
        elapsed = time.perf_counter() - start
        if not rows["pose"]:
            raise ValueError("no frames decoded")
        out = {k: np.asarray(v, dtype=np.float32) for k, v in rows.items()}
        width, height = size
        if crop:
            to_full(out["hands"], rois["hand"], width, height)
            to_full(out["face"], rois["face"], width, height)
            out["hand_roi"] = np.asarray(rois["hand"], np.float32)
            out["face_roi"] = np.asarray(rois["face"], np.float32)
        out["hand_present"] = ~np.isnan(out["hands"][..., 0, 0])
        out["pose_present"] = ~np.isnan(out["pose"][:, 0, 0])
        out["face_present"] = ~np.isnan(out["face"][:, 0, 0])
        out["fps"] = np.float32(fps)
        out["frame_size"] = np.array([width, height], dtype=np.int32)
        out["extract_seconds"] = np.float32(elapsed)
        return out

    def _pass(self, frames: Iterator[np.ndarray], fps: float, rows: dict, size: list,
              tasks: tuple[str, ...], rois: dict | None = None) -> None:
        """Run the named tasks over all frames; with rois, each task sees its own crop."""
        options = dict(zip(("hand", "pose", "face"), self.options, strict=True))
        running = {t: options[t][0].create_from_options(options[t][1]) for t in tasks}
        handlers = {"hand": self._hands, "pose": self._pose, "face": self._face}
        try:
            for i, rgb in enumerate(frames):
                size[1], size[0] = rgb.shape[:2]
                ts = round(i * 1000.0 / fps)
                for t, task in running.items():
                    img = rgb if not rois else crop_pixels(rgb, rois[t])
                    image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                     data=np.ascontiguousarray(img))
                    handlers[t](task.detect_for_video(image, ts), rows)
        finally:
            for task in running.values():
                task.close()

    @staticmethod
    def _hands(res, rows: dict) -> None:
        pts = np.full((2, N_HAND, 3), np.nan, np.float32)
        world = np.full((2, N_HAND, 3), np.nan, np.float32)
        score = np.zeros(2, np.float32)
        dets = sorted(zip(res.hand_landmarks, res.hand_world_landmarks, res.handedness),
                      key=lambda d: -d[2][0].score)
        free = [0, 1]
        for lms, wlms, handed in dets:
            slot = HAND_SLOT.get(handed[0].category_name, free[0])
            if slot not in free:  # both detections claim the same label: keep the stronger one
                slot = free[0]
            free.remove(slot)
            pts[slot] = [(p.x, p.y, p.z) for p in lms]
            world[slot] = [(p.x, p.y, p.z) for p in wlms]
            score[slot] = handed[0].score
        rows["hands"].append(pts)
        rows["hands_world"].append(world)
        rows["hand_score"].append(score)

    @staticmethod
    def _pose(res, rows: dict) -> None:
        pts = np.full((N_POSE, 4), np.nan, np.float32)
        world = np.full((N_POSE, 3), np.nan, np.float32)
        if res.pose_landmarks:
            pts[:] = [(p.x, p.y, p.z, p.visibility) for p in res.pose_landmarks[0]]
            world[:] = [(p.x, p.y, p.z) for p in res.pose_world_landmarks[0]]
        rows["pose"].append(pts)
        rows["pose_world"].append(world)

    @staticmethod
    def _face(res, rows: dict) -> None:
        pts = np.full((N_FACE, 3), np.nan, np.float32)
        if res.face_landmarks:
            pts[:] = [(p.x, p.y, p.z) for p in res.face_landmarks[0]]
        rows["face"].append(pts)


# Pose landmarks spanning the signing space (face, shoulders, elbows, wrists, hand points).
SIGNING_POINTS = list(range(23))
HEAD_POINTS = list(range(11))
MIN_VIS = 0.3


def _box(pts: np.ndarray) -> tuple[float, float, float, float] | None:
    ok = pts[..., 3] > MIN_VIS
    if not ok.any():
        return None
    xy = pts[..., :2][ok]
    return float(xy[:, 0].min()), float(xy[:, 1].min()), float(xy[:, 0].max()), float(xy[:, 1].max())


def _expand(box, fx: float, fy: float) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    return (max(0.0, x0 - fx * w), max(0.0, y0 - fy * h), min(1.0, x1 + fx * w),
            min(1.0, y1 + fy * h))


def signing_roi(pose: np.ndarray, margin: float) -> tuple[float, float, float, float]:
    """Union over the sample of the signing-space points, plus a margin on each side."""
    box = _box(pose[:, SIGNING_POINTS])
    return _expand(box, margin, margin) if box else (0.0, 0.0, 1.0, 1.0)


def head_roi(pose: np.ndarray, scale: float) -> tuple[float, float, float, float]:
    """Union over the sample of the head points, enlarged so the face fills the crop."""
    box = _box(pose[:, HEAD_POINTS])
    return _expand(box, scale, scale) if box else (0.0, 0.0, 1.0, 1.0)


def crop_pixels(rgb: np.ndarray, roi) -> np.ndarray:
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = roi
    return rgb[int(y0 * h):max(int(y1 * h), int(y0 * h) + 1),
               int(x0 * w):max(int(x1 * w), int(x0 * w) + 1)]


def to_full(points: np.ndarray, roi, width: int, height: int) -> None:
    """In place: crop-normalized (x, y, z) -> full-frame normalized. z follows x's scale."""
    x0, y0, x1, y1 = roi
    cw, ch = (x1 - x0), (y1 - y0)
    points[..., 0] = points[..., 0] * cw + x0
    points[..., 1] = points[..., 1] * ch + y0
    points[..., 2] = points[..., 2] * cw


def detection_stats(lm: dict[str, np.ndarray], wrist_visibility: float = 0.5,
                    match_dist: float = 0.08) -> dict[str, float]:
    """Per-sample quality metrics for the pilot.

    `missed_hand_rate` is the key one: among frames where the pose model sees a wrist inside
    the image (visibility above threshold), how often no detected hand lies near it.
    """
    t = len(lm["pose"])
    hp = lm["hand_present"]
    pose = lm["pose"]
    with warnings.catch_warnings():  # frames without pose give NaN and count as resting
        warnings.simplefilter("ignore", RuntimeWarning)
        mid_y = (np.nanmean(pose[:, [11, 12], 1], axis=1) + np.nanmean(pose[:, [23, 24], 1], axis=1)) / 2
    counts = {"all": [0, 0], "signing": [0, 0], "rest": [0, 0]}  # [missed, visible]
    for wrist in POSE_WRIST.values():
        w = pose[:, wrist]
        vis = (w[:, 3] > wrist_visibility) & (w[:, 0] >= 0) & (w[:, 0] <= 1) & \
              (w[:, 1] >= 0) & (w[:, 1] <= 1)
        # A hand "covers" the wrist if any detected hand's wrist point (index 0) is close.
        hand_wrists = lm["hands"][:, :, 0, :2]  # T x 2 x 2
        dist = np.linalg.norm(hand_wrists - w[:, None, :2], axis=-1)
        covered = np.nan_to_num(dist, nan=np.inf).min(axis=1) < match_dist
        # Signing: wrist above the midpoint between shoulders and hips (arm not hanging).
        signing = w[:, 1] < mid_y
        for name, sel in (("all", vis), ("signing", vis & signing), ("rest", vis & ~signing)):
            counts[name][0] += int((sel & ~covered).sum())
            counts[name][1] += int(sel.sum())

    def rate(name: str) -> float:
        missed, visible = counts[name]
        return missed / visible if visible else float("nan")

    return {
        "n_frames": t,
        "any_hand_rate": float(hp.any(axis=1).mean()) if t else 0.0,
        "both_hands_rate": float(hp.all(axis=1).mean()) if t else 0.0,
        "pose_rate": float(lm["pose_present"].mean()) if t else 0.0,
        "face_rate": float(lm["face_present"].mean()) if t else 0.0,
        "missed_hand_rate": rate("all"),
        "missed_signing_rate": rate("signing"),
        "missed_rest_rate": rate("rest"),
        "signing_wrist_frames": counts["signing"][1],
        "rest_wrist_frames": counts["rest"][1],
        "ms_per_frame": 1000.0 * float(lm["extract_seconds"]) / t if t else 0.0,
    }
