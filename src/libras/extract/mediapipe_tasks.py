"""Raw landmark extraction with MediaPipe Tasks (ADR-0012).

One .npz per sample with every point the models return, at the source frame rate, in
normalized image coordinates (plus world coordinates for hands and pose). Missing
detections are NaN, with explicit presence masks. Nothing is selected or normalized here:
that happens at load time.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
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

    def run(self, frames: Iterator[np.ndarray], fps: float) -> dict[str, np.ndarray]:
        """Extract every frame. Landmarkers are created per sample: VIDEO mode needs
        increasing timestamps, and tracking must not leak from one sample to the next."""
        rows: dict[str, list] = {k: [] for k in (
            "hands", "hands_world", "hand_score", "pose", "pose_world", "face")}
        fps = fps or self.cfg["default_fps"]
        height = width = 0
        hand, pose, face = (cls.create_from_options(opt) for cls, opt in self.options)
        start = time.perf_counter()
        try:
            for i, rgb in enumerate(frames):
                height, width = rgb.shape[:2]
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
                ts = round(i * 1000.0 / fps)
                self._hands(hand.detect_for_video(image, ts), rows)
                self._pose(pose.detect_for_video(image, ts), rows)
                self._face(face.detect_for_video(image, ts), rows)
        finally:
            for task in (hand, pose, face):
                task.close()
        elapsed = time.perf_counter() - start
        if not rows["pose"]:
            raise ValueError("no frames decoded")
        out = {k: np.asarray(v, dtype=np.float32) for k, v in rows.items()}
        out["hand_present"] = ~np.isnan(out["hands"][..., 0, 0])
        out["pose_present"] = ~np.isnan(out["pose"][:, 0, 0])
        out["face_present"] = ~np.isnan(out["face"][:, 0, 0])
        out["fps"] = np.float32(fps)
        out["frame_size"] = np.array([width, height], dtype=np.int32)
        out["extract_seconds"] = np.float32(elapsed)
        return out

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


def detection_stats(lm: dict[str, np.ndarray], wrist_visibility: float = 0.5,
                    match_dist: float = 0.08) -> dict[str, float]:
    """Per-sample quality metrics for the pilot.

    `missed_hand_rate` is the key one: among frames where the pose model sees a wrist inside
    the image (visibility above threshold), how often no detected hand lies near it.
    """
    t = len(lm["pose"])
    hp = lm["hand_present"]
    missed = visible = 0
    for wrist in POSE_WRIST.values():
        w = lm["pose"][:, wrist]
        vis = (w[:, 3] > wrist_visibility) & (w[:, 0] >= 0) & (w[:, 0] <= 1) & \
              (w[:, 1] >= 0) & (w[:, 1] <= 1)
        # A hand "covers" the wrist if any detected hand's wrist point (index 0) is close.
        hand_wrists = lm["hands"][:, :, 0, :2]  # T x 2 x 2
        dist = np.linalg.norm(hand_wrists - w[:, None, :2], axis=-1)
        covered = np.nan_to_num(dist, nan=np.inf).min(axis=1) < match_dist
        visible += int(vis.sum())
        missed += int((vis & ~covered).sum())
    return {
        "n_frames": t,
        "any_hand_rate": float(hp.any(axis=1).mean()) if t else 0.0,
        "both_hands_rate": float(hp.all(axis=1).mean()) if t else 0.0,
        "pose_rate": float(lm["pose_present"].mean()) if t else 0.0,
        "face_rate": float(lm["face_present"].mean()) if t else 0.0,
        "missed_hand_rate": missed / visible if visible else float("nan"),
        "ms_per_frame": 1000.0 * float(lm["extract_seconds"]) / t if t else 0.0,
    }
