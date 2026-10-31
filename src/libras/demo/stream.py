"""Live recognition of isolated signs: streaming landmark extraction, sign segmentation by
arm activity, and classification with a trained run. No GUI here, so it is testable headless.

The extraction matches the training cache (mp_v2, ADR-0015) as closely as a stream allows:
pose on the full frame, hands and face on signing-space and head crops derived from the pose
of the last frames (the offline extractor uses one crop per sample).
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np
import torch
import yaml

from libras.data.features import n_features, prepare, to_features
from libras.models import build_model

SHOULDERS, WRISTS = (11, 12), (15, 16)


class StreamingExtractor:
    """One frame in, one frame of raw landmarks out (same layout as the .npz cache)."""

    def __init__(self, extraction_cfg: dict, model_paths: dict[str, Path],
                 roi_window: int = 45, roi_every: int = 10):
        import mediapipe as mp
        from mediapipe.tasks.python import vision

        from libras.extract.mediapipe_tasks import Extractor

        self.mp, self.cfg = mp, extraction_cfg
        options = Extractor(extraction_cfg, model_paths).options
        self.hand, self.pose, self.face = (cls.create_from_options(opt) for cls, opt in options)
        self.vision = vision
        self.history: deque[np.ndarray] = deque(maxlen=roi_window)
        self.roi_every, self.n = roi_every, 0
        self.rois = {"hand": (0.0, 0.0, 1.0, 1.0), "face": (0.0, 0.0, 1.0, 1.0)}

    def close(self) -> None:
        for task in (self.hand, self.pose, self.face):
            task.close()

    def _image(self, rgb: np.ndarray):
        return self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

    def process(self, rgb: np.ndarray, ts_ms: int) -> dict[str, np.ndarray]:
        from libras.extract.mediapipe_tasks import (
            Extractor,
            crop_pixels,
            head_roi,
            signing_roi,
            to_full,
        )

        rows: dict[str, list] = {k: [] for k in (
            "hands", "hands_world", "hand_score", "pose", "pose_world", "face")}
        Extractor._pose(self.pose.detect_for_video(self._image(rgb), ts_ms), rows)
        pose = rows["pose"][0]
        if not np.isnan(pose[0, 0]):
            self.history.append(pose)
        if self.history and self.n % self.roi_every == 0:
            past = np.asarray(self.history)
            crop = self.cfg["crop"]
            self.rois = {"hand": signing_roi(past, crop["signing_margin"]),
                         "face": head_roi(past, crop["head_scale"])}
        self.n += 1
        Extractor._hands(self.hand.detect_for_video(
            self._image(crop_pixels(rgb, self.rois["hand"])), ts_ms), rows)
        Extractor._face(self.face.detect_for_video(
            self._image(crop_pixels(rgb, self.rois["face"])), ts_ms), rows)
        h, w = rgb.shape[:2]
        hands, face = rows["hands"][0], rows["face"][0]
        to_full(hands, self.rois["hand"], w, h)
        to_full(face, self.rois["face"], w, h)
        return {"hands": hands, "pose": pose, "face": face}


def arm_raised(frame: dict[str, np.ndarray], reach: float = 1.2, min_vis: float = 0.5) -> bool:
    """A wrist is up in the signing space: above the shoulders' height plus ``reach`` shoulder
    widths. (The extraction metric uses the shoulder-hip midpoint; at a desk the hips are often
    out of the picture, so the demo measures from the shoulders.)"""
    pose = frame["pose"]
    if np.isnan(pose[0, 0]):
        return False
    sh = pose[list(SHOULDERS), :2]
    limit = sh[:, 1].mean() + reach * np.linalg.norm(sh[0] - sh[1])
    return any(pose[w, 3] > min_vis and pose[w, 1] < limit for w in WRISTS)


class Segmenter:
    """Cuts the stream into signs: a sign starts when an arm is raised for ``start`` frames and
    ends after ``end`` frames at rest. Each segment keeps ``pad`` frames of context on both
    sides, as the training clips start and end at rest."""

    def __init__(self, start: int = 3, end: int = 10, pad: int = 10, min_len: int = 12,
                 max_len: int = 240, reach: float = 1.2):
        self.start, self.end, self.pad = start, end, pad
        self.min_len, self.max_len, self.reach = min_len, max_len, reach
        self.before: deque[dict] = deque(maxlen=pad + start)
        self.current: list[dict] | None = None
        self.up = self.down = 0

    @property
    def active(self) -> bool:
        return self.current is not None

    def push(self, frame: dict) -> list[dict] | None:
        """Add a frame; returns a finished segment, if this frame ended one."""
        up = arm_raised(frame, self.reach)
        if self.current is None:
            self.before.append(frame)
            self.up = self.up + 1 if up else 0
            if self.up >= self.start:
                self.current, self.down = list(self.before), 0
            return None
        self.current.append(frame)
        self.down = 0 if up else self.down + 1
        if self.down >= self.end or len(self.current) >= self.max_len:
            return self.cut()
        return None

    def cut(self) -> list[dict] | None:
        """End the current segment now (also used for manual mode)."""
        seg, self.current, self.up = self.current, None, 0
        self.before.clear()
        if seg is None:
            return None
        trail = max(0, self.down - self.pad)  # drop rest frames beyond the padding
        seg = seg[:len(seg) - trail] if trail else seg
        return seg if len(seg) >= self.min_len else None


class Recognizer:
    """A trained run (LOSO fold or --final) as a classifier of landmark segments."""

    def __init__(self, run_dir: Path, device: str = "cpu", fold: int | None = None):
        self.cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        self.classes = json.loads((run_dir / "meta.json").read_text())["classes"]
        weights = run_dir / ("model.pt" if fold is None else f"fold_{fold:02d}/model.pt")
        fc = self.cfg["features"]
        self.model = build_model(self.cfg["model"], n_features(fc), len(self.classes))
        self.model.load_state_dict(torch.load(weights, map_location=device))
        self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def classify(self, frames: list[dict], frame_size: tuple[int, int], k: int = 5
                 ) -> list[tuple[str, float]]:
        hands = np.stack([f["hands"] for f in frames])
        pose = np.stack([f["pose"] for f in frames])
        face = np.stack([f["face"] for f in frames])
        lm = {"frame_size": np.asarray(frame_size), "hands": hands,
              "hand_present": ~np.isnan(hands[..., 0, 0]), "pose": pose,
              "pose_present": ~np.isnan(pose[:, 0, 0]), "face": face,
              "face_present": ~np.isnan(face[:, 0, 0])}
        x = to_features(prepare(lm, self.cfg["features"]), self.cfg["features"])
        logits = self.model(torch.from_numpy(x)[None].to(self.device))[0][0]
        prob = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        top = np.argsort(-prob)[:k]
        return [(self.classes[i], float(prob[i])) for i in top]
