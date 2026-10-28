import numpy as np
import pytest

pytest.importorskip("mediapipe")
from libras.extract.mediapipe_tasks import N_FACE, N_HAND, N_POSE, detection_stats


def fake_landmarks(t: int = 4) -> dict:
    pose = np.zeros((t, N_POSE, 4), np.float32)
    pose[:, 15] = (0.3, 0.6, 0, 0.9)  # left wrist visible
    pose[:, 16] = (0.7, 0.6, 0, 0.1)  # right wrist not visible
    hands = np.full((t, 2, N_HAND, 3), np.nan, np.float32)
    hands[:2, 0, 0] = (0.31, 0.61, 0)  # left hand found near the wrist in frames 0-1 only
    return {
        "pose": pose, "hands": hands, "face": np.zeros((t, N_FACE, 3), np.float32),
        "hand_present": ~np.isnan(hands[..., 0, 0]),
        "pose_present": np.ones(t, bool), "face_present": np.array([1, 1, 0, 0], bool),
        "extract_seconds": np.float32(0.4),
    }


def test_missed_hand_rate_counts_only_visible_wrists():
    s = detection_stats(fake_landmarks())
    assert s["missed_hand_rate"] == pytest.approx(0.5)  # 2 of 4 visible-wrist frames missed
    assert s["any_hand_rate"] == pytest.approx(0.5)
    assert s["both_hands_rate"] == 0.0
    assert s["face_rate"] == pytest.approx(0.5)
    assert s["ms_per_frame"] == pytest.approx(100.0)
    # Wrist at y=0.6 is above the shoulder/hip midpoint (pose rows default to y=0 -> mid 0),
    # so it is "rest" here; flip the hips lower to make it "signing".
    lm = fake_landmarks()
    lm["pose"][:, [23, 24], 1] = 2.0  # midpoint = 1.0 -> wrist (0.6) counts as signing
    s = detection_stats(lm)
    assert s["missed_signing_rate"] == pytest.approx(0.5) and s["signing_wrist_frames"] == 4


def test_crop_mapping_round_trip():
    from libras.extract.mediapipe_tasks import crop_pixels, head_roi, signing_roi, to_full

    pose = np.zeros((3, N_POSE, 4), np.float32)
    pose[..., 3] = 1.0
    pose[:, :, 0], pose[:, :, 1] = 0.5, 0.5
    pose[1, 15, :2] = (0.4, 0.7)  # a wrist moves in one frame only
    roi = signing_roi(pose, 0.0)
    assert roi == pytest.approx((0.4, 0.5, 0.5, 0.7))  # union over all frames
    assert head_roi(pose, 1.0) == pytest.approx((0.5, 0.5, 0.5, 0.5))
    assert crop_pixels(np.zeros((100, 200, 3), np.uint8), (0.1, 0.2, 0.6, 0.7)).shape == (50, 100, 3)
    pts = np.array([[[0.5, 0.5, 0.1]]], np.float32)  # centre of the crop
    to_full(pts, (0.2, 0.4, 0.6, 0.8), 200, 100)
    assert pts[0, 0] == pytest.approx((0.4, 0.6, 0.04))
