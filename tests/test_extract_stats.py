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
