import numpy as np

from libras.demo.stream import Segmenter, arm_raised


def frame(raised: bool) -> dict:
    pose = np.full((33, 4), 0.5, np.float32)
    pose[:, 3] = 1.0
    pose[11, :2], pose[12, :2] = (0.6, 0.4), (0.4, 0.4)  # shoulder width 0.2
    wrist_y = 0.45 if raised else 0.9  # rest: far below shoulders + 1.2 widths (0.64)
    pose[15, :2], pose[16, :2] = (0.65, wrist_y), (0.35, 0.9)
    return {"pose": pose, "hands": np.full((2, 21, 3), np.nan), "face": np.full((478, 3), np.nan)}


def test_arm_raised_uses_the_wrist_height_relative_to_the_shoulders():
    assert arm_raised(frame(True)) and not arm_raised(frame(False))
    no_pose = frame(True)
    no_pose["pose"][:] = np.nan
    assert not arm_raised(no_pose)


def test_segmenter_cuts_one_sign_with_context_on_both_sides():
    seg = Segmenter(start=3, end=5, pad=4, min_len=5)
    stream = [False] * 20 + [True] * 15 + [False] * 20
    out = [s for s in (seg.push(frame(r)) for r in stream) if s is not None]
    assert len(out) == 1
    raised = [arm_raised(f) for f in out[0]]
    assert sum(raised) == 15
    assert raised[:4] == [False] * 4 and raised[-4:] == [False] * 4  # padding kept, rest dropped


def test_short_blips_are_not_signs():
    seg = Segmenter(start=3, end=5, pad=4, min_len=12)
    stream = [False] * 10 + [True] * 3 + [False] * 10
    assert all(seg.push(frame(r)) is None for r in stream)
