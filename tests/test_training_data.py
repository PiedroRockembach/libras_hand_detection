import numpy as np
import torch

from libras.data.augment import augment, mirror_permutation, time_positions
from libras.data.dataset import Item, SignDataset, loso_folds
from libras.data.features import assign_hands, fill_gaps, n_features, prepare, resample, to_features
from libras.models import build_model

FEAT = {"pose_points": [0, 11, 12, 13, 14, 15, 16], "face_points": [1, 61, 291],
        "frames": 16, "max_hand_gap": 3, "hand_local": True, "velocity": True}
AUG = {"scale": 0.1, "rotate_deg": 10, "noise": 0.01, "time_crop": 0.1, "time_warp": 0.3,
       "hand_drop": 0.5, "frame_drop": 0.05, "mirror": 0.5}


def fake_landmarks(t=30, seed=0):
    rng = np.random.default_rng(seed)
    pose = np.full((t, 33, 4), 0.5, np.float32)
    pose[:, 11, :2], pose[:, 12, :2] = (0.6, 0.4), (0.4, 0.4)  # shoulders (image coords)
    pose[:, 15, :2], pose[:, 16, :2] = (0.65, 0.3), (0.35, 0.3)  # wrists
    hands = rng.normal(0, 0.01, (t, 2, 21, 3)).astype(np.float32)
    hands[:, 0, :, :2] += (0.65, 0.3)
    hands[:, 1, :, :2] += (0.35, 0.3)
    return {"frame_size": np.array([1920, 1080]), "hands": hands,
            "hand_present": np.ones((t, 2), bool), "pose": pose, "pose_present": np.ones(t, bool),
            "face": rng.random((t, 478, 3)).astype(np.float32), "face_present": np.ones(t, bool)}


def test_hands_follow_the_nearest_pose_wrist():
    lm = fake_landmarks()
    swapped = lm["hands"][:, ::-1, :, :2].copy()
    hands, present = assign_hands(swapped, lm["hand_present"], lm["pose"][..., :2],
                                  lm["pose_present"])
    assert np.allclose(hands, lm["hands"][..., :2])
    # one hand detected, in the wrong slot: it moves to the slot of the nearer wrist
    one = np.full_like(swapped, np.nan)
    one[:, 1] = lm["hands"][:, 0, :, :2]
    hands, present = assign_hands(one, np.tile([False, True], (30, 1)), lm["pose"][..., :2],
                                  lm["pose_present"])
    assert present[:, 0].all() and not present[:, 1].any()


def test_short_gaps_are_interpolated_and_long_ones_kept():
    x = np.arange(10, dtype=np.float32)[:, None]
    valid = np.ones(10, bool)
    valid[2:4] = False  # gap of 2
    valid[5:9] = False  # gap of 4
    filled, ok = fill_gaps(x, valid, max_gap=3)
    assert np.allclose(filled[2:4, 0], [2, 3]) and ok[2:4].all()
    assert not ok[5:9].any()


def test_points_are_in_shoulder_units_centered_on_the_shoulders():
    seq = prepare(fake_landmarks(), FEAT)
    k = 42  # pose points start after the two hands
    left, right = seq.points[0, k + 1], seq.points[0, k + 2]  # pose 11 and 12
    assert np.allclose((left + right) / 2, 0, atol=1e-5)
    assert np.isclose(np.linalg.norm(left - right), 1, atol=1e-5)


def test_resample_keeps_endpoints_and_features_have_the_declared_size():
    x = np.arange(5, dtype=np.float32)[:, None, None]
    assert np.allclose(resample(x, np.linspace(0, 4, 9))[[0, -1], 0, 0], [0, 4])
    f = to_features(prepare(fake_landmarks(), FEAT), FEAT)
    assert f.shape == (16, n_features(FEAT)) and np.isfinite(f).all()


def test_mirror_swaps_hands_and_symmetric_points():
    seq = prepare(fake_landmarks(), FEAT)
    perm = mirror_permutation(np.nanmean(seq.points, axis=0))
    assert list(perm[:21]) == list(range(21, 42))
    k = 42
    assert perm[k + 1] == k + 2 and perm[k + 5] == k + 6  # shoulders and wrists swap


def test_augmentation_is_repeatable_from_the_seed():
    seq = prepare(fake_landmarks(), FEAT)
    perm = mirror_permutation(np.nanmean(seq.points, axis=0))
    a = augment(seq, AUG, np.random.default_rng(5), perm)
    b = augment(seq, AUG, np.random.default_rng(5), perm)
    assert np.array_equal(a.points, b.points, equal_nan=True)
    pos = time_positions(30, 16, AUG, np.random.default_rng(1))
    assert np.all(np.diff(pos) > 0) and pos[0] >= 0 and pos[-1] <= 29


def test_loso_never_shares_a_signer_between_train_val_and_test():
    folds = loso_folds([f"s{i:02d}" for i in range(12)])
    assert len(folds) == 12
    for f in folds:
        assert not set(f["train"]) & set(f["test"] + f["val"])
        assert f["test"] != f["val"] and len(f["train"]) == 10
    assert sorted(s for f in folds for s in f["test"]) == sorted(f"s{i:02d}" for i in range(12))


def test_models_give_logits_and_imprinting_classifies_by_prototype():
    items = [Item(f"x{i}", "minds", "s", g, prepare(fake_landmarks(seed=i), FEAT))
             for i, g in enumerate("ab")]
    ds = SignDataset(items, ["a", "b"], FEAT, AUG, seed=0)
    x = torch.stack([ds[i][0] for i in range(2)])
    for cfg in ({"encoder": "stats_mlp", "hidden": 32, "dropout": 0.0, "cosine_scale": 16.0},
                {"encoder": "gru", "hidden": 16, "layers": 2, "bidirectional": True,
                 "dropout": 0.0, "embedding": 24, "cosine_scale": 16.0}):
        model = build_model(cfg, n_features(FEAT), 2).eval()
        logits, emb = model(x)
        assert logits.shape == (2, 2)
        model.classifier.imprint(emb.detach())
        assert model(x)[0].argmax(1).tolist() == [0, 1]


def test_f1_ignores_classes_the_test_signer_never_recorded():
    from libras.train.evaluate import scores

    logits = np.eye(3)[[0, 1, 1]]  # class 2 absent from the test set and never predicted
    s = scores(logits, [0, 1, 1], n_classes=3)
    assert s["top1"] == 1.0 and s["f1_macro"] == 1.0
