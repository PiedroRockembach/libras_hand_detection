"""Samples, signer-independent splits and the torch Dataset for isolated signs."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from libras.data.augment import augment, mirror_permutation, time_positions
from libras.data.features import Prepared, load, to_features
from libras.registry.schema import TABLES
from libras.registry.select import landmark_path, select_samples
from libras.registry.store import generated_dir, read_table


@dataclass
class Item:
    sample_id: str
    source_id: str
    signer: str  # the person: views of the same person (same_person_as) share one id
    gloss: str
    seq: Prepared
    train_only: bool = False  # extra view (e.g. MINDS Kinect): never in validation or test


def _load(args: tuple[str, dict]) -> Prepared:
    return load(Path(args[0]), args[1])


def load_items(root: Path, data_cfg: dict, feat_cfg: dict, workers: int = 4
               ) -> tuple[list[Item], list[Item]]:
    """Registry samples of the configured sources (plus ``extra_train_sources``, training
    only), and the MALTA look-alikes used as the out-of-domain test (ADR-0014 §5).
    Returns (in-domain items, MALTA items)."""
    extra = data_cfg.get("extra_train_sources", [])
    rows = select_samples(root, data_cfg["sources"] + extra, None, None,
                          data_cfg.get("ood_malta_like"))
    rows = [r for r in rows if "duplicate_video" not in r.get("quality_flags", "")]
    paths = [str(landmark_path(root, data_cfg["landmarks"], r)) for r in rows]
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} samples have no landmarks, e.g. {missing[0]}")
    with ProcessPoolExecutor(workers) as pool:
        seqs = list(pool.map(_load, [(p, feat_cfg) for p in paths], chunksize=16))
    person = {s["signer_uid"]: s["same_person_as"] or s["signer_uid"]
              for s in read_table(generated_dir(root) / "signers.csv", TABLES["signers"])}
    items = [Item(r["sample_id"], r["source_id"], person.get(r["signer_uid"], r["signer_uid"]),
                  r["gloss"], s, r["source_id"] in extra) for r, s in zip(rows, seqs)]
    return ([i for i in items if i.source_id != "malta"],
            [i for i in items if i.source_id == "malta"])


def loso_folds(signers: list[str]) -> list[dict]:
    """Leave-one-signer-out (ADR-0004): fold i tests signer i and validates (model selection)
    on the next signer in sorted order; the rest train."""
    signers = sorted(signers)
    n = len(signers)
    return [{"fold": i, "test": [signers[i]], "val": [signers[(i + 1) % n]],
             "train": [s for j, s in enumerate(signers) if j not in (i, (i + 1) % n)]}
            for i in range(n)]


class SignDataset(torch.utils.data.Dataset):
    """Isolated signs as (features (frames, F), label). Augments when ``aug_cfg`` is given;
    call ``set_epoch`` so every epoch draws new, but seed-determined, augmentations."""

    def __init__(self, items: list[Item], classes: list[str], feat_cfg: dict,
                 aug_cfg: dict | None = None, seed: int = 0):
        self.items, self.feat_cfg, self.aug_cfg, self.seed = items, feat_cfg, aug_cfg, seed
        index = {g: i for i, g in enumerate(classes)}
        self.labels = [index[i.gloss] for i in items]
        self.mirror_perm = None
        if aug_cfg and aug_cfg.get("mirror", 0.0):
            mean = np.nanmean(np.concatenate([i.seq.points for i in items]), axis=0)
            self.mirror_perm = mirror_permutation(mean)
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        self.rng = np.random.default_rng([self.seed, epoch])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        seq = self.items[i].seq
        if self.aug_cfg is None:
            x = to_features(seq, self.feat_cfg)
        else:
            seq = augment(seq, self.aug_cfg, self.rng, self.mirror_perm)
            pos = time_positions(len(seq.points), self.feat_cfg["frames"], self.aug_cfg, self.rng)
            x = to_features(seq, self.feat_cfg, pos)
        return torch.from_numpy(x), self.labels[i]
