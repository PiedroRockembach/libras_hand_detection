"""Metrics, and re-evaluation of a finished training run without retraining.

    python -m libras.train.evaluate --run experiments/a1_gru_minds/<run>

Rebuilds the test sets with the current sample selection (e.g. after MALTA duplicates were
removed), loads each fold's model.pt and writes <run>/eval-<UTC time>-<commit>/ with
meta.json, fold_XX.json and summary.json. The training run itself is never modified.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import confusion_matrix, f1_score

from libras.data.dataset import Item, SignDataset, load_items
from libras.data.features import n_features
from libras.models import build_model
from libras.provenance import git_state


@torch.no_grad()
def predict(model, ds: SignDataset, device: str, batch: int) -> np.ndarray:
    model.eval()
    out = []
    for x, _ in torch.utils.data.DataLoader(ds, batch_size=batch):
        out.append(model(x.to(device))[0].float().cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, 0))


def scores(logits: np.ndarray, labels: list[int], n_classes: int) -> dict:
    y = np.asarray(labels)
    if len(y) == 0:
        return {"n": 0}
    top5 = np.argsort(-logits, axis=1)[:, :5]
    pred = top5[:, 0]
    return {"n": len(y), "top1": float((pred == y).mean()),
            "top5": float((top5 == y[:, None]).any(axis=1).mean()),
            # Over classes the test signer recorded or the model predicted: some MINDS signers
            # lack a few signs, and an absent, never-predicted class is not an error.
            "f1_macro": float(f1_score(y, pred, labels=sorted(set(y) | set(pred)),
                                       average="macro", zero_division=0)),
            "confusion": confusion_matrix(y, pred, labels=range(n_classes)).tolist()}


def malta_scores(logits: np.ndarray, items: list[Item], labels: list[int], n_classes: int) -> dict:
    """Overall and per sub-collection (ADR-0015: MALTA results are reported per collection)."""
    res = {"all": scores(logits, labels, n_classes)}
    colls = sorted({i.sample_id.split(":")[1] for i in items})
    for c in colls:
        idx = [k for k, i in enumerate(items) if i.sample_id.split(":")[1] == c]
        s = scores(logits[idx], [labels[k] for k in idx], n_classes)
        res[c] = {k: v for k, v in s.items() if k != "confusion"}
    return res


def mean_std(values: list[float]) -> dict:
    return {"mean": float(np.mean(values)), "std": float(np.std(values)), "n": len(values)}


def summarize(results: list[dict]) -> dict:
    keys = ("top1", "top5", "f1_macro")
    summary = {"folds": len(results),
               "test": {k: mean_std([r["test"][k] for r in results]) for k in keys}}
    if results and results[0]["malta"]:
        colls = [c for c in results[0]["malta"] if c != "all"]
        summary["malta"] = {k: mean_std([r["malta"]["all"][k] for r in results]) for k in keys}
        summary["malta_by_collection"] = {
            c: {"n": results[0]["malta"][c]["n"],
                "top1": mean_std([r["malta"][c]["top1"] for r in results])} for c in colls}
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--run", type=Path, required=True, help="run directory (absolute or under "
                   "the data root)")
    p.add_argument("--data-root", type=Path, default=os.environ.get("LIBRAS_DATA_ROOT"))
    p.add_argument("--device", default="auto")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    if not args.data_root:
        p.error("set --data-root or LIBRAS_DATA_ROOT")
    root = args.data_root.resolve()
    run = args.run if args.run.is_absolute() else root / args.run
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    classes = json.loads((run / "meta.json").read_text())["classes"]
    folds = json.loads((run / "splits.json").read_text())
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    items, malta = load_items(root, cfg["data"], cfg["features"], args.workers)
    fc, batch = cfg["features"], cfg["train"]["batch_size"]
    malta_ds = SignDataset(malta, classes, fc)
    git = git_state()
    out = run / (f"eval-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-"
                 f"{(git['commit'] or 'nogit')[:7]}")
    out.mkdir()
    (out / "meta.json").write_text(json.dumps({
        "command": sys.argv, **git, "device": device, "n_samples": len(items),
        "n_malta": len(malta), "malta_ids": [i.sample_id for i in malta]}, indent=1))

    results = []
    for fold in folds:
        model = build_model(cfg["model"], n_features(fc), len(classes)).to(device)
        state = torch.load(run / f"fold_{fold['fold']:02d}" / "model.pt", map_location=device)
        model.load_state_dict(state)
        test_ds = SignDataset([i for i in items if i.signer in fold["test"] and not i.train_only],
                              classes, fc)
        r = {"fold": fold["fold"], "test_signer": fold["test"],
             "test": scores(predict(model, test_ds, device, batch), test_ds.labels, len(classes)),
             "malta": malta_scores(predict(model, malta_ds, device, batch), malta,
                                   malta_ds.labels, len(classes)) if malta else {}}
        (out / f"fold_{fold['fold']:02d}.json").write_text(json.dumps(r, indent=1))
        results.append(r)

    summary = summarize(results)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    t = summary["test"]
    print(f"{len(malta)} MALTA clips. LOSO top1 {t['top1']['mean']:.3f} ± {t['top1']['std']:.3f}")
    if "malta" in summary:
        m = summary["malta"]
        print(f"MALTA top1 {m['top1']['mean']:.3f} ± {m['top1']['std']:.3f}  "
              f"top5 {m['top5']['mean']:.3f}  F1-macro {m['f1_macro']['mean']:.3f}")
        for c, v in summary["malta_by_collection"].items():
            print(f"  {c:18} n={v['n']:3d}  top1 {v['top1']['mean']:.3f}")
    print(f"results: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
