"""Train and evaluate an isolated-sign model with leave-one-signer-out (ADR-0004, ADR-0014).

    python -m libras.train --config configs/experiments/a1_gru_minds.yaml
    python -m libras.train --config ... --folds 0 1 --epochs 3     # quick check

Each fold trains from scratch on 10 signers, picks the epoch by accuracy on a separate
validation signer, and is tested on the held-out signer and on the MALTA look-alikes
(out of domain). Everything goes to LIBRAS_DATA_ROOT/experiments/<name>/<run>/:
config.yaml (as run), meta.json (commit, seed, versions), splits.json, fold_XX/{metrics.json,
model.pt} and summary.json (mean +- std over folds). Numbers are only reported from there.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from libras.data.dataset import Item, SignDataset, load_items, loso_folds
from libras.data.features import n_features
from libras.models import build_model
from libras.progress import Progress
from libras.provenance import git_state
from libras.train.evaluate import malta_scores, predict, scores, summarize


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_fold(fold: dict, items: list[Item], malta: list[Item], classes: list[str], cfg: dict,
             device: str, out: Path, progress: Progress) -> dict:
    seed = cfg["seed"] * 1000 + fold["fold"]
    seed_everything(seed)
    by = {k: [i for i in items if i.signer in fold[k] and (k == "train" or not i.train_only)]
          for k in ("train", "val", "test")}
    fc, tc = cfg["features"], cfg["train"]
    train_ds = SignDataset(by["train"], classes, fc, cfg["augment"], seed)
    val_ds, test_ds = (SignDataset(by[k], classes, fc) for k in ("val", "test"))
    malta_ds = SignDataset(malta, classes, fc)

    model = build_model(cfg["model"], n_features(fc), len(classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=tc["lr"], total_steps=tc["epochs"] * -(-len(train_ds) // tc["batch_size"]),
        pct_start=tc.get("warmup", 0.1))
    gen = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=tc["batch_size"], shuffle=True,
                                         generator=gen, drop_last=False)
    best, history = (-1.0, -1), []
    for epoch in range(tc["epochs"]):
        model.train()
        train_ds.set_epoch(epoch)
        total, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x)[0], y, label_smoothing=tc["label_smoothing"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.get("grad_clip", 1.0))
            opt.step()
            sched.step()
            total, n = total + loss.item() * len(y), n + len(y)
        val = scores(predict(model, val_ds, device, tc["batch_size"]), val_ds.labels, len(classes))
        history.append({"epoch": epoch, "loss": total / n, "val_top1": val["top1"]})
        if val["top1"] > best[0]:
            best = (val["top1"], epoch)
            torch.save(model.state_dict(), out / "model.pt")
        progress.advance()

    model.load_state_dict(torch.load(out / "model.pt", map_location=device))
    test = scores(predict(model, test_ds, device, tc["batch_size"]), test_ds.labels, len(classes))
    ood = malta_scores(predict(model, malta_ds, device, tc["batch_size"]), malta, malta_ds.labels,
                       len(classes)) if malta else {}
    metrics = {"fold": fold["fold"], "test_signer": fold["test"], "val_signer": fold["val"],
               "seed": seed, "best_epoch": best[1], "val_top1": best[0],
               "n_train": len(train_ds), "test": test, "malta": ood, "history": history}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    return metrics


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=os.environ.get("LIBRAS_DATA_ROOT"))
    p.add_argument("--folds", type=int, nargs="*", help="only these folds (default: all)")
    p.add_argument("--epochs", type=int, help="override train.epochs (quick checks)")
    p.add_argument("--device", default="auto")
    p.add_argument("--workers", type=int, default=8, help="processes loading landmarks")
    args = p.parse_args()
    if not args.data_root:
        p.error("set --data-root or LIBRAS_DATA_ROOT")
    root = args.data_root.resolve()
    cfg = yaml.safe_load(args.config.read_text())
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    items, malta = load_items(root, cfg["data"], cfg["features"], args.workers)
    classes = sorted({i.gloss for i in items})
    folds = loso_folds(sorted({i.signer for i in items if not i.train_only}))
    if args.folds is not None:
        folds = [f for f in folds if f["fold"] in args.folds]

    git = git_state()
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"-{(git['commit'] or 'nogit')[:7]}"
    run_dir = root / "experiments" / cfg["name"] / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    (run_dir / "meta.json").write_text(json.dumps({
        "command": sys.argv, **git, "config_file": str(args.config), "device": device,
        "torch": torch.__version__, "numpy": np.__version__, "classes": classes,
        "n_samples": len(items), "n_malta": len(malta), "folds_run": [f["fold"] for f in folds],
        "n_parameters": sum(t.numel() for t in build_model(
            cfg["model"], n_features(cfg["features"]), len(classes)).parameters()),
    }, indent=1, ensure_ascii=False))
    (run_dir / "splits.json").write_text(json.dumps(folds, indent=1))
    print(f"{len(items)} samples, {len(classes)} classes, {len(malta)} MALTA, "
          f"{len(folds)} folds on {device} -> {run_dir}", flush=True)

    results = []
    with Progress(f"train-{cfg['name']}", f"Treino {cfg['name']}",
                  len(folds) * cfg["train"]["epochs"], "épocas") as progress:
        for fold in folds:
            out = run_dir / f"fold_{fold['fold']:02d}"
            out.mkdir()
            r = run_fold(fold, items, malta, classes, cfg, device, out, progress)
            results.append(r)
            ood = f"  MALTA {r['malta']['all']['top1']:.3f}" if r["malta"] else ""
            print(f"fold {fold['fold']:2d} test {fold['test'][0]:>10}  top1 {r['test']['top1']:.3f}"
                  f"  f1 {r['test']['f1_macro']:.3f}  (best epoch {r['best_epoch']}){ood}", flush=True)

    summary = summarize(results)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    t = summary["test"]
    print(f"LOSO top1 {t['top1']['mean']:.3f} ± {t['top1']['std']:.3f}  "
          f"top5 {t['top5']['mean']:.3f}  F1-macro {t['f1_macro']['mean']:.3f} ± "
          f"{t['f1_macro']['std']:.3f}")
    if "malta" in summary:
        print(f"MALTA top1 {summary['malta']['top1']['mean']:.3f} ± "
              f"{summary['malta']['top1']['std']:.3f}")
    print(f"results: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
