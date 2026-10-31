"""Isolated-sign demo (demo v0, ADR-0009/0014) and webcam test-set recording.

    python -m libras.demo live  --run experiments/a6_gru_mirror_kinect/final-<run>
    python -m libras.demo video --run ... VIDEO [--out annotated.mp4] [--whole]
    python -m libras.demo record --run ... --signer p01 --takes 3 --consent-confirmed

live:   webcam window. Signs are cut automatically (arm raised -> lowered) and the top-5 is
        shown. Keys: space = start/stop a sign by hand, a = automatic cutting on/off, q = quit.
video:  the same pipeline on a file, printing each detected sign (no window; runs headless).
record: prompts each gloss of the model, records takes into
        LIBRAS_DATA_ROOT/raw/webcam_v0/<signer>/ with a takes.csv, and logs the session to
        PROVENANCE.jsonl. These clips are the out-of-domain test set that was never used to
        choose a model; record only people who agreed to it (--consent-confirmed).

The demo runs natively (it needs the webcam and a window): see docs/explicacao_guiada.md.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import unicodedata
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

from libras.demo.stream import Recognizer, Segmenter, StreamingExtractor

EXTRACTION = Path(__file__).resolve().parents[3] / "configs" / "extraction"


def extractor(root: Path, run_cfg: dict) -> StreamingExtractor:
    from libras.extract.__main__ import model_paths

    cfg = yaml.safe_load((EXTRACTION / f"{run_cfg['data']['landmarks']}.yaml").read_text())
    return StreamingExtractor(cfg, model_paths(cfg, root))


def fmt(top: list[tuple[str, float]]) -> str:
    return "  ".join(f"{g} {p:.0%}" for g, p in top)


def cmd_video(args, root: Path, rec: Recognizer) -> None:
    import cv2

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ex, seg = extractor(root, rec.cfg), Segmenter()
    writer, frames, size, i = None, [], None, 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            size = (bgr.shape[1], bgr.shape[0])
            f = ex.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), round(i * 1000 / fps))
            frames.append(f)
            done = None if args.whole else seg.push(f)
            if done:
                print(f"frames {i - len(done) + 1}-{i}: {fmt(rec.classify(done, size))}", flush=True)
            if args.out:
                if writer is None:
                    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
                draw(bgr, f, seg.active)
                writer.write(bgr)
            i += 1
    finally:
        ex.close()
        cap.release()
        if writer:
            writer.release()
    if args.whole and frames:
        print(f"whole video ({len(frames)} frames): {fmt(rec.classify(frames, size))}")
    elif not args.whole and seg.active and (done := seg.cut()):
        print(f"frames {i - len(done)}-{i - 1}: {fmt(rec.classify(done, size))}")


def draw(bgr, frame: dict, active: bool) -> None:
    """Landmarks and the 'sign in progress' dot, on the frame as the model sees it."""
    import cv2

    h, w = bgr.shape[:2]
    for hand in frame["hands"]:
        for x, y, _ in hand:
            if not np.isnan(x):
                cv2.circle(bgr, (int(x * w), int(y * h)), 3, (0, 220, 255), -1)
    for j in (0, 11, 12, 13, 14, 15, 16):
        x, y = frame["pose"][j, :2]
        if not np.isnan(x):
            cv2.circle(bgr, (int(x * w), int(y * h)), 5, (255, 180, 0), -1)
    if active:
        cv2.circle(bgr, (w - 30, 30), 12, (0, 0, 255), -1)


def overlay(view, top: list[tuple[str, float]], note: str = "") -> None:
    """Top-k bars and a help line, drawn last (after any flip) so the text reads normally."""
    import cv2

    y = 70
    for g, p in top:
        cv2.rectangle(view, (20, y - 18), (20 + int(260 * p), y + 4), (60, 160, 60), -1)
        cv2.putText(view, f"{g} {p:.0%}", (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 30
    if note:
        cv2.putText(view, note, (20, view.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)


def open_camera(index: int):
    import cv2

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(f"no webcam at index {index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


def cmd_live(args, root: Path, rec: Recognizer) -> None:
    import cv2

    cap, ex, seg = open_camera(args.camera), extractor(root, rec.cfg), Segmenter()
    auto, manual, top, t0 = True, None, [], time.monotonic()
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            size = (bgr.shape[1], bgr.shape[0])
            f = ex.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), int((time.monotonic() - t0) * 1000))
            done = None
            if manual is not None:
                manual.append(f)
            elif auto:
                done = seg.push(f)
            if done:
                top = rec.classify(done, size)
                print(fmt(top), flush=True)
            draw(bgr, f, seg.active or manual is not None)
            view = cv2.flip(bgr, 1) if args.mirror_view else bgr
            overlay(view, top, "espaco: sinal manual | a: automatico | q: sair"
                    + ("" if auto else " (automatico desligado)"))
            cv2.imshow("Libras demo v0", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("a"):
                auto = not auto
                seg.cut()
            if key == ord(" "):
                if manual is None:
                    manual = []
                    seg.cut()
                else:
                    if len(manual) >= 8:
                        top = rec.classify(manual, size)
                        print(fmt(top), flush=True)
                    manual = None
    finally:
        ex.close()
        cap.release()
        cv2.destroyAllWindows()


def slug(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return "".join(c if c.isalnum() else "_" for c in ascii_text.lower()).strip("_")


def cmd_record(args, root: Path, rec: Recognizer) -> None:
    import cv2

    from libras.provenance import log_event

    if not args.consent_confirmed:
        raise SystemExit("record only people who agreed to it; then pass --consent-confirmed")
    base = root / "raw" / "webcam_v0"
    out_dir = base / args.signer
    out_dir.mkdir(parents=True, exist_ok=True)
    index = base / "takes.csv"
    new_index = not index.exists()
    glosses = args.glosses or rec.classes
    cap = open_camera(args.camera)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    saved: list[str] = []
    try:
        with index.open("a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if new_index:
                w.writerow(["file", "signer", "gloss", "take", "fps", "frames", "width", "height",
                            "recorded_at", "consent"])
            for gloss in glosses:
                take = 1
                while take <= args.takes:
                    frames = record_take(cap, f"{gloss}  ({take}/{args.takes})")
                    if frames is None:
                        return
                    if not frames:  # skipped
                        break
                    name = f"{slug(gloss)}__{take}__{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}.mp4"
                    h, wd = frames[0].shape[:2]
                    vw = cv2.VideoWriter(str(out_dir / name), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                         (wd, h))
                    for fr in frames:
                        vw.write(fr)
                    vw.release()
                    rel = str((out_dir / name).relative_to(root))
                    w.writerow([rel, args.signer, gloss, take, f"{fps:.3f}", len(frames), wd, h,
                                datetime.now(UTC).isoformat(timespec="seconds"), "confirmed"])
                    fh.flush()
                    saved.append(rel)
                    take += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if saved:
            log_event(root, "webcam.record", inputs=[], outputs=saved + [str(index.relative_to(root))],
                      result={"signer": args.signer, "takes": len(saved)})
        print(f"{len(saved)} takes saved under {out_dir}")


def record_take(cap, prompt: str) -> list | None:
    """Space starts and stops; r redoes; s skips the gloss; q quits. None = quit."""
    import cv2

    frames, recording = [], False
    while True:
        ok, bgr = cap.read()
        if not ok:
            return None
        if recording:
            frames.append(bgr.copy())
        view = cv2.flip(bgr, 1)
        cv2.putText(view, prompt, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        hint = "GRAVANDO - espaco para parar" if recording else "espaco: gravar | s: pular | q: sair"
        cv2.putText(view, hint, (20, view.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255) if recording else (255, 255, 255), 2)
        cv2.imshow("Gravacao webcam_v0", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return None
        if key == ord("s") and not recording:
            return []
        if key == ord("r"):
            frames, recording = [], False
        if key == ord(" "):
            if recording and len(frames) > 5:
                return frames
            recording = True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("mode", choices=["live", "video", "record"])
    p.add_argument("video", nargs="?", type=Path, help="video mode: input file")
    p.add_argument("--run", type=Path, required=True, help="trained run (absolute or under the "
                   "data root)")
    p.add_argument("--fold", type=int, help="use this LOSO fold's model instead of a final model")
    p.add_argument("--data-root", type=Path, default=os.environ.get("LIBRAS_DATA_ROOT"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--mirror-view", action=argparse.BooleanOptionalAction, default=True,
                   help="live: show the image like a mirror (the model sees it unflipped)")
    p.add_argument("--out", type=Path, help="video mode: write an annotated copy")
    p.add_argument("--whole", action="store_true", help="video mode: the whole file is one sign")
    p.add_argument("--signer", help="record: pseudonymous id of the person recorded")
    p.add_argument("--takes", type=int, default=3)
    p.add_argument("--glosses", nargs="*", help="record: these glosses (default: the model's)")
    p.add_argument("--consent-confirmed", action="store_true")
    args = p.parse_args()
    if not args.data_root:
        p.error("set --data-root or LIBRAS_DATA_ROOT")
    root = args.data_root.resolve()
    run = args.run if args.run.is_absolute() else root / args.run
    rec = Recognizer(run, args.device, args.fold)
    if args.mode == "video":
        if not args.video:
            p.error("video mode needs a file")
        cmd_video(args, root, rec)
    elif args.mode == "live":
        cmd_live(args, root, rec)
    else:
        if not args.signer:
            p.error("record needs --signer")
        cmd_record(args, root, rec)
    return 0


if __name__ == "__main__":
    sys.exit(main())
