"""Pre-compute the demo results, in case the live demo can't run.

Runs YOLO-World and YOLOv8 on every sample image (SAMPLES in demo/core.py) and writes
results/demo/<name>_side_by_side.png and results/demo/prebaked.json.

Usage (from the repo root):  python scripts/prebake.py [--size S] [--conf 0.1]
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "demo"))
import core
from core import (
    DEVICE_NAME, IMAGES, SAMPLES, ModelBank, coco_class_for, draw_detections, load_image, parse_vocab, side_by_side,
)

OUT = REPO / "results" / "demo"


def timed(fn, n: int):
    """Run fn n times, return (last result, median inference ms, median predict ms)."""
    res, inf, pred = None, [], []
    for _ in range(n):
        res = fn()
        inf.append(res.inference_ms)
        pred.append(res.predict_ms)
    return res, statistics.median(inf), statistics.median(pred)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="S", choices=list(core.SIZES))
    ap.add_argument("--conf", type=float, default=core.DEFAULT_CONF)
    ap.add_argument("--conf-closed", type=float, default=core.DEFAULT_CONF_CLOSED)
    ap.add_argument("--repeats", type=int, default=5, help="timed repeats per image (median reported)")
    args = ap.parse_args()

    import torch
    import ultralytics

    OUT.mkdir(parents=True, exist_ok=True)
    bank = ModelBank()
    print(f"Warming up size {args.size} on {DEVICE_NAME} ...")
    bank.warmup([args.size])
    coco80 = {n.lower() for n in bank._closed(args.size).names.values()}

    records = []
    for s in SAMPLES:
        path = IMAGES / s["file"]
        vocab = parse_vocab(s["vocab"])
        img = load_image(path)
        conf = float(s.get("conf", args.conf))
        # encode the vocabulary once (timed), then time the detector alone
        first = bank.run_world(img, vocab, args.size, conf=conf)
        world, w_inf, w_pred = timed(lambda: bank.run_world(img, vocab, args.size, conf=conf), args.repeats)
        closed, c_inf, c_pred = timed(lambda: bank.run_closed(img, args.size, conf=args.conf_closed), args.repeats)

        stem = Path(s["file"]).stem
        left = draw_detections(img, world.detections, title=f"YOLO-World-{args.size} (open vocabulary)",
                               subtitle="words: " + ", ".join(vocab), title_color=(11, 122, 62), min_long_side=1000)
        not_coco = [w for w in vocab if coco_class_for(w, coco80) is None]
        mapped = {w: coco_class_for(w, coco80) for w in vocab if coco_class_for(w, coco80) not in (None, w.lower())}
        right = draw_detections(img, closed.detections, title=f"YOLOv8{args.size.lower()} (closed set, 80 COCO classes)",
                                subtitle=("no COCO class for: " + ", ".join(not_coco)) if not_coco else "every word has a COCO class",
                                title_color=(154, 52, 18), min_long_side=1000)
        png = OUT / f"{stem}_side_by_side.png"
        side_by_side(left, right).save(png, optimize=True)

        found = sorted({d.label for d in world.detections})
        rec = {
            "image": f"demo/images/{s['file']}",
            "output_png": str(png.relative_to(REPO)).replace("\\", "/"),
            "vocabulary": vocab,
            "vocab_words_without_coco80_class": not_coco,
            "vocab_words_mapped_to_coco80": mapped,
            "yolo_world": {
                "model": f"yolov8{args.size.lower()}-worldv2.pt",
                "conf_threshold": conf,
                "text_encoding_ms_once": round(first.encode_ms or 0.0, 1),
                "inference_ms_median": round(w_inf, 2),
                "predict_ms_median": round(w_pred, 2),
                "detections": [d.as_dict() for d in world.detections],
                "vocab_words_found": found,
                "vocab_words_not_found": [w for w in vocab if w not in found],
            },
            "yolov8_closed_set": {
                "model": f"yolov8{args.size.lower()}.pt",
                "conf_threshold": args.conf_closed,
                "inference_ms_median": round(c_inf, 2),
                "predict_ms_median": round(c_pred, 2),
                "detections": [d.as_dict() for d in closed.detections],
            },
            "missed_by_yolo_world_but_found_by_yolov8": sorted(
                w for w, c in {**{w: w.lower() for w in vocab}, **mapped}.items()
                if c in {d.label for d in closed.detections} and w not in found),
            "note": s.get("note", ""),
            "failure_case": bool(s.get("failure_case", False)),
        }
        records.append(rec)
        print(f"{s['file']:22s} world {w_inf:5.1f} ms  found {found}")
        print(f"{'':22s} yolov8 {c_inf:5.1f} ms  found {sorted({d.label for d in closed.detections})}")

    meta = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": DEVICE_NAME,
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "python": platform.python_version(),
        "imgsz": 640,
        "timing_note": ("inference_ms = model forward pass only (Ultralytics' timer), median of "
                        f"{args.repeats} runs after warm-up; predict_ms adds pre-processing and NMS. "
                        "text_encoding_ms_once = CLIP text encoding of the vocabulary, done once per vocabulary."),
        "device_note": f"Run on {DEVICE_NAME}; timings are indicative, not a clean benchmark.",
        "failure_cases": [r["image"] for r in records if r["failure_case"]],
        "images": records,
    }
    (OUT / "prebaked.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} PNGs + prebaked.json to {OUT}")


if __name__ == "__main__":
    main()
