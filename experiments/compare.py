"""Experiment 1: accuracy and speed of closed-set and open-vocabulary detectors on a COCO val2017 subset.

ap and timing write to the same results/compare.csv, one row per (model, n_images):
  --phase ap      pycocotools bbox AP on the subset, plus VRAM peak and params.
  --phase timing  batch-1 latency/FPS after warm-up and text-encoding cost. Run it with nothing else on the GPU.
  --phase plot    redraw results/compare.png/.svg from the CSV.
  --phase all     ap + timing + plot.

Run:
  python -u experiments/compare.py --phase ap --n-images 500 --device cuda:0
  python -u experiments/compare.py --phase timing --n-images 500 --device cuda:0

All models: the 80 plain COCO category names as vocabulary, score threshold 0.001, max 100 detections per image,
fp16 (--fp32 to change), batch size 1. Timing is wall-clock time per image for preprocess + forward +
postprocess, excluding disk I/O and JPEG decode. FPS is 1000 / median ms (the mean is also saved).

The Ultralytics models go through predict(), which keeps one class per box. OWLv2 and Grounding DINO keep the
top 100 (box, class) pairs. sanity_check.py measures what that costs the YOLO models (about 0.6 to 1.0 AP).
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # checkpoints are downloaded beforehand

import numpy as np
import pandas as pd

from common import (DATA, RESULTS, Timer, coco_eval, coco_names, ensure_images, env_info, is_cuda, load_coco,
                    load_images_rgb, log, plot_style, seed_everything, subset_ids, to_coco_dets)

CSV = RESULTS / "compare.csv"
META = RESULTS / "compare_meta.json"
DETS = DATA / "dets"

_SUPERVISED = "Supervised on COCO train2017 (all 80 classes)"
_WORLD = "O365v1 + GoldG (GQA + Flickr30k, COCO images excluded), per the official model zoo; converted by Ultralytics"
_YOLOE = "O365v1 + GoldG (GQA + Flickr30k, COCO images excluded) with SAM 2.1 pseudo-masks (YOLOE paper)"

MODELS = {
    "yolov8s": dict(label="YOLOv8-S (closed-set)", family="ultralytics", kind="yolo", ckpt="weights/yolov8s.pt",
                    open_vocab=False, train=_SUPERVISED),
    "yolov8m": dict(label="YOLOv8-M (closed-set)", family="ultralytics", kind="yolo", ckpt="weights/yolov8m.pt",
                    open_vocab=False, train=_SUPERVISED),
    "yolov8l": dict(label="YOLOv8-L (closed-set)", family="ultralytics", kind="yolo", ckpt="weights/yolov8l.pt",
                    open_vocab=False, train=_SUPERVISED),
    "yolov8s-worldv2": dict(label="YOLO-World-S v2", family="ultralytics", kind="world",
                            ckpt="weights/yolov8s-worldv2.pt", open_vocab=True, train=_WORLD),
    "yolov8m-worldv2": dict(label="YOLO-World-M v2", family="ultralytics", kind="world",
                            ckpt="weights/yolov8m-worldv2.pt", open_vocab=True, train=_WORLD),
    "yolov8l-worldv2": dict(label="YOLO-World-L v2", family="ultralytics", kind="world",
                            ckpt="weights/yolov8l-worldv2.pt", open_vocab=True, train=_WORLD),
    "yoloe-v8s": dict(label="YOLOE-v8-S", family="ultralytics", kind="yoloe", ckpt="weights/yoloe-v8s-seg.pt",
                      open_vocab=True, train=_YOLOE),
    "yoloe-11s": dict(label="YOLOE-11-S", family="ultralytics", kind="yoloe", ckpt="weights/yoloe-11s-seg.pt",
                      open_vocab=True, train=_YOLOE),
    "yoloe-26s": dict(label="YOLOE-26-S", family="ultralytics", kind="yoloe", ckpt="weights/yoloe-26s-seg.pt",
                      open_vocab=True, train=_YOLOE),
    "gdino-t": dict(label="Grounding DINO-T", family="gdino", ckpt="IDEA-Research/grounding-dino-tiny",
                    open_vocab=True, train="O365 + GoldG + Cap4M (groundingdino_swint_ogc; official repo model zoo)"),
    "owlv2-b16": dict(label="OWLv2-B/16", family="owlv2", ckpt="google/owlv2-base-patch16-ensemble", open_vocab=True,
                      train="CLIP pre-train; self-train on WebLI pseudo-boxes; fine-tune on LVIS-base (COCO train2017 images); weight ensemble"),
}


def build_runner(key, device, half):
    from runners import GroundingDINORunner, OWLv2Runner, UltralyticsRunner

    m = MODELS[key]
    if m["family"] == "ultralytics":
        return UltralyticsRunner(m["ckpt"], m["kind"], device=device, half=half)
    if m["family"] == "owlv2":
        return OWLv2Runner(device=device, half=half)
    return GroundingDINORunner(device=device, half=half)


def load_rows() -> pd.DataFrame:
    return pd.read_csv(CSV) if CSV.exists() else pd.DataFrame()


def upsert(row: dict) -> None:
    """Insert or update the (model, n_images) row."""
    df = load_rows()
    if len(df):
        mask = (df["model"] == row["model"]) & (df["n_images"] == row["n_images"])
        if mask.any():
            for k, v in row.items():
                if k not in df.columns:
                    df[k] = None
                df[k] = df[k].astype(object)
                df.loc[mask, k] = v
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(CSV, index=False)


# AP phase
def run_ap(key, args, coco, ids, cat_ids, names, rgb, bgr):
    import torch

    m = MODELS[key]
    cuda = is_cuda(args.device)
    log(f"[AP] {key}: loading")
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    r = build_runner(key, args.device, not args.fp32)
    r.set_vocab(names)
    if m.get("kind") == "yolo":  # the 80 output indices have to be in COCO's sorted category order
        assert list(r.names) == names, "YOLOv8 class order differs from COCO category order"
    ims = bgr if m["family"] == "ultralytics" else rgb
    dets, t0 = [], time.time()
    for n, (iid, im) in enumerate(zip(ids, ims)):
        b, s, c = r.predict(im, conf=args.conf)
        dets += to_coco_dets(iid, b, s, [cat_ids[k] for k in c])
        if (n + 1) % 100 == 0:
            log(f"[AP] {key}: {n + 1}/{len(ids)} images ({time.time() - t0:.0f}s)")
    wall = time.time() - t0
    DETS.mkdir(parents=True, exist_ok=True)
    (DETS / f"compare_{key}_n{len(ids)}.json").write_text(json.dumps(dets))
    ap = coco_eval(coco, dets, ids)
    info = r.info()
    row = {"model": key, "label": m["label"], "n_images": len(ids), "checkpoint": m["ckpt"],
           "open_vocab": m["open_vocab"], "training_data": m["train"], **info, **ap,
           "vram_peak_MB_ap_pass": round(torch.cuda.max_memory_allocated() / 2**20) if cuda else None,
           "score_thr": args.conf, "max_det": 100, "ap_pass_wall_s": round(wall, 1), "gpu": args.gpu_name,
           "ap_timestamp": time.strftime("%Y-%m-%d %H:%M")}
    upsert(row)
    log(f"[AP] {key}: AP={ap['AP']} AP50={ap['AP50']} (wall {wall:.0f}s)")
    del r
    if cuda:
        torch.cuda.empty_cache()
    return row


# timing phase (run alone on the GPU)
def run_timing(key, args, ids, names, rgb, bgr):
    import torch

    m = MODELS[key]
    cuda = is_cuda(args.device)
    log(f"[T] {key}: loading")
    r = build_runner(key, args.device, not args.fp32)
    ims = bgr if m["family"] == "ultralytics" else rgb

    # text encoding (done once per vocabulary)
    text = {}
    if m["open_vocab"]:
        with Timer(cuda) as T:
            r.set_vocab(names)  # cold: includes loading the text encoder for Ultralytics models
        text["text_encode_cold_ms"] = round(T.ms, 1)
        reps = []
        for _ in range(10):
            with Timer(cuda) as T:
                r.encode_text_only(names)
            reps.append(T.ms)
        text["text_encode_warm_ms"] = round(float(np.median(reps)), 2)
    else:
        r.set_vocab(names)

    # warm-up, then the timed pass over all subset images
    for im in ims[: args.warmup]:
        r.predict(im, conf=args.conf)
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    t_all = []
    for im in ims:
        with Timer(cuda) as T:
            r.predict(im, conf=args.conf)
        t_all.append(T.ms)
    # same thing at conf 0.25 on the first 100 images (fewer boxes go through NMS)
    t_dep = []
    for im in ims[:100]:
        with Timer(cuda) as T:
            r.predict(im, conf=0.25)
        t_dep.append(T.ms)
    extra = {}
    if m["family"] == "owlv2":  # standard HF call that re-encodes all 80 text queries every image
        t_naive = []
        for im in ims[:100]:
            with Timer(cuda) as T:
                r.predict_naive(im, conf=args.conf)
            t_naive.append(T.ms)
        extra["ms_per_img_text_every_image"] = round(float(np.mean(t_naive)), 2)
    # FPS uses the median: the mean gets pulled up by a few slow frames (see ms_per_img_std)
    t = np.array(t_all)
    med = round(float(np.median(t)), 2)
    row = {"model": key, "n_images": len(ids), "precision": "fp32" if args.fp32 else "fp16", "batch": 1,
           "warmup_iters": args.warmup, "ms_per_img_mean": round(float(t.mean()), 2),
           "ms_per_img_median": med, "ms_per_img_std": round(float(t.std()), 2),
           "fps": round(1000.0 / med, 1), "ms_per_img_conf0.25_first100": round(float(np.mean(t_dep)), 2),
           "vram_peak_MB_timing": round(torch.cuda.max_memory_allocated() / 2**20) if cuda else None, **text, **extra,
           "gpu": args.gpu_name, "timing_timestamp": time.strftime("%Y-%m-%d %H:%M")}
    upsert(row)
    log(f"[T] {key}: {med} ms/img (median), {row['fps']} FPS; text {text}")
    del r
    if cuda:
        torch.cuda.empty_cache()


# where the text label goes next to a marker: (horizontal alignment, dx, dy) in points
LABEL_POS = {"yolov8s-worldv2": ("left", 14, -5), "yoloe-26s": ("right", -14, -5), "yoloe-11s": ("right", -14, -5),
             "yoloe-v8s": ("left", 14, -5)}


def plot(n_images: int | None = None):
    df = load_rows()
    if not len(df) or "fps" not in df.columns or df["fps"].isna().all():
        log("[plot] no timing data yet - plot skipped (timing pending)")
        return
    if n_images is None:
        n_images = int(df["n_images"].max())
    df = df[(df["n_images"] == n_images) & df["fps"].notna() & df["AP"].notna()]
    if not len(df):
        log("[plot] no rows with both AP and FPS - plot skipped")
        return
    gpu = next((g for g in df.get("gpu", pd.Series(dtype=object)).dropna()), "GPU")
    prec = next((p for p in df.get("precision", pd.Series(dtype=object)).dropna()), "fp16")
    plt = plot_style()
    fig, ax = plt.subplots(figsize=(16, 6.6))
    # one marker shape + color per model family
    style = {"yolo": ("s", "#535c69", "Closed-set YOLOv8 (trained on COCO)"),
             "world": ("o", "#0072B2", "YOLO-World v2 (open vocabulary)"),
             "yoloe": ("D", "#009E73", "YOLOE (open vocabulary)"),
             "gdino": ("^", "#D55E00", "Grounding DINO-T (open vocabulary)"),
             "owlv2": ("v", "#CC79A7", "OWLv2-B/16 (open vocabulary)")}
    seen = set()
    for _, r in df.iterrows():
        m = MODELS[r["model"]]
        fam = m.get("kind", m["family"])
        mk, col, fam_label = style[fam]
        ax.scatter(r["fps"], r["AP"], s=260, marker=mk, color=col, edgecolor="black", linewidth=1.2, zorder=3,
                   label=None if fam in seen else fam_label)
        seen.add(fam)
        short = str(r["label"]).replace(" (closed-set)", "").replace(" v2", "")
        # closed-set labels go under the marker, the rest above; the crowded small models get theirs to one side
        ha, dx, dy = LABEL_POS.get(r["model"], ("center", 0, -22 if fam == "yolo" else 13))
        ax.annotate(short, (r["fps"], r["AP"]), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, fontsize=14, color=col, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(df["fps"].min() / 1.25, df["fps"].max() * 1.3)
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    lo, hi = df["AP"].min(), df["AP"].max()
    ax.set_ylim(lo - 0.12 * (hi - lo) - 2, hi + 0.12 * (hi - lo) + 2)
    ax.axvline(30, color="#555555", linestyle="--", linewidth=2)
    ax.text(30, ax.get_ylim()[0], "  real time (30 FPS)", fontsize=14, color="#333333", va="bottom")
    ax.set_xlabel(f"Speed, FPS (log scale) - {gpu}, batch 1, {prec}")
    ax.set_ylabel("COCO box AP (%)")
    ax.set_title(f"Accuracy vs speed on {n_images} COCO val2017 images (80 COCO class names as prompts)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=13, frameon=True)
    fig.tight_layout()
    fig.savefig(RESULTS / "compare.png", dpi=200)
    fig.savefig(RESULTS / "compare.svg")
    log("[plot] wrote results/compare.png and .svg")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", choices=["ap", "timing", "plot", "all"], default="all")
    p.add_argument("--n-images", type=int, default=500)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--fp32", action="store_true")
    args = p.parse_args()

    seed_everything()
    t_start = time.time()
    if args.phase == "plot":
        plot(args.n_images)
        return
    info = env_info()
    log(f"env: {info}")
    args.gpu_name = info["gpu"] if is_cuda(args.device) else "CPU"
    coco = load_coco()
    ids = subset_ids(coco, args.n_images)
    cat_ids, names = coco_names(coco)
    rgb = load_images_rgb(ensure_images(coco, ids))
    bgr = [np.ascontiguousarray(im[..., ::-1]) for im in rgb]
    run = {"phase": args.phase, "args": vars(args), "env": info, "start": time.strftime("%Y-%m-%d %H:%M")}
    for key in args.models:
        try:
            if args.phase in ("ap", "all"):
                run_ap(key, args, coco, ids, cat_ids, names, rgb, bgr)
            if args.phase in ("timing", "all"):
                run_timing(key, args, ids, names, rgb, bgr)
        except Exception:  # one model failing should not stop the rest
            import traceback

            traceback.print_exc()
            log(f"FAILED: {key}")
    run["wall_s"] = round(time.time() - t_start, 1)
    meta = json.loads(META.read_text()) if META.exists() else {}
    meta.setdefault("runs", []).append(run)
    meta["vocabulary"] = names
    META.write_text(json.dumps(meta, indent=1))
    if args.phase in ("timing", "all"):
        plot(len(ids))
    log(f"done in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
