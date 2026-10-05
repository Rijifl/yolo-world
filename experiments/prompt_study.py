"""Experiment 2: how the prompt (wording and vocabulary size) changes YOLO-World's accuracy and speed.

Same 500-image COCO val2017 subset and predict() pipeline as compare.py (YOLO-World v2 via Ultralytics).
Rows go to results/prompt_study.csv, keyed by (part, model, condition, n_images).

  A  wording and accuracy. The 80 COCO classes with four wordings from experiments/prompts.json: the COCO name,
     one synonym, a short description, and the template "a photo of a {name}". The ground truth is unchanged.
     Per-class AP goes to results/prompt_study_per_class.csv.
  B  vocabulary size and speed. Vocabularies of 1, 10, 80, 365 and ~1200 classes (COCO names first, then LVIS
     names in a fixed random order). Text-encoding time (once per vocabulary) and batch-1 ms/image.
     Run it with nothing else on the GPU.
  C  vocabulary size and accuracy. The 80 COCO names alone, with one blank " " entry appended (the padding the
     official YOLO-World demos add), and with LVIS names added as extra words. AP is scored on the 80 COCO
     classes. predict() keeps one label per box, and a box whose label is one of the extra words is dropped,
     so a box that goes to a near-synonym (LVIS "sofa" for COCO "couch") counts as a miss. All conditions use
     max_det=300, then the top 100 COCO-class boxes per image, so extra words cannot use up the 100 boxes.
Charts: results/prompt_study_wording.png (A), prompt_study_vocab.png (C), prompt_study_speed.png (B).

Run:
  python -u experiments/prompt_study.py --parts A C --device cuda:0
  python -u experiments/prompt_study.py --parts B --device cuda:0
  python -u experiments/prompt_study.py --parts plot
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import pandas as pd

from common import (RESULTS, SEED, Timer, clean_lvis_names, coco_eval, coco_eval_per_class, coco_names,
                    ensure_images, env_info, is_cuda, load_coco, load_images_rgb, log, plot_style, seed_everything,
                    subset_ids, to_coco_dets)

CSV = RESULTS / "prompt_study.csv"
CSV_CLASS = RESULTS / "prompt_study_per_class.csv"
PROMPTS = Path(__file__).with_name("prompts.json")

WORLD = {
    "yolov8s-worldv2": dict(label="YOLO-World-S v2", ckpt="weights/yolov8s-worldv2.pt"),
    "yolov8l-worldv2": dict(label="YOLO-World-L v2", ckpt="weights/yolov8l-worldv2.pt"),
}
VARIANTS = ["name", "synonym", "description", "template"]
VARIANT_LABEL = {"name": "COCO\nname", "synonym": "synonym", "description": "short\ndescription",
                 "template": '"a photo\nof a ..."'}
SIZES = [1, 10, 80, 365, 1203]
KEY = ["part", "model", "condition", "n_images"]


def upsert(csv: Path, rows: list[dict], key: list[str]) -> None:
    df = pd.read_csv(csv, dtype={"condition": str}) if csv.exists() else pd.DataFrame()
    new = pd.DataFrame(rows)
    if len(df):
        old = df.merge(new[key].drop_duplicates(), on=key, how="left", indicator=True)
        df = pd.concat([df[(old["_merge"] == "left_only").values], new], ignore_index=True)
    else:
        df = new
    df.to_csv(csv, index=False)


def build_vocab(names: list[str], size: int) -> list[str]:
    """First `size` entries of: the 80 COCO names, then cleaned LVIS names in a seed-0 shuffled order
    (so the 365-class vocabulary is a subset of the largest one)."""
    extra = clean_lvis_names(names)
    random.Random(SEED).shuffle(extra)
    return (list(names) + extra)[:size]


def predict_all(r, ims, ids, cat_ids, conf, n_coco=80, max_det=100):
    """Detections in COCO json format; class indices >= n_coco (the extra words) are dropped, top 100 kept."""
    dets = []
    for iid, im in zip(ids, ims):
        b, s, c = r.predict(im, conf=conf, max_det=max_det)
        keep = np.flatnonzero(c < n_coco)
        keep = keep[np.argsort(-s[keep], kind="stable")][:100]
        b, s, c = b[keep], s[keep], c[keep]
        dets += to_coco_dets(iid, b, s, [cat_ids[k] for k in c])
    return dets


# part A: wording
def part_a(key, r, args, coco, ids, cat_ids, names, bgr, base):
    classes = json.loads(PROMPTS.read_text(encoding="utf-8"))["classes"]
    assert [c["name"] for c in classes] == names, "prompts.json is not in COCO category order"
    rows, per_rows = [], []
    for v in VARIANTS:
        prompts = [c[v] for c in classes]
        r.set_vocab(prompts)
        ap, per = coco_eval_per_class(coco, predict_all(r, bgr, ids, cat_ids, args.conf), ids)
        log(f"[A] {key} {v}: AP={ap['AP']} AP50={ap['AP50']}")
        rows.append({**base, "part": "A", "condition": v, "vocab_size": len(prompts), **ap})
        for c, cid in zip(classes, cat_ids):
            per_rows.append({"model": key, "n_images": len(ids), "condition": v, "class": c["name"],
                             "prompt": c[v], "synonym_kind": c["synonym_kind"], "AP": per[cid]})
    upsert(CSV, rows, KEY)
    upsert(CSV_CLASS, per_rows, ["model", "n_images", "condition", "class"])


# part C: vocabulary size and accuracy
def part_c(key, r, args, coco, ids, cat_ids, names, bgr, base):
    conds = [("80", list(names)), ("80+blank", list(names) + [" "])]
    for size in SIZES:
        if size > 80:
            vocab = build_vocab(names, size)
            conds.append((str(len(vocab)), vocab))
    rows = []
    for cond, vocab in conds:
        r.set_vocab(vocab)
        ap = coco_eval(coco, predict_all(r, bgr, ids, cat_ids, args.conf, max_det=300), ids)
        log(f"[C] {key} vocab {cond}: AP={ap['AP']} AP50={ap['AP50']}")
        rows.append({**base, "part": "C", "condition": cond, "vocab_size": len(vocab), **ap})
    upsert(CSV, rows, KEY)


# part B (timed)
def part_b(key, r, args, ids, names, bgr, base):
    cuda = is_cuda(args.device)
    r.set_vocab(list(names))  # loads the CLIP text encoder, so the per-size encoding times below are warm
    for im in bgr[: args.warmup]:
        r.predict(im, conf=args.conf)
    rows = []
    for size in SIZES:
        vocab = build_vocab(names, size)
        with Timer(cuda) as T:
            r.set_vocab(vocab)
        enc_ms = T.ms
        for im in bgr[: args.warmup]:
            r.predict(im, conf=args.conf)
        t, t25 = [], []
        for im in bgr:
            with Timer(cuda) as T:
                r.predict(im, conf=args.conf)
            t.append(T.ms)
        for im in bgr[:100]:
            with Timer(cuda) as T:
                r.predict(im, conf=0.25)
            t25.append(T.ms)
        t = np.array(t)
        med = round(float(np.median(t)), 2)  # median, same as compare.py
        row = {**base, "part": "B", "condition": str(len(vocab)), "vocab_size": len(vocab),
               "text_encode_ms": round(enc_ms, 1), "ms_per_img_mean": round(float(t.mean()), 2),
               "ms_per_img_median": med, "fps": round(1000.0 / med, 1),
               "ms_per_img_conf0.25_first100": round(float(np.mean(t25)), 2)}
        log(f"[B] {key} vocab {len(vocab)}: {med} ms/img (median), {row['fps']} FPS, "
            f"text encoding {row['text_encode_ms']} ms")
        rows.append(row)
    upsert(CSV, rows, KEY)


def plot(n_images: int | None = None) -> None:
    if not CSV.exists():
        log("[plot] no results/prompt_study.csv yet")
        return
    df = pd.read_csv(CSV, dtype={"condition": str})
    if n_images is None:
        n_images = int(df["n_images"].max())
    df = df[df["n_images"] == n_images]
    plt = plot_style()
    # these charts are narrower than compare.png, so the text is bigger
    plt.rcParams.update({"font.size": 22, "axes.titlesize": 23, "axes.labelsize": 23, "xtick.labelsize": 21,
                         "ytick.labelsize": 21, "legend.fontsize": 19})
    colors = {"yolov8s-worldv2": "#56B4E9", "yolov8l-worldv2": "#0072B2"}
    models = [k for k in WORLD if k in set(df["model"])]

    a = df[df["part"] == "A"]
    if len(a):
        fig, ax = plt.subplots(figsize=(9, 6.6))
        w = 0.8 / max(len(models), 1)
        for i, k in enumerate(models):
            vals = [a[(a["model"] == k) & (a["condition"] == v)]["AP"].mean() for v in VARIANTS]
            xs = np.arange(len(VARIANTS)) + (i - (len(models) - 1) / 2) * w
            bars = ax.bar(xs, vals, w * 0.92, color=colors[k], edgecolor="black", label=WORLD[k]["label"])
            ax.bar_label(bars, fmt="%.1f", fontsize=19, padding=2)
        ax.set_xticks(np.arange(len(VARIANTS)), [VARIANT_LABEL[v] for v in VARIANTS])
        ax.set_ylabel("COCO box AP (%)")
        ax.set_title(f"Same 80 classes, four wordings ({n_images} images)")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.28)
        ax.legend(loc="upper center", ncol=2, columnspacing=1.0, handlelength=1.2)
        fig.tight_layout()
        fig.savefig(RESULTS / "prompt_study_wording.png", dpi=200)
        log("[plot] wrote results/prompt_study_wording.png")

    c = df[df["part"] == "C"]
    if len(c):
        fig, ax = plt.subplots(figsize=(9, 6.6))
        conds = list(dict.fromkeys(c.sort_values("vocab_size")["condition"]))
        order = [x for x in ["80", "80+blank"] if x in conds] + [x for x in conds if x not in ("80", "80+blank")]
        w = 0.8 / max(len(models), 1)
        for i, k in enumerate(models):
            vals = [c[(c["model"] == k) & (c["condition"] == x)]["AP"].mean() for x in order]
            xs = np.arange(len(order)) + (i - (len(models) - 1) / 2) * w
            bars = ax.bar(xs, vals, w * 0.92, color=colors[k], edgecolor="black", label=WORLD[k]["label"])
            ax.bar_label(bars, fmt="%.1f", fontsize=19, padding=2)
        ax.set_xticks(np.arange(len(order)), ["80 + blank" if x == "80+blank" else x for x in order])
        ax.set_ylim(0, ax.get_ylim()[1] * 1.28)
        ax.set_xlabel("Vocabulary size (80 COCO + extra LVIS names)")
        ax.set_ylabel("AP on the 80 COCO classes (%)")
        ax.set_title("Bigger vocabulary, same 80 classes scored")
        ax.legend(loc="upper center", ncol=2, columnspacing=1.0, handlelength=1.2)
        fig.tight_layout()
        fig.savefig(RESULTS / "prompt_study_vocab.png", dpi=200)
        log("[plot] wrote results/prompt_study_vocab.png")

    b = df[df["part"] == "B"]
    if len(b):
        fig, ax = plt.subplots(figsize=(9, 6.6))
        for k in models:
            d = b[b["model"] == k].sort_values("vocab_size")
            if len(d):
                ax.plot(d["vocab_size"], d["fps"], marker="o", markersize=10, linewidth=2.5, color=colors[k],
                        label=WORLD[k]["label"])
                for x, y in zip(d["vocab_size"], d["fps"]):
                    ax.annotate(f"{y:.0f}", (x, y), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=19)
        ax.set_xscale("log")
        ax.set_xticks(sorted(set(b["vocab_size"])), [str(v) for v in sorted(set(b["vocab_size"]))])
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
        gpu = next((g for g in b["gpu"].dropna()), "GPU")
        ax.set_xlabel(f"Vocabulary size (classes, log scale)\n{gpu}, batch 1")
        ax.set_ylabel("FPS")
        ax.set_title("Speed as the vocabulary grows")
        ax.legend(loc="lower left")
        fig.tight_layout()
        fig.savefig(RESULTS / "prompt_study_speed.png", dpi=200)
        log("[plot] wrote results/prompt_study_speed.png")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parts", nargs="+", default=["A", "C", "B"], choices=["A", "B", "C", "plot"])
    p.add_argument("--models", nargs="+", default=list(WORLD), choices=list(WORLD))
    p.add_argument("--n-images", type=int, default=500)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--fp32", action="store_true")
    args = p.parse_args()

    seed_everything()
    t_start = time.time()
    parts = [x for x in args.parts if x != "plot"]
    if not parts:
        plot(args.n_images)
        return
    from runners import UltralyticsRunner

    info = env_info()
    log(f"env: {info}")
    coco = load_coco()
    ids = subset_ids(coco, args.n_images)
    cat_ids, names = coco_names(coco)
    rgb = load_images_rgb(ensure_images(coco, ids))
    bgr = [np.ascontiguousarray(im[..., ::-1]) for im in rgb]
    failed = False
    for key in args.models:
        base = {"model": key, "label": WORLD[key]["label"], "n_images": len(ids),
                "gpu": info["gpu"] if is_cuda(args.device) else "CPU",
                "precision": "fp32" if args.fp32 else "fp16", "timestamp": time.strftime("%Y-%m-%d %H:%M")}
        r = UltralyticsRunner(WORLD[key]["ckpt"], "world", device=args.device, half=not args.fp32)
        for part in parts:
            try:
                if part == "A":
                    part_a(key, r, args, coco, ids, cat_ids, names, bgr, base)
                elif part == "C":
                    part_c(key, r, args, coco, ids, cat_ids, names, bgr, base)
                else:
                    part_b(key, r, args, ids, names, bgr, base)
            except Exception:  # keep going with the next part / model
                import traceback

                traceback.print_exc()
                log(f"FAILED: part {part} for {key}")
                failed = True
        del r
    plot(len(ids))
    log(f"done in {time.time() - t_start:.0f}s")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
