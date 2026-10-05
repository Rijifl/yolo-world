"""Sanity check of the evaluation pipeline with YOLOv8-S (official COCO val2017 mAP50-95 = 44.9).

Compares, on the same image subset and scored by the same pycocotools code:
  (1) the predict() pipeline used in compare.py (single label per box NMS, conf 0.001, max 100 det)
  (2) Ultralytics model.val() (multi-label NMS, conf 0.001, max_det 300, rect batches), predictions exported
      as COCO JSON and re-scored with pycocotools (+ Ultralytics' own internal mAP for reference).
Also done for YOLO-World-S v2 (Ultralytics docs: 37.7 zero-shot COCO mAP).

Writes results/sanity_check.csv.
  python -u experiments/sanity_check.py --n-images 500 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time

import pandas as pd

from common import (DATA, RESULTS, coco_eval, coco_names, ensure_images, load_coco, log, seed_everything,
                    subset_ids)


def build_yolo_dataset(coco, ids, cat_ids, names, paths):
    """YOLO-format copy of the subset. Val list is named '.../coco.../val2017.txt' so Ultralytics treats it as COCO
    (coco80->91 class map, image_id = int(file stem)). iscrowd boxes are dropped (as the Ultralytics converter does);
    pycocotools re-scoring uses the original annotations, crowd regions included."""
    root = DATA / "coco_subset_yolo"
    if root.exists():
        shutil.rmtree(root)
    (root / "images" / "val2017").mkdir(parents=True)
    (root / "labels" / "val2017").mkdir(parents=True)
    idx = {c: i for i, c in enumerate(cat_ids)}
    lines = []
    for i, p in zip(ids, paths):
        img = coco.imgs[i]
        W, H = img["width"], img["height"]
        dst = root / "images" / "val2017" / p.name
        shutil.copy(p, dst)
        lab = []
        for a in coco.loadAnns(coco.getAnnIds(imgIds=[i], iscrowd=False)):
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            lab.append(f"{idx[a['category_id']]} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
        (root / "labels" / "val2017" / (p.stem + ".txt")).write_text("\n".join(lab))
        lines.append(str(dst.resolve()))
    (root / "val2017.txt").write_text("\n".join(lines))
    yaml = root / "coco_subset.yaml"
    yaml.write_text(f"path: {root.resolve().as_posix()}\ntrain: val2017.txt\nval: val2017.txt\nnames:\n" +
                    "".join(f"  {i}: {n}\n" for i, n in enumerate(names)))
    return yaml


def val_json_ap(weight, yaml, coco, ids, args, classes=None):
    from ultralytics import YOLO

    m = YOLO(weight)
    if classes is not None:
        m.set_classes(classes)
    proj = DATA / "ultra_runs"
    name = f"val_{os.path.basename(weight).split('.')[0]}"
    res = m.val(data=str(yaml), imgsz=640, batch=1, conf=0.001, iou=0.7, max_det=300, quantize=16,
                device=args.device, save_json=True, project=str(proj), name=name, exist_ok=True, plots=False,
                verbose=False, workers=0)
    pred = json.loads((proj / name / "predictions.json").read_text())
    # keep top-100 per image so pycocotools maxDets=100 sees the same budget as my pipeline
    by_img = {}
    for d in pred:
        by_img.setdefault(d["image_id"], []).append(d)
    pred100 = [d for v in by_img.values() for d in sorted(v, key=lambda d: -d["score"])[:100]]
    return coco_eval(coco, pred100, ids), round(float(res.box.map) * 100, 2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-images", type=int, default=500)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    seed_everything()
    t0 = time.time()
    coco = load_coco()
    ids = subset_ids(coco, args.n_images)
    cat_ids, names = coco_names(coco)
    paths = ensure_images(coco, ids)
    yaml = build_yolo_dataset(coco, ids, cat_ids, names, paths)
    rows = []

    # (1) my predict()-path detections, produced by compare.py --phase ap
    for key in ["yolov8s", "yolov8s-worldv2"]:
        f = DATA / "dets" / f"compare_{key}_n{len(ids)}.json"
        if f.exists():
            ap = coco_eval(coco, json.loads(f.read_text()), ids)
            rows.append({"model": key, "pipeline": "compare.py predict() path (single-label NMS, max_det 100)",
                         "n_images": len(ids), **ap, "ultralytics_internal_mAP": None})
    # (2) model.val()
    for key, w, cls in [("yolov8s", "weights/yolov8s.pt", None), ("yolov8s-worldv2", "weights/yolov8s-worldv2.pt", names)]:
        ap, internal = val_json_ap(w, yaml, coco, ids, args, cls)
        rows.append({"model": key, "pipeline": "Ultralytics model.val() (multi-label NMS, max_det 300 -> top-100 per image)",
                     "n_images": len(ids), **ap, "ultralytics_internal_mAP": internal})
        log(f"{key} val(): pycocotools AP={ap['AP']} internal={internal}")
    ref = {"yolov8s": "44.9 (Ultralytics docs, full val2017, 5000 imgs)",
           "yolov8s-worldv2": "37.7 (Ultralytics docs, zero-shot COCO val2017)"}
    df = pd.DataFrame(rows)
    df["official_reference_AP"] = df["model"].map(ref)
    df.to_csv(RESULTS / "sanity_check.csv", index=False)
    print(df.to_string())
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
