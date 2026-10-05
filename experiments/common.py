"""Shared helpers for the experiment scripts: repo paths, the fixed-seed COCO val2017 subset (500 images,
seed=0), image download, pycocotools bbox evaluation, and environment info.
"""
from __future__ import annotations

import json
import os
import platform
import random
import re
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
COCO_DIR = DATA / "coco"
IMG_DIR = COCO_DIR / "val2017"
ANN_FILE = COCO_DIR / "instances_val2017.json"
ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMG_URL = "http://images.cocodataset.org/val2017/{}"
LVIS_ZIP = DATA / "lvis" / "lvis_v1_val.json.zip"
LVIS_URL = "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip"
RESULTS = ROOT / "results"
SUBSET_FILE = RESULTS / "coco_subset_ids.json"
SEED = 0
N_SUBSET = 500

os.chdir(ROOT)  # checkpoint paths like weights/yolov8s.pt are relative to the repo root
RESULTS.mkdir(exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def seed_everything(seed: int = SEED) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# data loading
def ensure_annotations() -> None:
    if ANN_FILE.exists():
        return
    COCO_DIR.mkdir(parents=True, exist_ok=True)
    z = COCO_DIR / "annotations_trainval2017.zip"
    if not z.exists():
        log(f"downloading {ANN_URL}")
        urllib.request.urlretrieve(ANN_URL, z)
    with zipfile.ZipFile(z) as f:
        ANN_FILE.write_bytes(f.read("annotations/instances_val2017.json"))


def load_coco():
    from pycocotools.coco import COCO

    ensure_annotations()
    return COCO(str(ANN_FILE))


def subset_ids(coco, n: int | None = None) -> list[int]:
    """Fixed 500-image subset: seed=0 sample of val2017 images with >=1 annotation (iscrowd included,
    as in the standard 'has annotations' filter). The list is written once to results/coco_subset_ids.json.
    If n < 500 the FIRST n ids of that saved list are used (so smaller runs are nested subsets)."""
    if SUBSET_FILE.exists():
        ids = json.loads(SUBSET_FILE.read_text())["image_ids"]
    else:
        cands = sorted(i for i in coco.getImgIds() if len(coco.getAnnIds(imgIds=[i])) > 0)
        rng = random.Random(SEED)
        ids = rng.sample(cands, N_SUBSET)
        SUBSET_FILE.write_text(json.dumps({
            "description": "COCO val2017 subset: random.Random(0).sample(sorted ids of val2017 images with >=1 "
                           "annotation in instances_val2017.json, 500). Order = sampling order; runs with "
                           "--n-images N use the first N ids.",
            "seed": SEED, "n_candidates": len(cands), "image_ids": ids}, indent=1))
    return ids if n is None else ids[:n]


def ensure_images(coco, ids: list[int]) -> list[Path]:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in ids:
        fn = coco.imgs[i]["file_name"]
        p = IMG_DIR / fn
        if not p.exists():
            for attempt in range(5):
                try:
                    urllib.request.urlretrieve(IMG_URL.format(fn), p)
                    break
                except Exception as e:
                    log(f"retry {fn}: {e}")
                    time.sleep(2)
        paths.append(p)
    return paths


def load_images_rgb(paths: list[Path]) -> list[np.ndarray]:
    # decode everything into memory up front so timing excludes disk I/O
    from PIL import Image

    return [np.asarray(Image.open(p).convert("RGB")) for p in paths]


def coco_names(coco) -> tuple[list[int], list[str]]:
    cat_ids = sorted(coco.getCatIds())
    return cat_ids, [coco.cats[c]["name"] for c in cat_ids]


def lvis_categories() -> list[dict]:
    """LVIS v1 categories (id order) from the official lvis_v1_val.json (source: LVIS_URL)."""
    cache = DATA / "lvis" / "lvis_v1_categories.json"
    if cache.exists():
        return json.loads(cache.read_text())
    LVIS_ZIP.parent.mkdir(parents=True, exist_ok=True)
    if not LVIS_ZIP.exists():
        log(f"downloading {LVIS_URL}")
        urllib.request.urlretrieve(LVIS_URL, LVIS_ZIP)
    with zipfile.ZipFile(LVIS_ZIP) as f:
        cats = json.loads(f.read("lvis_v1_val.json"))["categories"]
    cats = sorted(({"id": c["id"], "name": c["name"], "synonyms": c["synonyms"]} for c in cats), key=lambda c: c["id"])
    cache.write_text(json.dumps(cats))
    return cats


def clean_lvis_names(exclude: list[str]) -> list[str]:
    """LVIS v1 category names as plain prompts: '_(...)' qualifiers and underscores removed, lower-cased,
    de-duplicated, and without any name in `exclude` (the COCO names). Order = LVIS id order."""
    seen = {n.lower().strip() for n in exclude}
    out = []
    for c in lvis_categories():
        n = re.sub(r"_?\(.*?\)", "", c["name"]).replace("_", " ").strip().lower()
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


# evaluation
AP_KEYS = ["AP", "AP50", "AP75", "APs", "APm", "APl"]


def _coco_eval_obj(coco, dets: list[dict], img_ids: list[int], cat_ids: list[int] | None = None):
    from pycocotools.cocoeval import COCOeval
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        dt = coco.loadRes(dets)
        E = COCOeval(coco, dt, "bbox")
        E.params.imgIds = list(img_ids)
        if cat_ids is not None:
            E.params.catIds = list(cat_ids)
        E.evaluate()
        E.accumulate()
        E.summarize()
    return E


def coco_eval(coco, dets: list[dict], img_ids: list[int], cat_ids: list[int] | None = None) -> dict:
    """Standard pycocotools bbox eval (maxDets=100) restricted to img_ids. Returns AP metrics in %."""
    if not dets:
        return {k: 0.0 for k in AP_KEYS}
    E = _coco_eval_obj(coco, dets, img_ids, cat_ids)
    return {k: round(float(v) * 100, 3) for k, v in zip(AP_KEYS, E.stats[:6])}


def coco_eval_per_class(coco, dets: list[dict], img_ids: list[int]) -> tuple[dict, dict[int, float | None]]:
    """coco_eval plus AP50:95 per category id (in %). A category with no ground truth in img_ids gets None."""
    cat_ids = sorted(coco.getCatIds())
    if not dets:
        return {k: 0.0 for k in AP_KEYS}, {c: None for c in cat_ids}
    E = _coco_eval_obj(coco, dets, img_ids)
    prec = E.eval["precision"]  # [T, R, K, A, M]; area index 0 = all, maxDets index -1 = 100
    per = {}
    for k, c in enumerate(E.params.catIds):
        p = prec[:, :, k, 0, -1]
        p = p[p > -1]
        per[int(c)] = round(float(p.mean()) * 100, 3) if p.size else None
    return {k: round(float(v) * 100, 3) for k, v in zip(AP_KEYS, E.stats[:6])}, per


def to_coco_dets(img_id: int, boxes_xyxy, scores, cat_ids) -> list[dict]:
    out = []
    for (x1, y1, x2, y2), s, c in zip(np.asarray(boxes_xyxy, dtype=float), np.asarray(scores, dtype=float), cat_ids):
        out.append({"image_id": int(img_id), "category_id": int(c),
                    "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                    "score": round(float(s), 5)})
    return out


# environment info
def is_cuda(device) -> bool:
    return str(device).startswith("cuda")


def env_info() -> dict:
    import torch
    import transformers
    import ultralytics

    info = {
        "python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "ultralytics": ultralytics.__version__,
        "transformers": transformers.__version__, "os": platform.platform(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
    }
    try:
        info["nvidia_driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                               capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    return info


class Timer:
    """Wall-clock timer in ms; calls torch.cuda.synchronize() on both sides."""

    def __init__(self, cuda: bool = True):
        import torch

        self.cuda = cuda and torch.cuda.is_available()
        self._torch = torch

    def __enter__(self):
        if self.cuda:
            self._torch.cuda.synchronize()
        self.t = time.perf_counter()
        return self

    def __exit__(self, *a):
        if self.cuda:
            self._torch.cuda.synchronize()
        self.ms = (time.perf_counter() - self.t) * 1000


def plot_style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 16, "axes.titlesize": 18, "axes.labelsize": 17, "xtick.labelsize": 15,
                         "ytick.labelsize": 15, "legend.fontsize": 14, "figure.dpi": 100,
                         "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False})
    return plt

