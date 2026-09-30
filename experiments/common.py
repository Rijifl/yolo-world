"""Shared helpers for the experiment scripts: repo paths, the fixed-seed COCO val2017 subset (500 images,
seed=0), image download, pycocotools bbox evaluation, GPU checks before timing, and environment info.
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
WEIGHTS = ROOT / "weights"
SUBSET_FILE = RESULTS / "coco_subset_ids.json"
SEED = 0
N_SUBSET = 500

os.chdir(ROOT)  # Ultralytics settings use weights_dir="weights" (relative) -> must run from repo root
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
    return {k: round(float(v) * 100, 2) for k, v in zip(AP_KEYS, E.stats[:6])}


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
        per[int(c)] = round(float(p.mean()) * 100, 2) if p.size else None
    return {k: round(float(v) * 100, 2) for k, v in zip(AP_KEYS, E.stats[:6])}, per


def to_coco_dets(img_id: int, boxes_xyxy, scores, cat_ids) -> list[dict]:
    out = []
    for (x1, y1, x2, y2), s, c in zip(np.asarray(boxes_xyxy, dtype=float), np.asarray(scores, dtype=float), cat_ids):
        out.append({"image_id": int(img_id), "category_id": int(c),
                    "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                    "score": round(float(s), 5)})
    return out


# GPU state and environment info
def other_gpu_python_procs() -> list[str]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception as e:
        return [f"nvidia-smi failed: {e}"]
    me = os.getpid()
    procs = []
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        pid, name = [s.strip() for s in line.split(",", 1)]
        if "python" in name.lower() and int(pid) != me:
            procs.append(line.strip())
    return procs


def gpu_apps() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=30).stdout.strip().replace("\n", " | ")
    except Exception as e:
        return f"nvidia-smi failed: {e}"


def wait_for_gpu(poll_s: int = 60, max_wait_s: int = 3 * 3600) -> float:
    """Block until no OTHER python process holds the GPU. Returns seconds waited."""
    t0 = time.time()
    while True:
        procs = other_gpu_python_procs()
        if not procs:
            waited = time.time() - t0
            log(f"GPU free of other python processes (waited {waited:.0f}s). GPU compute apps: {gpu_apps() or 'none'}")
            return waited
        if time.time() - t0 > max_wait_s:
            log(f"WARNING: gave up waiting for GPU after {max_wait_s}s; other procs: {procs}")
            return time.time() - t0
        log(f"GPU busy with other python process(es) {procs}; polling again in {poll_s}s")
        time.sleep(poll_s)


def gpu_clock_state() -> dict:
    """Instantaneous P-state / SM clock / throttle reasons (read while a GPU load is running)."""
    q = "pstate,clocks.sm,clocks.max.sm,power.draw,temperature.gpu,utilization.gpu,memory.used,clocks_throttle_reasons.active"
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=30).stdout.strip()
    except Exception as e:
        return {"error": str(e)}
    keys = ["pstate", "sm_mhz", "max_sm_mhz", "power_w", "temp_c", "util_pct", "mem_used", "throttle_reasons"]
    line = out.splitlines()[0] if out else ""  # inside a Slurm job only the allocated GPU is visible
    return dict(zip(keys, [s.strip() for s in line.split(",")]))


def probe_clock_under_load(device) -> dict:
    """Run ~3 s of matmuls and sample the clock mid-way; used to refuse timing while the GPU is throttled."""
    import threading

    import torch

    if not str(device).startswith("cuda"):
        return {}
    a = torch.randn(1024, 1024, device=device, dtype=torch.half)
    res = {}

    def sample():
        time.sleep(1.5)
        res.update(gpu_clock_state())

    th = threading.Thread(target=sample)
    th.start()
    t = time.time()
    while time.time() - t < 3:
        for _ in range(20):
            a @ a
        torch.cuda.synchronize()
    th.join()
    return res


def clock_throttled(clock: dict) -> bool:
    """True only when the SM clock under load is below half of its maximum (the stuck-laptop case).
    Unreadable values ('[N/A]', as some data-center GPUs report) count as not throttled."""
    try:
        sm = float(str(clock["sm_mhz"]).split()[0])
        mx = float(str(clock["max_sm_mhz"]).split()[0])
    except (KeyError, ValueError, IndexError):
        return False
    return mx > 0 and sm < 0.5 * mx


def is_cuda(device) -> bool:
    return str(device).startswith("cuda")


def power_state() -> str:
    if platform.system() != "Windows":
        return "unknown"
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance -Namespace root/wmi -ClassName BatteryStatus).PowerOnline;"
                              "(Get-CimInstance Win32_Battery).EstimatedChargeRemaining"],
                             capture_output=True, text=True, timeout=60).stdout.split()
        online = out[0] if out else "?"
        charge = out[1] if len(out) > 1 else "?"
        return f"AC power online={online}, battery={charge}%"
    except Exception as e:
        return f"unknown ({e})"


def env_info() -> dict:
    import torch
    import transformers
    import ultralytics

    info = {
        "python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "ultralytics": ultralytics.__version__,
        "transformers": transformers.__version__, "os": platform.platform(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "power": power_state(),
    }
    try:
        info["nvidia_driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version,power.limit,clocks.max.sm",
                                                "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    return info


class file_lock:
    """Cross-process lock via an exclusive lock file, so parallel runs can update the same CSV.
    A lock older than `stale_s` (left by a killed process) is removed."""

    def __init__(self, target: Path, stale_s: float = 120.0):
        self.path = Path(str(target) + ".lock")
        self.stale_s = stale_s

    def __enter__(self):
        while True:
            try:
                self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                return self
            except FileExistsError:
                try:
                    if time.time() - self.path.stat().st_mtime > self.stale_s:
                        self.path.unlink()
                        continue
                except FileNotFoundError:
                    continue
                time.sleep(0.2)

    def __exit__(self, *a):
        os.close(self.fd)
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


class Timer:
    """CUDA-synchronised wall-clock timer (ms)."""

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


# Okabe-Ito colour-blind-safe palette
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
