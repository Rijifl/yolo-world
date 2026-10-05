"""Download and cache the model files the demo needs, so it runs with Wi-Fi off.

Usage (from the repo root):
    python scripts/download_models.py                 # S/M/L YOLO-World + S/M/L YOLOv8 + CLIP text encoder
    python scripts/download_models.py --small         # only the S models + CLIP (~400 MB)
    python scripts/download_models.py --experiments   # also YOLOE, Grounding DINO-T and OWLv2 for experiments/

Everything goes into ./weights/ (git-ignored). Safe to re-run: files that are already there are skipped
(CLIP is checked by SHA-256). At the end it loads YOLO-World S once to check that everything is there.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WEIGHTS = REPO / "weights"
CLIP_DIR = WEIGHTS / "clip"

SIZES_ALL = ["s", "m", "l"]


def human(n: float) -> str:
    return f"{n / 1e6:.0f} MB"


def fetch_yolo(name: str) -> None:
    target = WEIGHTS / name
    if target.exists() and target.stat().st_size > 1e6:
        print(f"  [skip]       {target.relative_to(REPO)}  ({human(target.stat().st_size)}, already cached)")
        return
    from ultralytics.utils.downloads import attempt_download_asset

    t0 = time.time()
    attempt_download_asset(str(target))
    if not target.exists():
        raise RuntimeError(f"download of {name} failed (check your internet connection)")
    print(f"  [downloaded] {target.relative_to(REPO)}  ({human(target.stat().st_size)}, {time.time() - t0:.0f}s)")


def fetch_clip() -> None:
    # CLIP ViT-B/32 goes to weights/clip/, which is where Ultralytics' YOLO-World looks for it
    # (ultralytics.nn.text_model.CLIP calls clip.load("ViT-B/32", download_root=WEIGHTS_DIR / "clip")).
    # clip._download() re-uses an existing file if its SHA-256 matches, otherwise it downloads it.
    import clip

    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    target = CLIP_DIR / "ViT-B-32.pt"
    existed = target.exists()
    t0 = time.time()
    path = clip.clip._download(clip.clip._MODELS["ViT-B/32"], str(CLIP_DIR))  # verifies SHA-256
    tag = "[verified]  " if existed else "[downloaded]"
    print(f"  {tag} {Path(path).relative_to(REPO)}  ({human(Path(path).stat().st_size)}, {time.time() - t0:.0f}s)")


HF_MODELS = ["IDEA-Research/grounding-dino-tiny", "google/owlv2-base-patch16-ensemble"]
EXPERIMENT_ASSETS = ["yoloe-v8s-seg.pt", "yoloe-11s-seg.pt", "yoloe-26s-seg.pt", "mobileclip_blt.ts", "mobileclip2_b.ts"]


def fetch_experiment_models() -> None:
    # extra models only used by experiments/: YOLOE (+ its MobileCLIP text encoders), Grounding DINO-T, OWLv2
    print("YOLOE and its MobileCLIP text encoders:")
    for name in EXPERIMENT_ASSETS:
        fetch_yolo(name)
    print("Hugging Face checkpoints (stored in the Hugging Face cache, not in weights/):")
    os.environ["HF_HUB_OFFLINE"] = "0"
    from huggingface_hub import snapshot_download

    for repo in HF_MODELS:
        t0 = time.time()
        path = snapshot_download(repo)
        print(f"  [cached]     {repo} -> {path}  ({time.time() - t0:.0f}s)")


def quick_test() -> None:
    # load YOLO-World S from weights/ and run set_classes + predict once
    import numpy as np
    import torch
    from ultralytics import YOLOWorld

    import ultralytics.nn.text_model as tm

    tm.WEIGHTS_DIR = WEIGHTS
    device = 0 if torch.cuda.is_available() else "cpu"
    m = YOLOWorld(str(WEIGHTS / "yolov8s-worldv2.pt"))
    m.to("cuda" if device == 0 else "cpu")
    m.set_classes(["person", "guitar"])
    m.predict(np.zeros((320, 320, 3), dtype=np.uint8), device=device, verbose=False)
    print(f"  OK on {'GPU: ' + torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--small", action="store_true", help="only download the S-size models (plus CLIP)")
    ap.add_argument("--no-test", action="store_true", help="skip the test at the end")
    ap.add_argument("--experiments", action="store_true",
                    help="also cache the models used only by experiments/ (YOLOE, Grounding DINO-T, OWLv2)")
    args = ap.parse_args()

    os.chdir(REPO)
    WEIGHTS.mkdir(exist_ok=True)
    sizes = ["s"] if args.small else SIZES_ALL

    print(f"Caching demo models into {WEIGHTS}")
    print("YOLO-World v2 (open vocabulary):")
    for s in sizes:
        fetch_yolo(f"yolov8{s}-worldv2.pt")
    print("YOLOv8 (closed set, 80 COCO classes):")
    for s in sizes:
        fetch_yolo(f"yolov8{s}.pt")
    print("CLIP ViT-B/32 text encoder (used by YOLO-World's set_classes):")
    fetch_clip()
    if args.experiments:
        fetch_experiment_models()
    if not args.no_test:
        print("Quick test:")
        quick_test()
    print("Done. The demo now works without internet: python demo/app.py")


if __name__ == "__main__":
    main()
