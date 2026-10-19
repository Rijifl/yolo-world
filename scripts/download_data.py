"""Download and cache the data the experiments need, so they run on machines without internet.

Usage (from the repo root):
    python scripts/download_data.py                   # COCO val2017 annotations + 500-image subset + LVIS categories
    python scripts/download_data.py --n-images 100    # only the first 100 subset images

Everything goes into ./data/ (git-ignored). Safe to re-run: files that are already there are skipped.
The subset ids come from results/coco_subset_ids.json (the same fixed seed=0 list the experiments use).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))

# common.py does os.chdir(REPO) on import, so all data paths below are relative to the repo root
from common import (ANN_FILE, IMG_DIR, N_SUBSET, ensure_annotations, ensure_images, load_coco,  # noqa: E402
                    lvis_categories, subset_ids)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-images", type=int, default=N_SUBSET, metavar="N",
                    help=f"number of subset images to download (first N of the fixed list, max {N_SUBSET})")
    args = ap.parse_args()
    if not 1 <= args.n_images <= N_SUBSET:
        ap.error(f"--n-images must be between 1 and {N_SUBSET}")

    print(f"Caching experiment data into {REPO / 'data'}")
    print("COCO val2017 annotations:")
    existed = ANN_FILE.exists()
    ensure_annotations()
    print(f"  {'[skip]      ' if existed else '[downloaded]'} {ANN_FILE.relative_to(REPO)}")

    print(f"COCO val2017 subset images (first {args.n_images} of the fixed list):")
    coco = load_coco()
    ids = subset_ids(coco, args.n_images)
    before = sum((IMG_DIR / coco.imgs[i]["file_name"]).exists() for i in ids)
    paths = ensure_images(coco, ids)  # retries each missing image a few times, does not raise
    missing = [p for p in paths if not p.exists()]
    print(f"  {before} already cached, {len(paths) - before - len(missing)} downloaded, {len(missing)} missing")
    if missing:
        raise SystemExit(f"{len(missing)} images failed to download (check your internet connection and re-run)")

    print("LVIS v1 category names (used by the prompt study):")
    cache = REPO / "data" / "lvis" / "lvis_v1_categories.json"
    existed = cache.exists()
    cats = lvis_categories()  # writes data/lvis/lvis_v1_categories.json if it is not there
    print(f"  {'[skip]   ' if existed else '[created]'} {cache.relative_to(REPO)}  ({len(cats)} categories)")
    print("Done. The experiments now work without internet: python experiments/compare.py")


if __name__ == "__main__":
    main()
