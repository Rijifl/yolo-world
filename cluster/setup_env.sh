#!/bin/bash
# One-time setup on the cluster LOGIN node (needs internet): Python 3.12 venv, packages, all model weights.
#   bash cluster/setup_env.sh
# Safe to re-run: finished steps are skipped or verified.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "== repo: $(pwd)"
mkdir -p cluster/logs results

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "== installing uv (fast Python/package installer) into ~/.local/bin"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

if [ ! -x .venv/bin/python ]; then
  echo "== creating .venv (Python 3.12)"
  uv venv --python 3.12 --seed .venv
fi
PY=.venv/bin/python

echo "== installing PyTorch (CUDA 12.4 wheels)"
uv pip install --python "$PY" torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 \
  || "$PY" -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

echo "== installing requirements.txt"
# unsafe-best-match: take each pinned version from whichever index has it (PyPI or the PyTorch index)
uv pip install --python "$PY" --index-strategy unsafe-best-match -r requirements.txt \
  || "$PY" -m pip install -r requirements.txt

echo "== import check"
"$PY" - <<'EOF'
import cv2, torch, ultralytics, transformers, pycocotools, clip  # noqa: F401
print("torch", torch.__version__, "| ultralytics", ultralytics.__version__, "| transformers", transformers.__version__)
EOF

echo "== downloading model weights (about 3 GB)"
HF_HUB_OFFLINE=0 "$PY" scripts/download_models.py --experiments --no-test

echo "== downloading COCO val2017 subset + LVIS categories"
"$PY" scripts/download_data.py

echo "== data check"
"$PY" - <<'EOF'
import json, pathlib
ids = json.load(open("results/coco_subset_ids.json"))["image_ids"]
have = len(list(pathlib.Path("data/coco/val2017").glob("*.jpg")))
assert pathlib.Path("data/coco/instances_val2017.json").exists(), "missing data/coco/instances_val2017.json"
assert pathlib.Path("data/lvis/lvis_v1_categories.json").exists(), "missing data/lvis/lvis_v1_categories.json"
assert have >= len(ids), f"only {have} of {len(ids)} subset images present"
print(f"subset images: {have}/{len(ids)} OK")
EOF
echo "== setup complete"
