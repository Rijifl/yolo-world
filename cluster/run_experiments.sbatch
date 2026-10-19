#!/bin/bash
#SBATCH --job-name=yoloworld-exp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=cluster/logs/job-%j.out
#
# All experiments in ONE job on ONE GPU, so every number comes from the same hardware.
# Submit from the repo root and request the GPU on the command line, for example:
#     sbatch --gpus=a100:1 cluster/run_experiments.sbatch
# Optional: N_IMAGES=50 sbatch ... for a quick trial run (the real run uses the default 500).

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PY=.venv/bin/python
N=${N_IMAGES:-500}
L=cluster/logs
mkdir -p "$L" results
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
FAIL=()
note() { echo; echo "===== $(date +%H:%M:%S) $* ====="; }
run() { # run <name> <command...>: log to $L/<name>.log, remember failures, keep going
  local name=$1; shift
  if "$@" >"$L/$name.log" 2>&1; then echo "ok    $name"; else echo "FAIL  $name (exit $?) - see $L/$name.log"; FAIL+=("$name"); fi
}

note "job ${SLURM_JOB_ID:-local} on $(hostname), $N images"
nvidia-smi

# Start from empty result files, so no row from an earlier run (or a trial run with fewer images) survives.
# Keeps the fixed image subset and results/demo/.
rm -f results/compare.csv results/compare_meta.json results/compare.png results/compare.svg       results/prompt_study*.csv results/prompt_study_*.png results/sanity_check.csv       results/summary.json results/README_results.md results/.lock

# ---- 1. accuracy (AP). Independent of timing, so the models run as parallel processes on the one GPU.
note "phase 1: AP passes (4 parallel processes)"
run ap_yolov8   $PY -u experiments/compare.py --phase ap --n-images "$N" \
                  --models yolov8s yolov8m yolov8l yolov8s-worldv2 &
run ap_open     $PY -u experiments/compare.py --phase ap --n-images "$N" \
                  --models yolov8m-worldv2 yolov8l-worldv2 yoloe-v8s yoloe-11s yoloe-26s &
run ap_hf       $PY -u experiments/compare.py --phase ap --n-images "$N" \
                  --models gdino-t owlv2-b16 &
run prompts_ac  $PY -u experiments/prompt_study.py --parts A C --n-images "$N" &
wait
# `run` inside a background subshell cannot update FAIL in this shell, so collect failures from the logs
for n in ap_yolov8 ap_open ap_hf prompts_ac; do
  if ! tail -n 3 "$L/$n.log" | grep -q "done in" || grep -q "FAILED:" "$L/$n.log"; then FAIL+=("$n"); fi
done

# ---- 2. sanity check of the evaluation pipeline (needs the detections written in phase 1)
note "phase 2: sanity check"
run sanity $PY -u experiments/sanity_check.py --n-images "$N"

# ---- 3. timing. Strictly one process at a time, nothing else on the GPU.
note "phase 3: timing (sequential, GPU otherwise idle)"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
run timing    $PY -u experiments/compare.py --phase timing --n-images "$N"
run prompts_b $PY -u experiments/prompt_study.py --parts B --n-images "$N"

# ---- 4. report + bundle
note "phase 4: report"
run report $PY experiments/make_report.py --n-images "$N"
mkdir -p data/dets
# the job's own log is still being written, so tar may warn "file changed as we read it"; that is harmless
tar -czf results_bundle.tar.gz --exclude=results/demo results data/dets cluster/logs || true
ls -la results results_bundle.tar.gz

note "finished. failures: ${FAIL[*]:-none}"
[ ${#FAIL[@]} -eq 0 ]
