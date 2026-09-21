#!/bin/bash
#SBATCH --job-name=cada_fam_ft
#SBATCH --partition=common-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=cada_family_ft_%j.out
#SBATCH --error=cada_family_ft_%j.err
#
# One job = all 6 family specialists, sequential. The .out file has every
# family banner / epoch / holdout line. Checkpoints go under
#   CADA/{n}/result/family-ft-n{n}-<stamp>/family{k}-<name>/
#
# Submit:
#   bash submit_cada_family_ft.sh 50
#   N_SIZE=100 sbatch --export=ALL,N_SIZE=100 run_cada_family_ft.slurm

set -euo pipefail
export PYTHONUNBUFFERED=1

source /opt/apps/centos8/anaconda3/etc/profile.d/conda.sh

CADA_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=cada_family_ft_hparams.sh
source "${CADA_DIR}/cada_family_ft_hparams.sh"

conda activate "${CONDA_ENV}"

RF_ROOT="$(cd "${CADA_DIR}/.." && pwd)"
cd "${RF_ROOT}"

if [ "${N_SIZE}" = "50" ]; then
  PATH_ID="${PATH_ID:-2024-1111-1139}"
  BATCH_SIZE="${BATCH_SIZE:-128}"
else
  PATH_ID="${PATH_ID:-2024-1121-1355}"
  BATCH_SIZE="${BATCH_SIZE:-64}"
fi

CSV_NAME="constraint_family_cluster_6way_${N_SIZE}.csv"
if [ -z "${CLUSTER_CSV}" ]; then
  for cand in \
    "${RF_ROOT}/${CSV_NAME}" \
    "${RF_ROOT}/../${CSV_NAME}" \
    "${HOME}/routefinder/${CSV_NAME}" \
    "${HOME}/wfl_sw/${CSV_NAME}" \
    "${PWD}/${CSV_NAME}"
  do
    if [ -f "${cand}" ]; then
      CLUSTER_CSV="${cand}"
      break
    fi
  done
fi

echo "===== Environment ====="
echo "Host: $(hostname)"
echo "Job: ${SLURM_JOB_ID:-local}"
echo "CWD: $(pwd)"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "Conda env: ${CONDA_DEFAULT_ENV}"

echo "===== Hparams ====="
echo "N_SIZE=${N_SIZE} CLUSTERS=${CLUSTERS} PATH_ID=${PATH_ID} EPOCH=${EPOCH}"
echo "CLUSTER_CSV=${CLUSTER_CSV:-<type mapping if missing>}"
echo "DATA_DIR=${DATA_DIR:-<auto>}"
echo "FT_EPOCHS=${FT_EPOCHS} HOLD_FRAC=${HOLD_FRAC} SEED=${SEED}"
echo "BATCH_SIZE=${BATCH_SIZE} EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
echo "LR=${LR} WEIGHT_DECAY=${WEIGHT_DECAY} LR_GAMMA=${LR_GAMMA} LR_DECAY_EPOCH=${LR_DECAY_EPOCH} GRAD_CLIP=${GRAD_CLIP}"
echo "LOSS=${LOSS} PO_ALPHA=${PO_ALPHA}"
echo "EVAL_ONLY=${EVAL_ONLY} SKIP_ZERO_SHOT=${SKIP_ZERO_SHOT}"

echo "===== GPU ====="
nvidia-smi || true

echo "===== PyTorch ====="
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

echo "===== Checkpoint ====="
CKPT="CADA/${N_SIZE}/result/${PATH_ID}/checkpoint-${EPOCH}.pt"
ls -lh "${CKPT}"
python - <<PY
from pathlib import Path
p = Path("${CKPT}")
head = p.read_bytes()[:40]
print("checkpoint head:", head[:40])
if head.startswith(b"version https://git-lfs.github.com"):
    raise SystemExit("ERROR: checkpoint is a Git LFS pointer, not a real file")
print("checkpoint looks like a real binary")
PY

echo "===== Data ====="
python - <<PY
import os
n = int("${N_SIZE}")
data_dir = "${DATA_DIR}"
candidates = [
    data_dir,
    "data",
    "Routing-POLAR-master/data",
    os.path.expanduser("~/routefinder/data"),
    os.path.expanduser("~/routefinder/Routing-POLAR-master/data"),
]
root = None
for c in candidates:
    if c and os.path.isdir(os.path.join(c, "cvrp", "test")):
        root = c
        break
if root is None:
    raise SystemExit("ERROR: could not find data/{type}/test")
print("data_dir =", os.path.abspath(root))
problems = [
    "cvrp","ovrp","vrpb","vrpl","vrptw","ovrptw","ovrpb","ovrpl",
    "ovrpbl","ovrpbtw","ovrpltw","ovrpbltw","vrpbl","vrpbtw","vrpltw","vrpbltw",
]
missing = []
for p in problems:
    inst = f"{root}/{p}/test/{n}.npz"
    if not os.path.isfile(inst):
        missing.append(inst)
if missing:
    print("MISSING files:")
    for m in missing:
        print(" ", m)
    raise SystemExit(1)
print("Found all 16 test npz for n=", n)
PY

echo "===== CADA import test ====="
python - <<PY
import sys
from pathlib import Path
n = "${N_SIZE}"
root = Path(".").resolve()
sys.path.insert(0, str(root / "CADA" / n))
sys.path.insert(0, str(root / "CADA"))
from model import VRPModel
from envs.env import MTVRPEnv
print("CADA model/env import OK for n=", n)
PY

EXTRA_FLAGS=()
if [ "${EVAL_ONLY}" = "1" ]; then
  EXTRA_FLAGS+=(--eval_only)
fi
if [ "${SKIP_ZERO_SHOT}" = "1" ]; then
  EXTRA_FLAGS+=(--skip_zero_shot)
fi

CSV_FLAG=()
if [ -n "${CLUSTER_CSV}" ] && [ -f "${CLUSTER_CSV}" ]; then
  CSV_FLAG+=(--cluster_csv "${CLUSTER_CSV}")
else
  echo "NOTE: cluster csv not found; using built-in 6-family type mapping"
fi

DATA_FLAG=()
if [ -n "${DATA_DIR}" ]; then
  DATA_FLAG+=(--data_dir "${DATA_DIR}")
fi

echo "===== Start CADA family fine-tune (all clusters in this job) ====="
python -u finetune_cada_family.py \
    --n_size "${N_SIZE}" \
    --clusters "${CLUSTERS}" \
    --epoch "${EPOCH}" \
    --path_id "${PATH_ID}" \
    --batch_size "${BATCH_SIZE}" \
    --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --epochs "${FT_EPOCHS}" \
    --holdout_frac "${HOLD_FRAC}" \
    --lr "${LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --lr_gamma "${LR_GAMMA}" \
    --lr_decay_epoch "${LR_DECAY_EPOCH}" \
    --grad_clip "${GRAD_CLIP}" \
    --seed "${SEED}" \
    --loss "${LOSS}" \
    --po_alpha "${PO_ALPHA}" \
    "${CSV_FLAG[@]}" \
    "${DATA_FLAG[@]}" \
    "${EXTRA_FLAGS[@]}"

echo "===== Done ====="
echo "Slurm stdout: cada_family_ft_${SLURM_JOB_ID:-local}.out"
echo "Also copied into CADA/${N_SIZE}/result/family-ft-n${N_SIZE}-*/run.log"
echo "Per-family ckpts: CADA/${N_SIZE}/result/family-ft-n${N_SIZE}-*/family{0-5}-*/tuned-family*-*.pt"
