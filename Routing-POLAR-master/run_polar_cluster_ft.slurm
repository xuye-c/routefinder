#!/bin/bash
#SBATCH --job-name=polar_cft
#SBATCH --partition=common-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=polar_cft_%j.out
#SBATCH --error=polar_cft_%j.err
#
# Knobs live in cluster_ft_hparams.sh (edit that file on the cluster).
# Submit:
#   bash submit_cluster_ft.sh 50
#   N_SIZE=50 CLUSTER=0 sbatch --export=ALL,N_SIZE=50,CLUSTER=0 run_polar_cluster_ft.slurm
# Override one knob without editing the file:
#   LR=1e-5 FT_EPOCHS=2 sbatch --export=ALL,N_SIZE=50,CLUSTER=0,LR,FT_EPOCHS run_polar_cluster_ft.slurm

source /opt/apps/centos8/anaconda3/etc/profile.d/conda.sh
conda activate polar

cd ~/routefinder/Routing-POLAR-master

# shellcheck source=cluster_ft_hparams.sh
source ./cluster_ft_hparams.sh

if [ "${N_SIZE}" = "50" ]; then
  PATH_ID="${PATH_ID:-n=50/2026-0728-0719}"
  BATCH_SIZE="${BATCH_SIZE:-128}"
else
  PATH_ID="${PATH_ID:-n=100/2026-0729-1221}"
  BATCH_SIZE="${BATCH_SIZE:-64}"
fi

CSV_NAME="umap_cluster_polar_encf_${N_SIZE}.csv"
if [ -z "${CLUSTER_CSV}" ]; then
  for cand in \
    "${HOME}/routefinder/${CSV_NAME}" \
    "${HOME}/wfl_sw/${CSV_NAME}" \
    "${PWD}/${CSV_NAME}" \
    "${PWD}/../${CSV_NAME}" \
    "${PWD}/../../${CSV_NAME}"
  do
    if [ -f "${cand}" ]; then
      CLUSTER_CSV="${cand}"
      break
    fi
  done
fi

if [ -z "${CLUSTER_CSV}" ] || [ ! -f "${CLUSTER_CSV}" ]; then
  echo "ERROR: cluster csv not found (${CSV_NAME})"
  echo "Copy it to ~/routefinder/ or set CLUSTER_CSV in cluster_ft_hparams.sh"
  exit 1
fi

echo "===== Environment ====="
echo "Host: $(hostname)"
echo "Job: ${SLURM_JOB_ID}"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "Conda env: ${CONDA_DEFAULT_ENV}"

echo "===== Hparams ====="
echo "N_SIZE=${N_SIZE} CLUSTER=${CLUSTER} PATH_ID=${PATH_ID} EPOCH=${EPOCH}"
echo "CLUSTER_CSV=${CLUSTER_CSV}"
echo "DATA_DIR=${DATA_DIR}"
echo "FT_EPOCHS=${FT_EPOCHS} HOLD_FRAC=${HOLD_FRAC} SEED=${SEED}"
echo "BATCH_SIZE=${BATCH_SIZE} EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
echo "LR=${LR} WEIGHT_DECAY=${WEIGHT_DECAY} LR_GAMMA=${LR_GAMMA} LR_DECAY_EPOCH=${LR_DECAY_EPOCH} GRAD_CLIP=${GRAD_CLIP}"
echo "LOSS=${LOSS} PO_ALPHA=${PO_ALPHA}"
echo "EVAL_ONLY=${EVAL_ONLY} SKIP_ZERO_SHOT=${SKIP_ZERO_SHOT}"

echo "===== GPU ====="
nvidia-smi

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
CKPT="result/${PATH_ID}/checkpoint-${EPOCH}.pt"
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
problems = [
    "cvrp","ovrp","vrpb","vrpl","vrptw","ovrptw","ovrpb","ovrpl",
    "ovrpbl","ovrpbtw","ovrpltw","ovrpbltw","vrpbl","vrpbtw","vrpltw","vrpbltw",
]
missing = []
for p in problems:
    inst = f"{data_dir}/{p}/test/{n}.npz"
    sol = f"{data_dir}/{p}/test/{n}_sol_pyvrp.npz"
    if not os.path.isfile(inst):
        missing.append(inst)
    if not os.path.isfile(sol):
        missing.append(sol)
if missing:
    print("MISSING files:")
    for m in missing:
        print(" ", m)
    raise SystemExit(1)
print("Found all 16 test npz and _sol_pyvrp.npz for n=", n)
PY

echo "===== POLAR import test (no pyvrp) ====="
python -c "from models.model import VRPModel; from envs.mtvrp.env import MTVRPEnv; print('POLAR model/env import OK')"

EXTRA_FLAGS=()
if [ "${EVAL_ONLY}" = "1" ]; then
  EXTRA_FLAGS+=(--eval_only)
fi
if [ "${SKIP_ZERO_SHOT}" = "1" ]; then
  EXTRA_FLAGS+=(--skip_zero_shot)
fi

echo "===== Start cluster fine-tune ====="
python finetune_cluster.py \
    --n_size "${N_SIZE}" \
    --cluster "${CLUSTER}" \
    --cluster_csv "${CLUSTER_CSV}" \
    --epoch "${EPOCH}" \
    --path_id "${PATH_ID}" \
    --data_dir "${DATA_DIR}" \
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
    "${EXTRA_FLAGS[@]}"

echo "===== Done ====="
