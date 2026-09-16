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
# One cluster per job. Submit with:
#   N_SIZE=50 CLUSTER=0 sbatch run_polar_cluster_ft.slurm
# or use submit_cluster_ft.sh
#
# Copy CSVs to the cluster first, e.g.:
#   scp umap_cluster_polar_encf_50.csv umap_cluster_polar_encf_100.csv \
#       xc224@dkucc-login-01:~/routefinder/

source /opt/apps/centos8/anaconda3/etc/profile.d/conda.sh
conda activate polar

cd ~/routefinder/Routing-POLAR-master

N_SIZE="${N_SIZE:-50}"
CLUSTER="${CLUSTER:-0}"
EPOCH="${EPOCH:-300}"
HOLD_FRAC="${HOLD_FRAC:-0.2}"
FT_EPOCHS="${FT_EPOCHS:-10}"
SEED="${SEED:-7}"
EVAL_ONLY="${EVAL_ONLY:-0}"

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
  echo "Copy it to ~/routefinder/ or pass CLUSTER_CSV=/path/to/${CSV_NAME}"
  exit 1
fi

echo "===== Environment ====="
echo "Host: $(hostname)"
echo "Job: ${SLURM_JOB_ID}"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "Conda env: ${CONDA_DEFAULT_ENV}"
echo "N_SIZE=${N_SIZE} CLUSTER=${CLUSTER} PATH_ID=${PATH_ID}"
echo "CLUSTER_CSV=${CLUSTER_CSV}"
echo "BATCH_SIZE=${BATCH_SIZE} FT_EPOCHS=${FT_EPOCHS} HOLD_FRAC=${HOLD_FRAC}"

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
problems = [
    "cvrp","ovrp","vrpb","vrpl","vrptw","ovrptw","ovrpb","ovrpl",
    "ovrpbl","ovrpbtw","ovrpltw","ovrpbltw","vrpbl","vrpbtw","vrpltw","vrpbltw",
]
missing = []
for p in problems:
    inst = f"data/{p}/test/{n}.npz"
    sol = f"data/{p}/test/{n}_sol_pyvrp.npz"
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

echo "===== Start cluster fine-tune ====="
python finetune_cluster.py \
    --n_size "${N_SIZE}" \
    --cluster "${CLUSTER}" \
    --cluster_csv "${CLUSTER_CSV}" \
    --epoch "${EPOCH}" \
    --path_id "${PATH_ID}" \
    --batch_size "${BATCH_SIZE}" \
    --epochs "${FT_EPOCHS}" \
    --holdout_frac "${HOLD_FRAC}" \
    --seed "${SEED}" \
    "${EXTRA_FLAGS[@]}"

echo "===== Done ====="
