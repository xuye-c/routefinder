#!/bin/bash
#SBATCH --job-name=polar_fam_ft
#SBATCH --partition=common-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=polar_family_ft_%j.out
#SBATCH --error=polar_family_ft_%j.err
#
# Submit from ~/routefinder/Routing-POLAR-master:
#   bash submit_polar_family_ft.sh 50

set -euo pipefail
export PYTHONUNBUFFERED=1

source /opt/apps/centos8/anaconda3/etc/profile.d/conda.sh

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/polar_family_ft_hparams.sh" ]; then
  POLAR_DIR="${SLURM_SUBMIT_DIR}"
elif [ -f "${HOME}/routefinder/Routing-POLAR-master/polar_family_ft_hparams.sh" ]; then
  POLAR_DIR="${HOME}/routefinder/Routing-POLAR-master"
else
  echo "ERROR: cannot find polar_family_ft_hparams.sh"
  exit 1
fi

cd "${POLAR_DIR}"
# shellcheck source=polar_family_ft_hparams.sh
source "${POLAR_DIR}/polar_family_ft_hparams.sh"

conda activate "${CONDA_ENV}"

if [ "${N_SIZE}" = "50" ]; then
  PATH_ID="${PATH_ID:-n=50/2026-0728-0719}"
  BATCH_SIZE="${BATCH_SIZE:-64}"
else
  PATH_ID="${PATH_ID:-n=100/2026-0729-1221}"
  BATCH_SIZE="${BATCH_SIZE:-32}"
fi

CSV_NAME="constraint_family_cluster_6way_${N_SIZE}.csv"
if [ -z "${CLUSTER_CSV}" ]; then
  for cand in \
    "${POLAR_DIR}/${CSV_NAME}" \
    "${POLAR_DIR}/../${CSV_NAME}" \
    "${HOME}/${CSV_NAME}" \
    "${HOME}/routefinder/${CSV_NAME}" \
    "${HOME}/wfl_sw/${CSV_NAME}"
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
echo "POLAR_DIR: ${POLAR_DIR}"
echo "CWD: $(pwd)"
echo "Python: $(which python)"

echo "===== Hparams ====="
echo "N_SIZE=${N_SIZE} FAMILIES=${FAMILIES} PATH_ID=${PATH_ID} EPOCH=${EPOCH}"
echo "CLUSTER_CSV=${CLUSTER_CSV:-missing}"
echo "DATA_DIR=${DATA_DIR}"
echo "FT_EPOCHS=${FT_EPOCHS} EPISODES=${EPISODES} TRAIN_MODULES=${TRAIN_MODULES}"
echo "LR=${LR} TRAIN_POMO=${TRAIN_POMO} LOSS=${LOSS}"

echo "===== GPU ====="
nvidia-smi || true

CKPT="result/${PATH_ID}/checkpoint-${EPOCH}.pt"
ls -lh "${CKPT}"
python - <<PY
from pathlib import Path
p = Path("${CKPT}")
head = p.read_bytes()[:40]
if head.startswith(b"version https://git-lfs.github.com"):
    raise SystemExit("ERROR: checkpoint is a Git LFS pointer")
print("checkpoint ok")
PY

if [ -z "${CLUSTER_CSV}" ] || [ ! -f "${CLUSTER_CSV}" ]; then
  echo "ERROR: need ${CSV_NAME} (6-family mapping)"
  exit 1
fi

python - <<PY
import os
n = int("${N_SIZE}")
root = "${DATA_DIR}"
missing = []
for p in ["cvrp","ovrp","vrpb","vrpl","vrptw","ovrptw","ovrpb","ovrpl","ovrpbl","ovrpbtw","ovrpltw","ovrpbltw","vrpbl","vrpbtw","vrpltw","vrpbltw"]:
    for f in (f"{root}/{p}/test/{n}.npz", f"{root}/{p}/test/{n}_sol_pyvrp.npz"):
        if not os.path.isfile(f):
            missing.append(f)
if missing:
    print("MISSING", missing[:8], "...")
    raise SystemExit(1)
print("data ok")
PY

EXTRA=()
if [ "${EVAL_ONLY}" = "1" ]; then EXTRA+=(--eval_only); fi
if [ "${SKIP_ZERO_SHOT}" = "1" ]; then EXTRA+=(--skip_zero_shot); fi

echo "===== Start Polar family specialization ====="
python -u finetune_polar_family.py \
    --n_size "${N_SIZE}" \
    --families "${FAMILIES}" \
    --cluster_csv "${CLUSTER_CSV}" \
    --epoch "${EPOCH}" \
    --path_id "${PATH_ID}" \
    --data_dir "${DATA_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --epochs "${FT_EPOCHS}" \
    --episodes "${EPISODES}" \
    --lr "${LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --grad_clip "${GRAD_CLIP}" \
    --seed "${SEED}" \
    --loss "${LOSS}" \
    --po_alpha "${PO_ALPHA}" \
    --train_pomo "${TRAIN_POMO}" \
    --train_modules "${TRAIN_MODULES}" \
    ${EXTRA[@]+"${EXTRA[@]}"}

echo "===== Done ====="
