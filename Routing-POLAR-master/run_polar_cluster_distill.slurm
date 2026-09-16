#!/bin/bash
#SBATCH --job-name=polar_cdist
#SBATCH --partition=common-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=polar_cdist_%j.out
#SBATCH --error=polar_cdist_%j.err
#
# Knobs: cluster_distill_hparams.sh
#   bash submit_cluster_distill.sh 50
#   N_SIZE=50 CLUSTER=0 sbatch --export=ALL,N_SIZE=50,CLUSTER=0 run_polar_cluster_distill.slurm

source /opt/apps/centos8/anaconda3/etc/profile.d/conda.sh
conda activate polar

cd ~/routefinder/Routing-POLAR-master
source ./cluster_distill_hparams.sh

if [ "${N_SIZE}" = "50" ]; then
  PATH_ID="${PATH_ID:-n=50/2026-0728-0719}"
  BATCH_SIZE="${BATCH_SIZE:-32}"
else
  PATH_ID="${PATH_ID:-n=100/2026-0729-1221}"
  BATCH_SIZE="${BATCH_SIZE:-16}"
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
  exit 1
fi

echo "===== Environment ====="
echo "Host: $(hostname) Job: ${SLURM_JOB_ID}"
echo "Python: $(which python) $(python --version)"
echo "===== Hparams ====="
echo "N_SIZE=${N_SIZE} CLUSTER=${CLUSTER} PATH_ID=${PATH_ID}"
echo "CLUSTER_CSV=${CLUSTER_CSV}"
echo "FT_EPOCHS=${FT_EPOCHS} HOLD_FRAC=${HOLD_FRAC} BATCH_SIZE=${BATCH_SIZE}"
echo "LR=${LR} L2SP=${L2SP} TARGET=${TARGET} TEACHER_POMO=${TEACHER_POMO}"
echo "FREEZE_ENCODER=${FREEZE_ENCODER} EVAL_ONLY=${EVAL_ONLY}"

nvidia-smi
python - <<'PY'
import torch
print("PyTorch:", torch.__version__, "CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

CKPT="result/${PATH_ID}/checkpoint-${EPOCH}.pt"
ls -lh "${CKPT}"
python - <<PY
from pathlib import Path
p = Path("${CKPT}")
if p.read_bytes()[:32].startswith(b"version https://git-lfs.github.com"):
    raise SystemExit("ERROR: checkpoint is a Git LFS pointer")
print("checkpoint ok")
PY

EXTRA_FLAGS=()
if [ "${EVAL_ONLY}" = "1" ]; then
  EXTRA_FLAGS+=(--eval_only)
fi
if [ "${SKIP_ZERO_SHOT}" = "1" ]; then
  EXTRA_FLAGS+=(--skip_zero_shot)
fi
if [ "${FREEZE_ENCODER}" = "0" ]; then
  EXTRA_FLAGS+=(--no_freeze_encoder)
else
  EXTRA_FLAGS+=(--freeze_encoder)
fi

echo "===== Start cluster distill ====="
python finetune_cluster_distill.py \
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
    --target "${TARGET}" \
    --teacher_pomo "${TEACHER_POMO}" \
    --l2sp "${L2SP}" \
    "${EXTRA_FLAGS[@]}"

echo "===== Done ====="
