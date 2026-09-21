#!/bin/bash
#SBATCH --job-name=cada_fam_kd
#SBATCH --partition=common-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=cada_family_distill_%j.out
#SBATCH --error=cada_family_distill_%j.err
#
# Slurm copies this file to /var/spool/... so $0 is NOT CADA/.
# Always cd via SLURM_SUBMIT_DIR or ~/routefinder/CADA.
#
# Submit from ~/routefinder/CADA:
#   sbatch --export=ALL,N_SIZE=50 run_cada_family_distill.slurm
#   bash submit_cada_family_distill.sh 50

set -euo pipefail
export PYTHONUNBUFFERED=1

source /opt/apps/centos8/anaconda3/etc/profile.d/conda.sh

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/cada_family_distill_hparams.sh" ]; then
  CADA_DIR="${SLURM_SUBMIT_DIR}"
elif [ -f "${HOME}/routefinder/CADA/cada_family_distill_hparams.sh" ]; then
  CADA_DIR="${HOME}/routefinder/CADA"
else
  echo "ERROR: cannot find cada_family_distill_hparams.sh"
  echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}"
  echo "tried \$HOME/routefinder/CADA"
  exit 1
fi

cd "${CADA_DIR}"
# shellcheck source=cada_family_distill_hparams.sh
source "${CADA_DIR}/cada_family_distill_hparams.sh"
FOLDS="${FOLDS:-10}"
DISTILL_EPOCHS="${DISTILL_EPOCHS:-1}"
TEACHER_EPOCH="${TEACHER_EPOCH:-1}"

conda activate "${CONDA_ENV}"

RF_ROOT="$(cd "${CADA_DIR}/.." && pwd)"
cd "${RF_ROOT}"

if [ "${N_SIZE}" = "50" ]; then
  PATH_ID="${PATH_ID:-2024-1111-1139}"
  BATCH_SIZE="${BATCH_SIZE:-32}"
  if [ -z "${TEACHER_ROOT}" ]; then
    TEACHER_ROOT="${RF_ROOT}/CADA/50/result/family-ft-n50-2026-0921-1150"
  fi
else
  PATH_ID="${PATH_ID:-2024-1121-1355}"
  BATCH_SIZE="${BATCH_SIZE:-16}"
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
echo "CADA_DIR: ${CADA_DIR}"
echo "CWD: $(pwd)"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "Conda env: ${CONDA_DEFAULT_ENV}"

echo "===== Hparams ====="
echo "N_SIZE=${N_SIZE} CLUSTERS=${CLUSTERS} PATH_ID=${PATH_ID} EPOCH=${EPOCH}"
echo "TEACHER_ROOT=${TEACHER_ROOT:-<auto>}"
echo "TEACHER_EPOCH=${TEACHER_EPOCH} DISTILL_EPOCHS=${DISTILL_EPOCHS} FOLDS=${FOLDS}"
echo "CLUSTER_CSV=${CLUSTER_CSV:-<type mapping if missing>}"
echo "DATA_DIR=${DATA_DIR:-<auto>}"
echo "BATCH_SIZE=${BATCH_SIZE} EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
echo "LR=${LR} WEIGHT_DECAY=${WEIGHT_DECAY} GRAD_CLIP=${GRAD_CLIP} L2SP=${L2SP}"
echo "TEACHER_POMO=${TEACHER_POMO} NO_FREEZE_ENCODER=${NO_FREEZE_ENCODER}"
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

echo "===== Student checkpoint ====="
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

echo "===== Teacher epoch-${TEACHER_EPOCH} checkpoints ====="
if [ -z "${TEACHER_ROOT}" ]; then
  echo "ERROR: TEACHER_ROOT is empty (n=100 needs an explicit family-ft result dir)"
  exit 1
fi
python - <<PY
from pathlib import Path
root = Path("${TEACHER_ROOT}")
epoch = int("${TEACHER_EPOCH}")
names = {
    0: "open_TW",
    1: "closed_TW",
    2: "open_backhaul_noTW",
    3: "open_plain_noTW",
    4: "closed_backhaul_noTW",
    5: "closed_plain_noTW",
}
missing = []
for c, name in names.items():
    fam = root / f"family{c}-{name}"
    p = fam / f"tuned-family{c}-{epoch}.pt"
    if not p.is_file():
        alts = sorted(fam.glob(f"tuned-family{c}-*-e{epoch}.pt")) if fam.is_dir() else []
        if alts:
            p = alts[0]
    if not p.is_file():
        missing.append(str(fam / f"tuned-family{c}-{epoch}.pt"))
    else:
        print("OK", p)
if missing:
    print("MISSING teachers:")
    for m in missing:
        print(" ", m)
    raise SystemExit(1)
print("All 6 family teachers found under", root)
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

EXTRA=()
if [ "${EVAL_ONLY}" = "1" ]; then
  EXTRA+=(--eval_only)
fi
if [ "${SKIP_ZERO_SHOT}" = "1" ]; then
  EXTRA+=(--skip_zero_shot)
fi
if [ "${NO_FREEZE_ENCODER}" = "1" ]; then
  EXTRA+=(--no_freeze_encoder)
fi
if [ -n "${CLUSTER_CSV}" ] && [ -f "${CLUSTER_CSV}" ]; then
  EXTRA+=(--cluster_csv "${CLUSTER_CSV}")
else
  echo "NOTE: cluster csv not found; using built-in 6-family type mapping"
fi
if [ -n "${DATA_DIR}" ]; then
  EXTRA+=(--data_dir "${DATA_DIR}")
fi
if [ -n "${TEACHER_ROOT}" ]; then
  EXTRA+=(--teacher_root "${TEACHER_ROOT}")
fi

echo "===== Start CADA family distill (6 teachers e${TEACHER_EPOCH} -> 1 student x ${DISTILL_EPOCHS} epoch) ====="
python -u distill_cada_family.py \
    --n_size "${N_SIZE}" \
    --clusters "${CLUSTERS}" \
    --epoch "${EPOCH}" \
    --path_id "${PATH_ID}" \
    --batch_size "${BATCH_SIZE}" \
    --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --epochs "${DISTILL_EPOCHS}" \
    --folds "${FOLDS}" \
    --holdout_frac "${HOLD_FRAC}" \
    --lr "${LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --grad_clip "${GRAD_CLIP}" \
    --seed "${SEED}" \
    --l2sp "${L2SP}" \
    --teacher_epoch "${TEACHER_EPOCH}" \
    --teacher_pomo "${TEACHER_POMO}" \
    ${EXTRA[@]+"${EXTRA[@]}"}

echo "===== Done ====="
echo "Slurm stdout: ${CADA_DIR}/cada_family_distill_${SLURM_JOB_ID:-local}.out"
echo "OOF table: CADA/${N_SIZE}/result/family-distill-n${N_SIZE}-e${TEACHER_EPOCH}-*/oof_distill.csv"
