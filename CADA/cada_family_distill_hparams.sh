#!/bin/bash
# Distill 6 family-CADA epoch-1 teachers into one pretrained CADA student.
# Everyone uses 1 epoch: teachers = tuned-family*-1.pt, student distill epochs = 1.
# Env vars override, e.g.:
#   N_SIZE=50 TEACHER_EPOCH=1 DISTILL_EPOCHS=1 sbatch --export=ALL,N_SIZE run_cada_family_distill.slurm

# ---- job / data ----
N_SIZE="${N_SIZE:-50}"
CLUSTERS="${CLUSTERS:-0,1,2,3,4,5}"
EPOCH="${EPOCH:-300}"                  # student pretrained checkpoint epoch
PATH_ID="${PATH_ID:-}"                 # empty => 50: 2024-1111-1139  100: 2024-1121-1355
CLUSTER_CSV="${CLUSTER_CSV:-}"
DATA_DIR="${DATA_DIR:-}"
CONDA_ENV="${CONDA_ENV:-polar}"

# teachers: 80/20 family FT that saved epoch-1 ckpts (10-fold job did not)
TEACHER_ROOT="${TEACHER_ROOT:-}"       # empty => CADA/50/result/family-ft-n50-2026-0921-1150
TEACHER_EPOCH="${TEACHER_EPOCH:-1}"

# ---- split / train ----
FOLDS="${FOLDS:-10}"
DISTILL_EPOCHS="${DISTILL_EPOCHS:-1}"
HOLD_FRAC="${HOLD_FRAC:-0.2}"
SEED="${SEED:-7}"
BATCH_SIZE="${BATCH_SIZE:-}"           # empty => 32 if n=50 else 16 (NLL + POMO tours)
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-100}"

# ---- optimizer ----
LR="${LR:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-6}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
L2SP="${L2SP:-1e-3}"
TEACHER_POMO="${TEACHER_POMO:-8}"
NO_FREEZE_ENCODER="${NO_FREEZE_ENCODER:-0}"

# ---- run mode ----
EVAL_ONLY="${EVAL_ONLY:-0}"
SKIP_ZERO_SHOT="${SKIP_ZERO_SHOT:-0}"
