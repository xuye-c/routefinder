#!/bin/bash
# CADA 6-family fine-tune knobs. Slurm sources this file.
# One job trains ALL families in --clusters (default 0-5) so the same .out
# contains every specialist. Env vars override, e.g.:
#   LR=1e-6 FT_EPOCHS=2 N_SIZE=50 sbatch --export=ALL,N_SIZE,LR,FT_EPOCHS run_cada_family_ft.slurm

# ---- job / data ----
N_SIZE="${N_SIZE:-50}"
CLUSTERS="${CLUSTERS:-0,1,2,3,4,5}"   # all six families in one job
EPOCH="${EPOCH:-300}"                  # pretrained checkpoint epoch
PATH_ID="${PATH_ID:-}"                 # empty => 50: 2024-1111-1139  100: 2024-1121-1355
CLUSTER_CSV="${CLUSTER_CSV:-}"         # empty => auto-discover constraint_family_cluster_6way_${N_SIZE}.csv
DATA_DIR="${DATA_DIR:-}"               # empty => routefinder/data or Routing-POLAR-master/data
CONDA_ENV="${CONDA_ENV:-polar}"

# ---- split / train loop ----
FT_EPOCHS="${FT_EPOCHS:-5}"
HOLD_FRAC="${HOLD_FRAC:-0.2}"
SEED="${SEED:-7}"
BATCH_SIZE="${BATCH_SIZE:-}"           # empty => 128 if n=50 else 64
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-100}"

# ---- optimizer (small LR: Polar 1e-4 / 10 epoch collapsed) ----
LR="${LR:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-6}"
LR_GAMMA="${LR_GAMMA:-0.1}"
LR_DECAY_EPOCH="${LR_DECAY_EPOCH:-0}"  # 0 => FT_EPOCHS-2
GRAD_CLIP="${GRAD_CLIP:-1.0}"

# ---- loss (rl = original CADA REINFORCE; po = preference) ----
LOSS="${LOSS:-rl}"
PO_ALPHA="${PO_ALPHA:-0.05}"

# ---- run mode ----
EVAL_ONLY="${EVAL_ONLY:-0}"
SKIP_ZERO_SHOT="${SKIP_ZERO_SHOT:-0}"
