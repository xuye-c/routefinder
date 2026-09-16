#!/bin/bash
# Cluster fine-tune knobs. Slurm sources this file.
# Edit this on the cluster, then resubmit. Env vars still override, e.g.:
#   LR=1e-5 FT_EPOCHS=2 sbatch --export=ALL,N_SIZE=50,CLUSTER=0,LR,FT_EPOCHS run_polar_cluster_ft.slurm
#
# Last run with LR=1e-4 and FT_EPOCHS=10 collapsed (holdout gap 0.6% -> 100%).
# Try LR=1e-5 or 1e-6 and FT_EPOCHS=1 or 2 first.

# ---- job / data ----
N_SIZE="${N_SIZE:-50}"
CLUSTER="${CLUSTER:-0}"
EPOCH="${EPOCH:-300}"                 # pretrained checkpoint epoch
PATH_ID="${PATH_ID:-}"                # empty => n=50/2026-0728-0719 or n=100/2026-0729-1221
CLUSTER_CSV="${CLUSTER_CSV:-}"        # empty => auto-discover umap_cluster_polar_encf_${N_SIZE}.csv
DATA_DIR="${DATA_DIR:-./data}"

# ---- split / train loop ----
FT_EPOCHS="${FT_EPOCHS:-10}"          # fine-tune epochs
HOLD_FRAC="${HOLD_FRAC:-0.2}"         # holdout fraction, stratified by type
SEED="${SEED:-7}"
BATCH_SIZE="${BATCH_SIZE:-}"          # empty => 128 if n=50 else 64
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-100}"

# ---- optimizer ----
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-6}"
LR_GAMMA="${LR_GAMMA:-0.1}"
LR_DECAY_EPOCH="${LR_DECAY_EPOCH:-0}" # 0 => FT_EPOCHS-2
GRAD_CLIP="${GRAD_CLIP:-1.0}"

# ---- loss ----
LOSS="${LOSS:-po}"                    # po | rl
PO_ALPHA="${PO_ALPHA:-0.05}"

# ---- run mode ----
EVAL_ONLY="${EVAL_ONLY:-0}"           # 1 = holdout eval of pretrained ckpt, no training
SKIP_ZERO_SHOT="${SKIP_ZERO_SHOT:-0}" # 1 = skip epoch-0 eval
