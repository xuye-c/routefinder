#!/bin/bash
# Distill knobs. Slurm sources this file.
# Frozen teacher BC + freeze encoder + L2-SP. Edit then:
#   bash submit_cluster_distill.sh 50

N_SIZE="${N_SIZE:-50}"
CLUSTER="${CLUSTER:-0}"
EPOCH="${EPOCH:-300}"
PATH_ID="${PATH_ID:-}"
CLUSTER_CSV="${CLUSTER_CSV:-}"
DATA_DIR="${DATA_DIR:-./data}"

FT_EPOCHS="${FT_EPOCHS:-10}"
HOLD_FRAC="${HOLD_FRAC:-0.2}"
SEED="${SEED:-7}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-100}"

LR="${LR:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-6}"
LR_GAMMA="${LR_GAMMA:-0.1}"
LR_DECAY_EPOCH="${LR_DECAY_EPOCH:-0}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"

# teacher = imitate frozen POLAR tours; pyvrp = imitate sol tours if present
TARGET="${TARGET:-teacher}"
# 0 = all customer POMO starts (heavier). 8 is a stable default.
TEACHER_POMO="${TEACHER_POMO:-8}"
L2SP="${L2SP:-1e-3}"
# 1 = freeze encoder+PromptNet, train decoder only
FREEZE_ENCODER="${FREEZE_ENCODER:-1}"

EVAL_ONLY="${EVAL_ONLY:-0}"
SKIP_ZERO_SHOT="${SKIP_ZERO_SHOT:-0}"
