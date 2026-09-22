#!/bin/bash
# Polar 6-family FT: generated data, freeze encoder, train Prompt+decoder.
# Training-time PyVRP LS (adds train wall time). Eval is greedy+8-aug, no LS.

N_SIZE="${N_SIZE:-50}"
FAMILIES="${FAMILIES:-0,1,2,3,4,5}"
EPOCH="${EPOCH:-300}"
PATH_ID="${PATH_ID:-}"
CLUSTER_CSV="${CLUSTER_CSV:-}"
DATA_DIR="${DATA_DIR:-./data}"
CONDA_ENV="${CONDA_ENV:-polar}"

FT_EPOCHS="${FT_EPOCHS:-3}"
EPISODES="${EPISODES:-4096}"
SEED="${SEED:-7}"
BATCH_SIZE="${BATCH_SIZE:-}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-100}"

LR="${LR:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-6}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
LOSS="${LOSS:-po}"
PO_ALPHA="${PO_ALPHA:-0.05}"
TRAIN_POMO="${TRAIN_POMO:-8}"
TRAIN_MODULES="${TRAIN_MODULES:-prompt_decoder}"
USE_LS="${USE_LS:-1}"
LS_NB_GRANULAR="${LS_NB_GRANULAR:-20}"

EVAL_ONLY="${EVAL_ONLY:-0}"
SKIP_ZERO_SHOT="${SKIP_ZERO_SHOT:-0}"
