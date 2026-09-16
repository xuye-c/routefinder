#!/bin/bash
# Submit distill jobs (one per cluster).
# Edit cluster_distill_hparams.sh first.
#   bash submit_cluster_distill.sh 50
#   bash submit_cluster_distill.sh 100
#   LR=1e-5 TARGET=teacher bash submit_cluster_distill.sh 50

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SLURM_FILE="${SCRIPT_DIR}/run_polar_cluster_distill.slurm"
WHICH="${1:-all}"

submit_one() {
  local n_size="$1"
  local cluster="$2"
  echo "Submitting distill n=${n_size} cluster=${cluster}"
  sbatch --export=ALL,N_SIZE="${n_size}",CLUSTER="${cluster}" "${SLURM_FILE}"
}

if [ "${WHICH}" = "50" ] || [ "${WHICH}" = "all" ]; then
  for c in 0 1 2 3 4; do
    submit_one 50 "${c}"
  done
fi

if [ "${WHICH}" = "100" ] || [ "${WHICH}" = "all" ]; then
  for c in 0 1 2 3 4 5 6; do
    submit_one 100 "${c}"
  done
fi
