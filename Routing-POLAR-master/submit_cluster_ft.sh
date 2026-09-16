#!/bin/bash
# Submit one Slurm job per cluster.
# Edit knobs in cluster_ft_hparams.sh, then:
#   bash submit_cluster_ft.sh           # n=50 (5 jobs) and n=100 (7 jobs)
#   bash submit_cluster_ft.sh 50        # only n=50
#   bash submit_cluster_ft.sh 100       # only n=100
#   EVAL_ONLY=1 bash submit_cluster_ft.sh 50
#
# One-off override:
#   LR=1e-5 FT_EPOCHS=2 bash submit_cluster_ft.sh 50

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SLURM_FILE="${SCRIPT_DIR}/run_polar_cluster_ft.slurm"
WHICH="${1:-all}"

submit_one() {
  local n_size="$1"
  local cluster="$2"
  echo "Submitting n=${n_size} cluster=${cluster}"
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
