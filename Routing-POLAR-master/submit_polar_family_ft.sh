#!/bin/bash
#   bash submit_polar_family_ft.sh 50
#   FAMILIES=1 bash submit_polar_family_ft.sh 50
#   CLUSTER_CSV=$HOME/constraint_family_cluster_6way_50.csv FAMILIES=1 bash submit_polar_family_ft.sh 50

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SLURM_FILE="${SCRIPT_DIR}/run_polar_family_ft.slurm"
WHICH="${1:-50}"

submit_one() {
  local n_size="$1"
  echo "Submitting Polar family FT n=${n_size} families=${FAMILIES:-0,1,2,3,4,5} modules=${TRAIN_MODULES:-prompt}"
  sbatch --export=ALL,N_SIZE="${n_size}" "${SLURM_FILE}"
}

if [ "${WHICH}" = "50" ] || [ "${WHICH}" = "all" ]; then
  submit_one 50
fi
if [ "${WHICH}" = "100" ] || [ "${WHICH}" = "all" ]; then
  submit_one 100
fi
