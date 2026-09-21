#!/bin/bash
# Submit one Slurm job per problem size. That job trains all 6 families
# sequentially so cada_family_ft_<jobid>.out contains every cluster.
#
#   bash submit_cada_family_ft.sh           # n=50 and n=100 (2 jobs)
#   bash submit_cada_family_ft.sh 50
#   bash submit_cada_family_ft.sh 100
#   EVAL_ONLY=1 bash submit_cada_family_ft.sh 50
#   CLUSTERS=0,1 LR=1e-6 FT_EPOCHS=2 bash submit_cada_family_ft.sh 50
#
# Copy constraint_family_cluster_6way_{50,100}.csv onto the cluster
# (optional; the python script can also assign families from type names).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SLURM_FILE="${SCRIPT_DIR}/run_cada_family_ft.slurm"
WHICH="${1:-all}"

submit_one() {
  local n_size="$1"
  echo "Submitting CADA family FT n=${n_size} clusters=${CLUSTERS:-0,1,2,3,4,5}"
  sbatch --export=ALL,N_SIZE="${n_size}" "${SLURM_FILE}"
}

if [ "${WHICH}" = "50" ] || [ "${WHICH}" = "all" ]; then
  submit_one 50
fi

if [ "${WHICH}" = "100" ] || [ "${WHICH}" = "all" ]; then
  submit_one 100
fi
