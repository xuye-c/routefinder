#!/bin/bash
# Submit family-CADA distillation (6 epoch-1 teachers -> one CADA student, 1 epoch).
#
#   bash submit_cada_family_distill.sh           # n=50 and n=100 (2 jobs)
#   bash submit_cada_family_distill.sh 50
#   bash submit_cada_family_distill.sh 100
#   TEACHER_ROOT=/path/to/family-ft bash submit_cada_family_distill.sh 50
#
# Copy distill_cada_family.py to ~/routefinder/ and these files into ~/routefinder/CADA/.
# n=50 teachers default to family-ft-n50-2026-0921-1150 (must have tuned-family*-1.pt).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SLURM_FILE="${SCRIPT_DIR}/run_cada_family_distill.slurm"
WHICH="${1:-all}"

submit_one() {
  local n_size="$1"
  echo "Submitting CADA family distill n=${n_size} teacher_epoch=${TEACHER_EPOCH:-1} distill_epochs=${DISTILL_EPOCHS:-1}"
  sbatch --export=ALL,N_SIZE="${n_size}" "${SLURM_FILE}"
}

if [ "${WHICH}" = "50" ] || [ "${WHICH}" = "all" ]; then
  submit_one 50
fi

if [ "${WHICH}" = "100" ] || [ "${WHICH}" = "all" ]; then
  submit_one 100
fi
