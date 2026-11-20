#!/bin/bash
#SBATCH -J clin-jepa-data
#SBATCH -p <partition>
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH --array=0-9
#SBATCH -o logs/%x_%A_%a.out
# usage: sbatch slurm/data_array.sh <step> <dataset>
#   step:    step02_observations | step03_actions | step04_discretization | step06_trajectories
#   dataset: WeavEHR name of the dataset, e.g. mimic-iv or aumc (Steps 02 and 03 only)

set -euo pipefail
set -a; source .env; set +a
if [ "$1" = "step06_trajectories" ]; then
    python -m clin_jepa.data.$1 --data_dir "$CLIN_JEPA_DATA" --chunk_idx $SLURM_ARRAY_TASK_ID
elif [ "$1" = "step04_discretization" ]; then
    python -m clin_jepa.data.$1 --data_dir "$CLIN_JEPA_DATA" \
        --chunk_idx $SLURM_ARRAY_TASK_ID --total_chunks $SLURM_ARRAY_TASK_COUNT
else
    python -m clin_jepa.data.$1 --weavehr_project "$WEAVEHR_PROJECT" --data_dir "$CLIN_JEPA_DATA" --dataset "$2" \
        --chunk_idx $SLURM_ARRAY_TASK_ID --total_chunks $SLURM_ARRAY_TASK_COUNT
fi
