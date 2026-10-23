#!/bin/bash
#SBATCH --job-name=genspp-search
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.log

# GenSPP scores candidates in eight CPU worker processes. Reserving a GPU for
# this cell would leave it idle for almost the whole search. This wrapper uses
# the same compute partition and limits as the baselines, but requests no GPU,
# and gives the shared runner model index 4.
#
# Submit from the repository root:
#
#   sbatch cluster/run-genspp.sbatch toy --smoke
#   sbatch cluster/run-genspp.sbatch toy
#   sbatch cluster/run-genspp.sbatch hatexplain
#   sbatch cluster/run-genspp.sbatch toy --g150         # Table 2's G = 150 row

set -euo pipefail

export SLURM_ARRAY_TASK_ID=4
export PYHIGHLIGHTS_USE_GPU=0
exec bash cluster/run.sbatch "$@"
