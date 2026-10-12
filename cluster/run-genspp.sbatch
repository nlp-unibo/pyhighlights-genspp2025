#!/bin/bash
#SBATCH --job-name=genspp-search
#SBATCH --partition=sbuild
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.log

# GenSPP scores candidates in eight CPU worker processes. Reserving a GPU for
# this cell would leave it idle for almost the whole search. This wrapper gives
# the shared runner model index 4 inside a CPU-only allocation.
#
# Submit from the repository root:
#
#   sbatch cluster/run-genspp.sbatch toy --smoke
#   sbatch cluster/run-genspp.sbatch toy
#   sbatch cluster/run-genspp.sbatch hatexplain
#
# `sbuild` is the known CPU partition. Override it at submission if the live
# cluster exposes a better CPU partition:
#
#   sbatch --partition=<name> cluster/run-genspp.sbatch toy

set -euo pipefail

export SLURM_ARRAY_TASK_ID=4
export PYHIGHLIGHTS_USE_GPU=0
exec bash cluster/run.sbatch "$@"
