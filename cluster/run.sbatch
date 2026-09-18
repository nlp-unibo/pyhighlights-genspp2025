#!/bin/bash
#SBATCH --job-name=genspp
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.log

# No --chdir: the job runs in the directory it was submitted from, so submit
# from the repository root.

# One job per model, five per corpus. The argument picks the corpus:
#
#   mkdir -p logs
#   sbatch --array=0 cluster/run.sbatch toy --smoke   # minutes, says it builds
#   sbatch cluster/run.sbatch toy                     # the five Toy cells
#   sbatch cluster/run.sbatch hatexplain              # the five HateXplain cells
#
# Why the model and not the corpus or the seed. A task runs its five seeds
# itself and writes them into one `results.json`; five per-seed jobs would
# write five run directories under one task name, and `MetricsAnalyzer` keeps
# the newest run per name -- four seeds would vanish from the table rather
# than fail. The model is the coarsest unit that still splits the cost where
# the cost is: from the paper's appendix, a seed takes ~8 min for a baseline
# on Toy and ~36 min for GenSPP, ~4 and ~78 on HateXplain. So GenSPP is three
# hours where a baseline is forty minutes, and running the five in one job
# would make every table wait on the slowest cell.
#
# Nothing here resumes. A job that dies leaves the cells it finished on disk
# and the rest undone; delete that model's results directory and resubmit its
# array index (`sbatch --array=4 cluster/run.sbatch toy`), so a table is not
# read off a half-finished tree beside a finished one.

set -euo pipefail

CORPUS=${1:?usage: sbatch cluster/run.sbatch <toy|hatexplain> [--smoke]}
shift || true

MODELS=(fr mgr mcd grat genspp)
MODEL=${MODELS[${SLURM_ARRAY_TASK_ID:-0}]}

# Override SCRATCH to point somewhere else; the default is this account's
# scratch directory.
SCRATCH=${SCRATCH:-/scratch.hpc/${USER:-$(id -un)}}
mkdir -p "$SCRATCH/cache" logs

EXTRA=()
if [ "$CORPUS" = "hatexplain" ]; then
  # The 1.4 GB GloVe file, staged once by hand. The task refuses without it
  # rather than training on a vocabulary of whatever the corpus happened to
  # contain -- which is the defect pyhighlights 0.9.0's `requires_embeddings`
  # exists to make impossible.
  GLOVE="$SCRATCH/glove/glove.twitter.27B.25d.txt"
  if [ ! -f "$GLOVE" ]; then
    echo "missing $GLOVE -- see README.md" >&2
    exit 2
  fi
  EXTRA+=(--embeddings "$GLOVE")
fi

# The pinned packages live in a virtualenv on scratch rather than in the
# image, which is pulled -- see `build.sbatch` for why. `OMP_NUM_THREADS` is
# set here because Slurm gives the job its cores, and torch sizing its pool to
# the whole node would oversubscribe every one of them.
VENV="$SCRATCH/venv"
if [ ! -x "$VENV/bin/python" ]; then
  echo "missing $VENV -- run cluster/build.sbatch first" >&2
  exit 2
fi

apptainer exec --nv \
  --bind "$PWD:$PWD" --bind "$SCRATCH:$SCRATCH" \
  --env PYTHONPATH="$PWD" \
  --env PYTHONUNBUFFERED=1 \
  --env OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" \
  --env XDG_CACHE_HOME="$SCRATCH/cache" \
  env.sif "$VENV/bin/python" run.py "$CORPUS" --model "$MODEL" "${EXTRA[@]}" "$@"
