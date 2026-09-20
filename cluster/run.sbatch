#!/bin/bash
#SBATCH --job-name=genspp
#SBATCH --array=0-4
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x_%A_%a.log

# No --chdir: the job runs in the directory it was submitted from, so submit
# from the repository root.

# The partition has to be named. The default is `sbuild`, which is the image
# builder: eight cores, 32095 MB and no GPU, so a job asking for a GPU and
# 32 GB is refused before it queues:
#
#   sbatch: error: Memory specification can not be satisfied
#   sbatch: error: Batch job submission failed: Requested node configuration
#   is not available
#
# Note that 32G is 32768 MB, which is 673 MB more than that node has
# whether or not a GPU is wanted. `l40s` is the choice among the GPU
# partitions because it has four nodes to the others' one, so five array
# tasks spread rather than queue behind each other; every GPU partition here
# is far larger than this study needs. Override it per submission with
# `sbatch --partition=<name> cluster/run.sbatch ...`.

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
# the newest run per name, so four seeds would vanish from the table rather
# than fail. The model is the coarsest unit that still splits the cost where
# the cost is: from the paper's appendix, a seed takes ~8 min for a baseline
# on Toy and ~36 min for GenSPP, ~4 and ~78 on HateXplain.
#
# Measured here under pyhighlights 0.12.0, GenSPP on Toy was 4.5 hours a seed
# rather than 36 minutes, 22 hours for its five against 40 minutes for a
# baseline cell. 0.13.0 scores the candidates in processes and the cell has
# not been timed since, so the limit below is still the one that fits the
# slower figure. The budget is the paper's: a population of 50 over 100 generations at a selection rate
# of 0.5 is 5050 candidates trained per seed. Hence `--time=3-00:00:00`, the
# partition's own limit: nothing here resumes, `results.json` is written once
# after the last seed, and a cell killed on its fifth seed loses all five.
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
mkdir -p "$SCRATCH/cache/pyhighlights" logs

EXTRA=()
if [ "$CORPUS" = "hatexplain" ]; then
  # The 1.4 GB GloVe file, staged by `build.sbatch`. The task refuses without
  # it rather than training on a vocabulary of whatever the corpus happened to
  # contain, which is the defect pyhighlights 0.9.0's `requires_embeddings`
  # exists to make impossible.
  GLOVE="$SCRATCH/glove/glove.twitter.27B.25d.txt"
  if [ ! -f "$GLOVE" ]; then
    echo "missing $GLOVE, see README.md" >&2
    exit 2
  fi
  EXTRA+=(--embeddings "$GLOVE")
fi

# The pinned packages are in the image, in `build.sbatch` and
# `cluster/env.def`, so there is nothing to check for but the image itself.
if [ ! -f env.sif ]; then
  echo "missing $PWD/env.sif, run cluster/build.sbatch first" >&2
  exit 2
fi

# A run writes on the node's own disk and the tree is copied up at the end.
# This repository is checked out on `/scratch.hpc`, and the cluster's guidance
# says why that matters: "problemi di latenza legati al IO sulla share di rete
# /scratch.hpc". Measured on a build node, same URL and the same fifteen
# seconds, a 16 kB-buffered write reached 41 kB/s there against 5.5 MB/s to
# local disk. A training run is the worst possible shape for that, a
# checkpoint written every epoch of five hundred, per seed, per model, so
# `save_path` points at local disk and scratch is written once, at the end.
#
# One directory per job, because the guidance also warns that two jobs sharing
# a node-local path clean up under each other. The copy runs on the way out
# however the job ends, so a job that hits its time limit still leaves the
# cells it finished; anything older than a day is a job that was killed
# outright.
WORK_ROOT="${TMPDIR:-/tmp}/${USER:-$(id -un)}-genspp"
mkdir -p "$WORK_ROOT"
find "$WORK_ROOT" -mindepth 1 -maxdepth 1 -mtime +0 -exec rm -rf {} + 2>/dev/null || true
WORK="$WORK_ROOT/${SLURM_JOB_ID:-$$}${SLURM_ARRAY_TASK_ID:+-$SLURM_ARRAY_TASK_ID}"
mkdir -p "$WORK/results"

collect() {
  mkdir -p "$PWD/results"
  cp -a "$WORK/results/." "$PWD/results/" && rm -rf "$WORK"
}
trap collect EXIT

# The corpora come with the job rather than being read off scratch a record at
# a time: `build.sbatch` stages the toy corpus there, and this copies the
# cache across in one pass: single-digit megabytes, against a run that would
# otherwise read it over the network share.
#
# HateXplain is not staged by the build, so its thirteen megabytes are fetched
# by each job that wants them. Five downloads rather than one, which is the
# price of not sharing a cache between five jobs that run at once.
#
# `PYHIGHLIGHTS_CACHE` and not `XDG_CACHE_HOME`, which the library does not
# read: it caches under `Path.home() / ".cache" / "pyhighlights"` unless that
# variable says otherwise, and apptainer binds the host's home into the
# container, so an unset variable puts corpora in a home directory that has a
# quota and has already been filled once by caches.
#
# Not copied back. Five array jobs writing one cache is a race nothing here
# arbitrates, and a corpus `build.sbatch` did not stage is one download per
# job rather than one corrupted archive for all five.
mkdir -p "$WORK/cache"
if [ -d "$SCRATCH/cache/pyhighlights" ]; then
  cp -a "$SCRATCH/cache/pyhighlights" "$WORK/cache/"
fi

# `OMP_NUM_THREADS` because Slurm gives the job its cores, and torch sizing
# its pool to the whole node would oversubscribe every one of them.
#
# GenSPP needs no exception to this since pyhighlights 0.13.0, which scores a
# search's candidates in worker processes and holds each worker to one torch
# thread itself, so eight workers are eight threads rather than sixty-four.
# Before it the workers were Python threads and the interpreter lock was the
# limit rather than the cores, measured on a running search at 78 threads and
# 240% of a possible 800%, and this job set one thread per candidate to
# compensate.

apptainer exec --nv \
  --bind "$PWD:$PWD" --bind "$SCRATCH:$SCRATCH" --bind "$WORK:$WORK" \
  --env PYTHONPATH="$PWD" \
  --env PYTHONUNBUFFERED=1 \
  --env OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" \
  --env PYHIGHLIGHTS_CACHE="$WORK/cache/pyhighlights" \
  --env XDG_CACHE_HOME="$WORK/cache" \
  env.sif python run.py "$CORPUS" --model "$MODEL" \
  --results "$WORK/results" "${EXTRA[@]}" "$@"
