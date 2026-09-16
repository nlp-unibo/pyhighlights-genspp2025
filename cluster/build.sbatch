#!/bin/bash
#SBATCH --job-name=genspp-build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=sbuild
#SBATCH --output=logs/build_%j.log
#SBATCH --chdir=/scratch.hpc/federico.ruggeri6/pyhighlights-genspp2025

# Containers are built on sbuild, never on a login node -- `apptainer` exists
# only on the compute nodes, so nothing containerized runs from a login shell
# at all (`apptainer: command not found`). This job also builds the registry
# and fetches the toy corpus, because both need the image.
#
#   mkdir -p logs && sbatch cluster/build.sbatch
#   squeue -u $USER          # then read logs/build_<jobid>.log

set -euo pipefail

SCRATCH=/scratch.hpc/federico.ruggeri6
mkdir -p "$SCRATCH/cache"

apptainer build --force env.sif cluster/env.def

# Two checks the queue should not be the place to discover.
#
# The registry: 42 keys, none invalid. An invalid key is an experiment missing
# from the table rather than an error, so it has to be named here.
apptainer exec --nv \
  --bind "$PWD:$PWD" --bind "$SCRATCH:$SCRATCH" \
  --env PYTHONPATH="$PWD" \
  --env XDG_CACHE_HOME="$SCRATCH/cache" \
  env.sif python - <<'PY'
from pathlib import Path
from cinnamon.registry import Registry
import genspp2025, pyhighlights

valid, invalid = Registry.build(
    directory=Path(genspp2025.__file__).parent,
    external_directories=[Path(pyhighlights.__file__).parent],
)
paper = [key for key in valid if key.namespace == "genspp2025"]
print(f"pyhighlights {pyhighlights.__version__}, {len(paper)} keys registered")
assert not invalid, sorted(str(key) for key in invalid)
PY

# The toy corpus: fetched from Zenodo and digest-verified now, while there is
# a person reading the log, rather than by five array jobs at once on a node
# that may have no outbound network at all.
apptainer exec \
  --bind "$PWD:$PWD" --bind "$SCRATCH:$SCRATCH" \
  --env PYTHONPATH="$PWD" \
  --env XDG_CACHE_HOME="$SCRATCH/cache" \
  env.sif python -c "
from pyhighlights_benchmarks.genspp2025.corpora import GenSPPToyLoader
splits = GenSPPToyLoader().load()
print({name: len(frame) for name, frame in splits.items()})
"

echo "built. HateXplain also needs GloVe staged -- see README.md"
