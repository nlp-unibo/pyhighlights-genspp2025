#!/bin/bash
#SBATCH --job-name=genspp-build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=sbuild
#SBATCH --output=logs/build_%j.log

# No --chdir: the job runs in the directory it was submitted from, so submit
# from the repository root.

# Containers are built on sbuild, never on a login node -- `apptainer` exists
# only on the compute nodes, so nothing containerized runs from a login shell
# at all (`apptainer: command not found`). This job also builds the registry
# and fetches the toy corpus, because both need the image.
#
#   mkdir -p logs && sbatch cluster/build.sbatch
#   squeue -u $USER          # then read logs/build_<jobid>.log

set -euo pipefail

# Override SCRATCH to point somewhere else; the default is this account's
# scratch directory.
SCRATCH=${SCRATCH:-/scratch.hpc/${USER:-$(id -un)}}
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

# GloVe, for HateXplain. 1.4 GB, so it is fetched once here rather than by
# five array jobs at the same time, and kept on scratch rather than in the
# image: an image is rebuilt whenever a floor moves, and this file never
# changes. The paper expects you to fetch it yourself, which is why the task
# takes a path and nothing in the library downloads it.
GLOVE_DIR="$SCRATCH/glove"
GLOVE="$GLOVE_DIR/glove.twitter.27B.25d.txt"
mkdir -p "$GLOVE_DIR"

if [ -f "$GLOVE" ]; then
  echo "glove already staged at $GLOVE"
else
  # `-c` so an interrupted job resumes its download instead of starting the
  # 1.4 GB again, and the archive is removed only once the member is out.
  wget -c -O "$GLOVE_DIR/glove.twitter.27B.zip" \
    https://nlp.stanford.edu/data/glove.twitter.27B.zip
  unzip -o -j "$GLOVE_DIR/glove.twitter.27B.zip" glove.twitter.27B.25d.txt \
    -d "$GLOVE_DIR"
  rm -f "$GLOVE_DIR/glove.twitter.27B.zip"
fi

# Stanford publishes no digest for the archive, so the check is on the shape
# of what came out: a truncated download unzips to a short file, and the wrong
# member gives the wrong width. 25 dimensions plus the token is 26 fields.
awk 'NR==1 { if (NF != 26) { print "glove: " NF " fields on line 1, expected 26" > "/dev/stderr"; exit 1 } }' "$GLOVE"
echo "glove staged: $(wc -l < "$GLOVE") vectors at $GLOVE"

echo "built."
