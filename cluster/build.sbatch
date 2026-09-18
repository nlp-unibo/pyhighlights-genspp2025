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

# Apptainer caches the layers it pulls under `$HOME/.apptainer` and assembles
# the image in `/tmp`, and a home directory under quota fails the build after
# the download rather than before it:
#
#   FATAL: While performing build: conveyor failed to get: write
#   /home/<user>/.apptainer/cache/blob/oci-layout: disk quota exceeded
#
# Both go to scratch, which is where this account has room. The cache is kept
# rather than cleared: a rebuild that changes only a pip floor reuses the base
# image's layers instead of fetching two gigabytes again.
export APPTAINER_CACHEDIR="$SCRATCH/apptainer/cache"
export APPTAINER_TMPDIR="$SCRATCH/apptainer/tmp"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

apptainer build --force env.sif cluster/env.def

# Two checks the queue should not be the place to discover, in one registry
# build: the keys resolve, and the toy corpus is on disk.
#
# The registry: every key resolves and none is invalid. An invalid key is an
# experiment missing from the table rather than an error, so it has to be
# named here. The count is printed rather than pinned: a key added to the
# reproduction is not a build failure.
#
# The toy corpus is fetched from Zenodo and digest-verified now, while there
# is a person reading the log, rather than by five array jobs at once on a
# node that may have no outbound network at all. It is read through the
# registered `dataset` configuration, so the archive, the digest and the split
# scheme are the ones a run uses rather than a second copy of them here.
apptainer exec --nv \
  --bind "$PWD:$PWD" --bind "$SCRATCH:$SCRATCH" \
  --env PYTHONPATH="$PWD" \
  --env XDG_CACHE_HOME="$SCRATCH/cache" \
  env.sif python - <<'PY'
from pathlib import Path

from cinnamon.registry import Registry

import genspp2025
import pyhighlights
from genspp2025.configurations.toy.keys import TOY

valid, invalid = Registry.build(
    directory=Path(genspp2025.__file__).parent,
    external_directories=[Path(pyhighlights.__file__).parent],
)
paper = [key for key in valid if key.namespace == "genspp2025"]
print(f"pyhighlights {pyhighlights.__version__}, {len(paper)} keys registered")
assert not invalid, sorted(str(key) for key in invalid)

splits = Registry.from_key(TOY).load()
print({name: len(frame) for name, frame in splits.items()})
PY

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
