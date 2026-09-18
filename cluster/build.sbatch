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

# The image is pulled on sbuild, never on a login node -- `apptainer` exists
# only on the compute nodes, so nothing containerized runs from a login shell
# at all (`apptainer: command not found`). This job also installs the pinned
# packages, builds the registry and fetches the toy corpus, because all three
# need the image.
#
#   mkdir -p logs && sbatch cluster/build.sbatch
#   squeue -u $USER          # then read logs/build_<jobid>.log
#
# The image is pulled rather than built from a definition file. A definition
# file's `%post` runs under fakeroot, and apptainer injects the host's
# `libfakeroot.so` into the container to do it, so a host newer than the base
# image fails before pip is reached at all:
#
#   /bin/sh: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.38' not found
#   (required by /.singularity.d/libs/libfakeroot.so)
#
# Every `pytorch/pytorch` runtime tag is Ubuntu 22.04, which carries glibc
# 2.35, so no tag of that base satisfies a host needing 2.38. A pull runs no
# `%post`, so the two pinned packages go into a virtualenv on scratch instead,
# built with the image's own Python and with the image's torch visible through
# `--system-site-packages`. Same pins, and nothing needs root.

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
# The cache goes to scratch, which is where this account has room, and is kept
# rather than cleared: a rebuild that changes only a pip floor reuses the base
# image's layers instead of fetching two gigabytes again.
export APPTAINER_CACHEDIR="$SCRATCH/apptainer/cache"
mkdir -p "$APPTAINER_CACHEDIR"

# The assembly directory is a different question, and scratch is the wrong
# answer for it. Converting the layers to a SIF unpacks the whole image as
# ordinary files and then squashes them back, which is tens of thousands of
# small writes: on a parallel filesystem that step ran at 1.4 GB in two and a
# half hours for a seven gigabyte image, with the process at one percent of a
# core because it was waiting on the filesystem rather than working. The same
# mount reports `nodev`, which apptainer warns about because the device nodes
# it unpacks cannot be created there at all.
#
# So the assembly happens on the node's own disk, and scratch is the fallback
# for a node whose local disk is too small to hold the unpacked image beside
# the SIF. Slow is better than a build that runs out of room.
LOCAL_TMP="${TMPDIR:-/tmp}/${USER:-$(id -un)}-apptainer"
mkdir -p "$LOCAL_TMP"
AVAILABLE=$(df -Pk "$LOCAL_TMP" | awk 'NR==2 { print int($4 / 1048576) }')
if [ "${AVAILABLE:-0}" -ge 20 ]; then
  export APPTAINER_TMPDIR="$LOCAL_TMP"
else
  echo "local disk has ${AVAILABLE:-0}G, assembling on scratch instead (slow)" >&2
  export APPTAINER_TMPDIR="$SCRATCH/apptainer/tmp"
fi
mkdir -p "$APPTAINER_TMPDIR"

# This reproduction has no transformer in it. Both corpora encode with a GRU,
# hidden 8 on Toy and hidden 16 on HateXplain, so nothing loads a pretrained
# language model and no dependency imposes a torch floor of its own. The base
# is the stock runtime image, the smallest one that runs the library.
IMAGE=docker://pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
# Pulled once. The conversion is the expensive half of this job and the tag is
# pinned, so a rebuild that exists to move a pip floor has no reason to repeat
# it. Delete `env.sif` to fetch the tag again.
if [ -f env.sif ]; then
  echo "image already built at $PWD/env.sif"
else
  apptainer pull env.sif "$IMAGE"
fi

# The pinned packages, in a virtualenv on scratch rather than in the image.
#
# Each pin is a defect rather than a preference. pyhighlights 0.12.0 reports
# what a run cost under `cost_`, which is the table `compare.py` prints beside
# Table 1. 0.11.0 below it is what `ToyLoader` reads a published corpus with,
# where earlier versions read it with a `GenSPPToyLoader` that no longer
# exists and `ToyLoader(url=...)` is an unknown argument -- so the failure mode
# is training on a corpus of the right shape and the wrong content. 0.9.0
# below that renamed `AnnotationAggregator(rationale=)` to `highlights=` and
# `RATIONALES` to `HIGHLIGHTS`, which the HateXplain pipeline calls, and
# carries the GenSPP fixes this reproduction depends on to mean anything.
#
# cinnamon-core `>=2.1.3` is pyhighlights' own floor, repeated because 2.1.2
# forgot every module below a scanned root at the next reset, so a second build
# resolved a second copy of every class --
#
#   TypeError: 'pyhighlights.components.tasks.SPPTask' resolves to SPPTask,
#   which is not a subclass of SPPTask.
#
# -- and it is yanked.
VENV="$SCRATCH/venv"
if [ ! -x "$VENV/bin/python" ]; then
  apptainer exec --bind "$SCRATCH:$SCRATCH" env.sif \
    python -m venv --system-site-packages "$VENV"
fi
apptainer exec --bind "$SCRATCH:$SCRATCH" env.sif \
  "$VENV/bin/pip" install --no-cache-dir --upgrade pip
apptainer exec --bind "$SCRATCH:$SCRATCH" env.sif \
  "$VENV/bin/pip" install --no-cache-dir \
  "pyhighlights==0.12.0" "cinnamon-core>=2.1.3,<3"

# A wiring that cannot import the library is worth failing here rather than
# three hours into a queue.
apptainer exec --bind "$SCRATCH:$SCRATCH" env.sif \
  "$VENV/bin/python" -c "import pyhighlights, torch; print(pyhighlights.__version__, torch.__version__)"

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
  --env PYTHONUNBUFFERED=1 \
  --env XDG_CACHE_HOME="$SCRATCH/cache" \
  env.sif "$VENV/bin/python" - <<'PY'
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
