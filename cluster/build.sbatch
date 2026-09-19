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

# The image is built on sbuild, never on a login node -- `apptainer` exists
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
#
# Cleared first. A build that is cancelled -- by `scancel`, by a time limit,
# by the registry -- leaves the unpacked image behind, nine gigabytes of it,
# and nothing else ever removes it. The next build then measures the free
# space with its own predecessor still occupying it, reads thirteen gigabytes
# where the disk has twenty-two, and falls back to scratch for no reason.
# Nothing here is worth keeping between jobs: the layer cache is on scratch
# and this directory only ever holds one build's scratch space.
LOCAL_TMP="${TMPDIR:-/tmp}/${USER:-$(id -un)}-apptainer"
rm -rf "$LOCAL_TMP"
mkdir -p "$LOCAL_TMP"
AVAILABLE=$(df -Pk "$LOCAL_TMP" | awk 'NR==2 { print int($4 / 1048576) }')
if [ "${AVAILABLE:-0}" -ge 20 ]; then
  export APPTAINER_TMPDIR="$LOCAL_TMP"
else
  echo "local disk has ${AVAILABLE:-0}G, assembling on scratch instead (slow)" >&2
  export APPTAINER_TMPDIR="$SCRATCH/apptainer/tmp"
fi
mkdir -p "$APPTAINER_TMPDIR"

# Built from `cluster/env.def`, which holds the base tag and the pins. A
# definition file rather than a bare `apptainer pull` for two reasons: the
# pinned packages go into the image itself instead of a virtualenv on scratch
# that every job has to bind and point at, and a build writes the finished SIF
# once where a cached pull writes it twice -- once into the cache and once to
# the destination, both on a filesystem measured here at 11.6 MB/s.
#
# `--ignore-fakeroot-command` is not optional. The user is not in
# `/etc/subuid`, so apptainer builds under a root-mapped namespace and wraps
# `%post` in its own `fakeroot`, injecting a host `libfakeroot.so` built
# against GLIBC 2.38. The base image is Ubuntu 22.04, which has 2.35, so
# `%post` dies before its first line:
#
#   /bin/sh: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.38' not found
#     (required by /.singularity.d/libs/libfakeroot.so)
#
# The flag drops that wrapper and leaves the namespace's own root mapping,
# which is all `%post` needs now that it only runs pip.
#
# Retried, because the registry drops connections partway through the base
# image:
#
#   error writing layer: stream error: stream ID 7; INTERNAL_ERROR;
#   received from peer
#
# That is the registry resetting an HTTP/2 stream, not anything this job did,
# and it fails the whole build after however much had already transferred.
# Three attempts rather than one, because a job that dies on it costs a queue
# wait as well as the download.
echo "building env.sif"
for attempt in 1 2 3; do
  if apptainer build --ignore-fakeroot-command --force env.sif cluster/env.def; then
    break
  fi
  if [ "$attempt" = 3 ]; then
    echo "build failed three times; the registry is unreachable or unwell" >&2
    exit 1
  fi
  echo "build failed (attempt $attempt), retrying in 60s" >&2
  sleep 60
done

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
