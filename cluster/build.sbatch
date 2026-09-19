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

# `sbatch cluster/build.sbatch --skip-glove` stops before the 1.4 GB GloVe
# file. Only HateXplain reads it, so a Toy table is not worth waiting on a
# slow mirror for -- and `build.sbatch` can be submitted again later to fetch
# it, since everything above it is a no-op once the image exists.
SKIP_GLOVE=
if [ "${1:-}" = "--skip-glove" ]; then
  SKIP_GLOVE=1
fi

# Where the job is, and how long it has been there. There are no progress bars
# in this log: the output is a file rather than a terminal, so apptainer prints
# no bar and `mksquashfs` prints nothing at all. Elapsed time per step is what
# is left, and it is what a second run is estimated from -- a build that is
# twenty minutes into the conversion reads as working rather than as hung.
# `SECONDS` is the shell's own counter, started when the job did.
step() {
  printf '[+%02d:%02d] %s\n' "$((SECONDS / 60))" "$((SECONDS % 60))" "$*"
}

# Override SCRATCH to point somewhere else; the default is this account's
# scratch directory.
SCRATCH=${SCRATCH:-/scratch.hpc/${USER:-$(id -un)}}

# Where the corpora are kept between jobs. `PYHIGHLIGHTS_CACHE` and not
# `XDG_CACHE_HOME`, which the library does not read: it caches under
# `Path.home() / ".cache" / "pyhighlights"` unless that variable says
# otherwise, and apptainer binds the host's home into the container, so an
# unset variable puts corpora in a home directory that has a quota and has
# already been filled once by caches. `XDG_CACHE_HOME` is set as well, for
# whatever else in the image respects it.
export PYHIGHLIGHTS_CACHE="$SCRATCH/cache/pyhighlights"
export XDG_CACHE_HOME="$SCRATCH/cache"
mkdir -p "$PYHIGHLIGHTS_CACHE"

# Apptainer caches the layers it fetches under `$HOME/.apptainer` by default,
# and a home directory under quota fails the build after the download rather
# than before it:
#
#   FATAL: While performing build: conveyor failed to get: write
#   /home/<user>/.apptainer/cache/blob/oci-layout: disk quota exceeded
#
# The cache goes to scratch, where this account has room, and is kept rather
# than cleared: a rebuild that moves only a pin reuses the base image's layers
# instead of fetching them again.
export APPTAINER_CACHEDIR="$SCRATCH/apptainer/cache"
mkdir -p "$APPTAINER_CACHEDIR"

# The image is assembled on the node's own disk, not on scratch. Assembly
# unpacks the layers as ordinary files and squashes them back, tens of
# thousands of small writes: on scratch that ran at 1.4 GB in two and a half
# hours for a seven gigabyte image, the process at one percent of a core
# because it was waiting on the filesystem. That mount also reports `nodev`,
# so the device nodes cannot be unpacked there at all.
#
# Scratch is the fallback for a node too small to hold the unpacked image
# beside the SIF -- slow beats running out of room.
#
# One directory per job, removed when the job ends however it ends. The
# cluster's own guidance warns that two jobs sharing a node-local path will
# clean up under each other, and a shared name did exactly that: a cancelled
# build left nine gigabytes of unpacked image behind, and the next build
# measured the free space around it and fell back to scratch for no reason.
# A job id in the name makes the directory this job's to delete. Anything
# older than a day is a job that was killed outright, since nothing here
# outlives a job by design.
LOCAL_ROOT="${TMPDIR:-/tmp}/${USER:-$(id -un)}-apptainer"
mkdir -p "$LOCAL_ROOT"
find "$LOCAL_ROOT" -mindepth 1 -maxdepth 1 -mtime +0 -exec rm -rf {} + 2>/dev/null || true
LOCAL_TMP="$LOCAL_ROOT/${SLURM_JOB_ID:-$$}"
mkdir -p "$LOCAL_TMP"
AVAILABLE=$(df -Pk "$LOCAL_TMP" | awk 'NR==2 { print int($4 / 1048576) }')
if [ "${AVAILABLE:-0}" -ge 20 ]; then
  export APPTAINER_TMPDIR="$LOCAL_TMP"
else
  echo "local disk has ${AVAILABLE:-0}G, assembling on scratch instead (slow)" >&2
  # Per job here too. This is the branch that unpacks nine gigabytes onto the
  # share, and a killed job used to leave them there -- which is how the
  # quota that stopped the first build filled up.
  export APPTAINER_TMPDIR="$SCRATCH/apptainer/tmp/${SLURM_JOB_ID:-$$}"
fi
mkdir -p "$APPTAINER_TMPDIR"
trap 'rm -rf "$LOCAL_TMP" "$APPTAINER_TMPDIR"' EXIT

# Built from `cluster/env.def`, which holds the base tag and the pins, rather
# than pulled: the pins end up in the image instead of in a virtualenv every
# job has to find, and a build writes the finished SIF once where a cached
# pull writes it twice, on a filesystem measured here at 11.6 MB/s.
#
# `--ignore-fakeroot-command` is not optional. The user is not in
# `/etc/subuid`, so apptainer wraps `%post` in its own `fakeroot` and injects
# a host `libfakeroot.so` built against GLIBC 2.38 -- and the base image is
# Ubuntu 22.04, which has 2.35, so `%post` dies before its first line:
#
#   /bin/sh: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.38' not found
#     (required by /.singularity.d/libs/libfakeroot.so)
#
# The flag drops that wrapper and leaves the namespace's own root mapping,
# which is all `%post` needs now that it only runs pip.
#
# Three attempts, because the registry resets an HTTP/2 stream partway through
# the base image -- `stream error: stream ID 7; INTERNAL_ERROR; received from
# peer` -- and failing a queued job on someone else's reset is expensive.
step "building env.sif from cluster/env.def"
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
step "building the registry and fetching the toy corpus"
apptainer exec --nv \
  --bind "$PWD:$PWD" --bind "$SCRATCH:$SCRATCH" \
  --env PYTHONPATH="$PWD" \
  --env PYTHONUNBUFFERED=1 \
  --env PYHIGHLIGHTS_CACHE="$PYHIGHLIGHTS_CACHE" \
  --env XDG_CACHE_HOME="$XDG_CACHE_HOME" \
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

if [ -n "$SKIP_GLOVE" ]; then
  step "built, without glove; Toy needs none, HateXplain will"
  exit 0
fi

# GloVe, for HateXplain. 1.4 GB, so it is fetched once here rather than by
# five array jobs at the same time, and kept on scratch rather than in the
# image: the image is rebuilt whenever a pin moves, and this file never
# changes. The paper expects you to fetch it yourself, which is why the task
# takes a path and nothing in the library downloads it.
#
# The mirror first, and `nlp.stanford.edu` second, although Stanford is the
# source the paper names: it has served this file here at 30 kB/s, which is
# thirteen hours for the 1.4 GB. The mirror is Stanford's own upload to the
# Hugging Face hub, behind a CDN. A falling-back order rather than a
# speed-based switch because wget has no minimum-rate option: `--read-timeout`
# fires when a host stops sending, not when it trickles, so a slow host holds
# the job to its wall clock and the next URL is never reached.
GLOVE_URLS=(
  https://huggingface.co/stanfordnlp/glove/resolve/main/glove.twitter.27B.zip
  https://nlp.stanford.edu/data/glove.twitter.27B.zip
)
GLOVE_DIR="$SCRATCH/glove"
GLOVE="$GLOVE_DIR/glove.twitter.27B.25d.txt"
mkdir -p "$GLOVE_DIR"

if [ -f "$GLOVE" ]; then
  echo "glove already staged at $GLOVE"
else
  # Downloaded and unzipped on the node's own disk, then copied to scratch in
  # one pass. Scratch collapses under small writes: measured on this node,
  # `curl` writing its 16 kB buffers reached 41 kB/s there against 5.5 MB/s to
  # local disk over the same seconds and the same URL, while `dd bs=1M` on the
  # same filesystem reached 11.6 MB/s. So the download is not what should
  # touch scratch -- the finished file is, once, in megabyte blocks.
  LOCAL_GLOVE="$LOCAL_TMP/glove"
  mkdir -p "$LOCAL_GLOVE"
  # A part-downloaded archive from when this ran on scratch, and a half-copied
  # file from a job that was killed mid-`dd`. Neither can be resumed from
  # here, both are on a share with a quota, and nothing else removes them.
  rm -f "$GLOVE_DIR/glove.twitter.27B.zip" "$GLOVE.partial"
  # `-c` so a second URL resumes what the first left rather than starting the
  # 1.4 GB again; the directory is this job's, so a resubmitted job starts
  # over. A dot is a
  # megabyte and a line thirty-two of them: fifty lines rather than fifty
  # thousand, which is as close to a progress bar as a log file gets.
  #
  # `--read-timeout` covers the other failure, a host that accepts the
  # connection and then sends nothing: without it wget waits on that for the
  # rest of the job rather than returning to try the next URL.
  fetched=
  for url in "${GLOVE_URLS[@]}"; do
    step "fetching glove (1.4 GB) from $url"
    if wget -c --progress=dot:giga --read-timeout=60 --tries=2 \
      -O "$LOCAL_GLOVE/glove.twitter.27B.zip" "$url"; then
      fetched=1
      break
    fi
    step "that host did not finish it; trying the next"
  done
  if [ -z "$fetched" ]; then
    echo "glove: no host finished the download; rerun to resume" >&2
    exit 1
  fi
  step "unzipping glove"
  unzip -o -j "$LOCAL_GLOVE/glove.twitter.27B.zip" glove.twitter.27B.25d.txt \
    -d "$LOCAL_GLOVE"

  # Written under a temporary name and renamed, so an interrupted copy does
  # not leave a short file where the next job's `-f` check would take it for a
  # finished one. A rename within one filesystem is atomic.
  step "copying glove to scratch"
  dd if="$LOCAL_GLOVE/glove.twitter.27B.25d.txt" of="$GLOVE.partial" \
    bs=1M status=none
  mv "$GLOVE.partial" "$GLOVE"
  rm -rf "$LOCAL_GLOVE"
fi

# Stanford publishes no digest for the archive, so the check is on the shape
# of what came out: a truncated download unzips to a short file, and the wrong
# member gives the wrong width. 25 dimensions plus the token is 26 fields.
awk 'NR==1 { if (NF != 26) { print "glove: " NF " fields on line 1, expected 26" > "/dev/stderr"; exit 1 } }' "$GLOVE"
echo "glove staged: $(wc -l < "$GLOVE") vectors at $GLOVE"

step "built."
