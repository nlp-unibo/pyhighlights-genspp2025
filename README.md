# genspp2025

A reproduction of Ruggeri and Signorelli, 2025, *Interlocking-free Selective
Rationalization Through Genetic-based Learning*, ACL 2025, built on
[pyhighlights](https://github.com/nlp-unibo/pyhighlights).

- Paper: <https://aclanthology.org/2025.acl-long.59/>
- Reference implementation: <https://github.com/nlp-unibo/gen-spp>

Two corpora — a synthetic one and HateXplain — against FR, MGR, MCD, G-RAT and
GenSPP. **This repository is configuration.** Every component it names is the
library's; what lives here is the paper's experimental design, the numbers it
published, and what it takes to run them on a cluster.

## Layout

| | |
|---|---|
| `genspp2025/configurations/` | the paper's values, one module per kind of thing, one package per corpus |
| `published.py` | Tables 1 and 2, transcribed. Nothing computed |
| `compare.py` | a results tree beside those numbers, and what it cost to produce |
| `run.py` | run one cell, or a corpus's five |
| `cluster/` | the Slurm jobs, and the image they build |

Nothing registers on import. A run builds the registry over this package with
the library beside it, which is why `pyhighlights` is installed on the cluster
and this repository is mounted rather than installed: an edit runs without
staging anything again, and the environment never holds a second copy of every
registration.

## Locally

```bash
uv venv && uv pip install -e ".[dev]"
uv run pytest                        # 15 tests, seconds
uv run python run.py toy --smoke     # one seed, one batch, minutes
```

`--smoke` bounds a baseline with `trainer_args`, which is what Lightning
reads. A search reads none of it — it builds a trainer per candidate — so the
GenSPP cell takes a search of its own instead: two candidates, one generation,
every other setting the paper's. Without it that cell ran the full fifty
candidates over a hundred generations under a flag that promises minutes.

## On the cluster

Submit from the repository root: neither job sets `--chdir`, so each runs in
the directory it was submitted from. Both stage into `$SCRATCH`, which
defaults to `/scratch.hpc/$USER` and can be set to any other path before
submitting.

```bash
mkdir -p logs
sbatch cluster/build.sbatch                       # image, registry, corpus, GloVe
sbatch --array=0 cluster/run.sbatch toy --smoke   # says every cell builds
sbatch cluster/run.sbatch toy                     # the five Toy cells
sbatch cluster/run.sbatch hatexplain              # the five HateXplain cells
python compare.py --results results
```

One job per model, five per corpus. From the paper's appendix, a seed takes
~8 min for a baseline on Toy and ~36 min for GenSPP, ~4 and ~78 on HateXplain
— so GenSPP is hours where a baseline is minutes, and splitting per model
keeps a table off the slowest cell's critical path.

### The cost table

`compare.py` prints a second table per corpus: runtime, inference time, memory
and parameters, from the `cost_` columns every seed writes. The paper reports
none of it, so there is no published half — it is filled by running the
experiments.

```
=== toy — cost ===
 model runtime/model runtime/seed inference/batch inference/pass memory/peak parameters trainable frozen models trained at once
    fr  0.45s ± 0.00 0.45s ± 0.00       4.1 ± 0.0   0.17s ± 0.00     888 ± 0       2.3k      1.7k    600              1       1
   mcd  0.57s ± 0.00 0.57s ± 0.00       5.9 ± 0.0   0.23s ± 0.00     891 ± 0       4.6k      3.4k   1.2k              1       1
genspp  2.27s ± 0.00 2.27s ± 0.00       2.6 ± 0.0   0.13s ± 0.00     898 ± 0       2.9k       859   2.0k              4       4
```

**`runtime/model` is the column to compare rows on.** A baseline trains one
model per seed; GenSPP trains its founders plus every generation's children,
several at a time, and reports the winner — so `runtime/seed` would say a
search is as cheap as the machine that ran it. The per-model figure is
`runtime × at-once / models-trained`, which for a baseline is its own wall
clock. `inference/batch` is in milliseconds.

**`memory/peak` is a ceiling, not a share.** It is the process high-water mark
for the whole seed, and a search scores its candidates on threads of one
process, so there is no per-model memory to divide out: most of the peak is
resident before the first candidate exists. Read it as what a machine has to
have, not as what a model uses.

The parameter counts are of the model as it was scored. The toy backbone's
one-hot table is frozen by construction, which is why `frozen` is never zero,
and GenSPP's generator is frozen too — the search settled it and descent never
moved it, so `trainable` is the predictor alone. A cell reads `-` where a model
has not been run or the tree predates the library reporting costs.

A search scores eight candidates at once, one per worker, which is the
released implementation's pool and the job's `--cpus-per-task=8`. It is worth
less than eight times: a candidate is small, so most of its cost is building a
Lightning trainer and stepping it from Python. Eight candidates of the Toy
search, measured on a 24-core machine — 14.1 s on one worker, 10.0 s on eight,
7.4 s on eight with torch held to a thread each.

### What `build.sbatch` stages

**The image.** Built from `cluster/env.def`, which holds the base tag and the
pins, so a job needs nothing on scratch but its results. A build rather than
an `apptainer pull` also writes the finished SIF once where a cached pull
writes it twice, and scratch here measures 11.6 MB/s sequential.

`--ignore-fakeroot-command` is not optional. Without it apptainer wraps
`%post` in its own `fakeroot` and injects the host's `libfakeroot.so`, so a
host newer than the base image fails before pip is reached:

```
/bin/sh: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.38' not found
(required by /.singularity.d/libs/libfakeroot.so)
```

Every `pytorch/pytorch` runtime tag is Ubuntu 22.04 with glibc 2.35, so no tag
of that base answers a host needing 2.38. The flag drops the wrapper and
leaves the namespace's own root mapping, which is all `%post` needs now that
it only runs pip.

The layers are cached on scratch but the image is **assembled on the node's
own disk**, because assembly unpacks the whole image as ordinary files and
squashes them back: on `/scratch.hpc` that managed 1.4 GB of a 7 GB image in
two and a half hours, the process at one percent of a core waiting on the
filesystem. A node with less than 20 GB free falls back to scratch and says
so, and the directory is cleared first — a cancelled build leaves nine
gigabytes of unpacked image behind, and the next one would measure the free
space around it.

The build gets three attempts, for the registry resetting an HTTP/2 stream
partway through the base image (`stream error: stream ID 7; INTERNAL_ERROR;
received from peer`).

Each step logs the elapsed time since the job started, as `[+12:34]`. There
are no progress bars — the output is a file, so apptainer prints no bar and
`mksquashfs` prints nothing at all — and the stamps are what tells a
conversion that is working from one that is hung. GloVe is the exception:
`wget` dots it, a megabyte a dot and thirty-two to a line.

**GloVe**, for HateXplain. 1.4 GB, fetched once into `$SCRATCH/glove` rather
than by five array jobs at the same time, and kept beside the image rather
than inside it: the image is rebuilt whenever a pin moves, and this file never
changes. Toy needs none of it, so `sbatch cluster/build.sbatch --skip-glove`
stops before it and a later submission picks it up — everything above it is a
no-op once the image exists.

It is fetched from Stanford's own upload to the Hugging Face hub before
`nlp.stanford.edu`, which has served it here at 30 kB/s: thirteen hours for
the 1.4 GB. The order is a fallback rather than a race, because `wget` has no
minimum-rate option — `--read-timeout` fires on a host that stops sending,
not on one that trickles. Stanford publishes no digest, so the check is on the shape
of what came out — 25 dimensions plus the token is 26 fields on line one.

`run.sbatch` still refuses to start without it, and so does the task:
`requires_embeddings` exists because the registered HateXplain task once ran
without its vector file and reported numbers for a two-word vocabulary.

**The toy corpus** is fetched from Zenodo
([10.5281/zenodo.22828019](https://doi.org/10.5281/zenodo.22828019), CC-BY-4.0,
released by both authors) and digest-verified. `build.sbatch` does it once
while a person is reading the log, rather than five array jobs at once on a
node that may have no outbound network.

## What is reproducible, and what is not

**Table 1 is.** All five models on both corpora, the paper's five seeds
`[2023, 15451, 1337, 2001, 2080]`, and the four columns it reports — macro F1,
token-level highlight F1, selection rate and selection size.

**Table 2 is not, except one row.** The skew experiment needs a selector
pre-trained to select the first token and then injected into the initial
population, and `GenSPPTrainer` builds every founder at random with no way to
seed one. `GenSPP (G = 150)` is the exception: it is `n_generations=150` and
nothing else. Tracked as pyhighlights' open point 7.

The `**` significance markers are Wilcoxon over seeds against the best
baseline. Nothing here computes them.

## Where this is not the release

Checked against the reference implementation file by file. The corpora, the
training settings and the search parameters match it; the differences are
documented at the point they matter, in the configurations and in the
library's `docsrc/source/benchmarks.rst`. In short: HateXplain is parsed from
upstream rather than from the release's pickles (13507 rows either way, every
one agreeing on tokens, label and highlight); one split scheme serves all five
models, so the five numbers are comparable to each other; validation is held
out of training, where the released genetic code trains on all of train; every
candidate of a search sees one batch order; and mutation is uniform at 0.05,
which explores the selector's decision threshold at 71% of the release's rate
rather than half of it.

## The corpus, and the proxy

There is no loader here for the toy corpus. It is
`pyhighlights.components.loaders.ToyLoader` with a `url`: one loader
generates, saves and reads, so a published toy corpus is a URL and a digest in
a configuration rather than a class somebody writes per dataset.

The artifact this configuration names holds the corpus in the library's own
columns, converted when the artifact is built, so nothing converts it on the
way in.

It is [10.5281/zenodo.22828019](https://doi.org/10.5281/zenodo.22828019),
`pyhighlights-genspp-toy-v2.zip`, the second version of the record — the first
holds the release's own pickle. `pyhighlights/tools/build_datasets.py
--skip-r2a` reproduces the published bytes from that pickle, and
`genspp2025/configurations/toy/datasets.py` pins their digest.

`genspp2025/components/corpora.py` holds a `ReleasedToyLoader` for the
**original** pickle — what the published record still carries, what the
reference implementation ships, and what a copy made before the conversion
is. That file stores `structure_indexes`, the positions a highlight marks,
where the library stores a vector; the proxy fills in that column and hands
the rest to `ToyLoader`:

```python
from genspp2025.components.corpora import ReleasedToyLoader

splits = ReleasedToyLoader(url="toy_dataset.pkl").load()
```

Nothing registers it. A test pins that it and a plain `ToyLoader` over the
converted file return the same rows — otherwise converting the artifact
changed the corpus rather than its serialisation.
