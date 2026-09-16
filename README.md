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
| `compare.py` | a results tree beside those numbers |
| `run.py` | run one cell, or a corpus's five |
| `cluster/` | the Apptainer image and the Slurm jobs |

Nothing registers on import. A run builds the registry over this package with
the library beside it, which is why the container installs `pyhighlights` and
mounts this repository rather than installing it: an edit runs without a
rebuild, and the container never holds a second copy of every registration.

## Locally

```bash
uv venv && uv pip install -e ".[dev]"
uv run pytest                        # 11 tests, seconds
uv run python run.py toy --smoke     # one seed, one batch, minutes
```

## On the cluster

```bash
mkdir -p logs
sbatch cluster/build.sbatch                       # image, registry, toy corpus
sbatch --array=0 cluster/run.sbatch toy --smoke   # says every cell builds
sbatch cluster/run.sbatch toy                     # the five Toy cells
sbatch cluster/run.sbatch hatexplain              # the five HateXplain cells
python compare.py --results results
```

One job per model, five per corpus. From the paper's appendix, a seed takes
~8 min for a baseline on Toy and ~36 min for GenSPP, ~4 and ~78 on HateXplain
— so GenSPP is hours where a baseline is minutes, and splitting per model
keeps a table off the slowest cell's critical path.

### Two things to stage first

**GloVe**, for HateXplain only. A 1.4 GB download the paper expects you to
fetch, so nothing here fetches it:

```bash
mkdir -p $SCRATCH/glove && cd $SCRATCH/glove
wget https://nlp.stanford.edu/data/glove.twitter.27B.zip
unzip -j glove.twitter.27B.zip glove.twitter.27B.25d.txt
```

`run.sbatch` refuses rather than starting without it, and so does the task:
`requires_embeddings` exists because the registered HateXplain task once ran
without its vector file and reported numbers for a two-word vocabulary.

**The toy corpus** is fetched from Zenodo
([10.5281/zenodo.22711449](https://doi.org/10.5281/zenodo.22711449), CC-BY-4.0,
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

## A note on `GenSPPToyLoader`

It lives in the library, not here — it reads a published corpus with a DOI,
like the library's other loaders, and the library's own `tools/build_datasets.py`
builds that Zenodo record. Until pyhighlights relocates it, the import is
`pyhighlights_benchmarks.genspp2025.corpora`.
