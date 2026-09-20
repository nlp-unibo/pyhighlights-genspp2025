# genspp2025

A reproduction of Ruggeri and Signorelli, 2025, *Interlocking-free Selective
Rationalization Through Genetic-based Learning*, ACL 2025, built on
[pyhighlights](https://github.com/nlp-unibo/pyhighlights).

Two corpora, a synthetic one and HateXplain, against FR, MGR, MCD, G-RAT and
GenSPP.

[Paper](https://aclanthology.org/2025.acl-long.59/) ·
[Reference implementation](https://github.com/nlp-unibo/gen-spp) ·
[Library](https://github.com/nlp-unibo/pyhighlights)

## Layout

| | |
|---|---|
| `genspp2025/configurations/` | the paper's values, one module per kind of thing, one package per corpus |
| `published.py` | Tables 1 and 2, transcribed. Nothing computed |
| `compare.py` | a results tree beside those numbers, and what it cost to produce |
| `run.py` | run one cell, or a corpus's five |
| `cluster/` | the Slurm jobs, and the image they build |

## Installation

The reproduction requires Python 3.10 or later.

```bash
uv venv && uv pip install -e ".[dev]"
```

## Usage

```bash
uv run pytest                        # 15 tests, seconds
uv run python run.py toy --smoke     # one seed, one batch, minutes
```

`--smoke` bounds a baseline with `trainer_args`, which is what Lightning
reads.
The GenSPP cell does not train one model.
It runs a genetic search that trains a fresh predictor for every candidate it
scores.
That search builds its own trainer each time, so `trainer_args` never reaches
it.
`--smoke` gives the cell a smaller search instead: two candidates over one
generation, every other setting the paper's.

## Cluster

Submit from the repository root.
Neither job sets `--chdir`, so each runs in the directory it was submitted
from.
Both stage into `$SCRATCH`, which defaults to `/scratch.hpc/$USER` and can be
set to any other path before submitting.

```bash
mkdir -p logs
sbatch cluster/build.sbatch                       # image, registry, corpus, GloVe
sbatch --array=0 cluster/run.sbatch toy --smoke   # says every cell builds
sbatch cluster/run.sbatch toy                     # the five Toy cells
sbatch cluster/run.sbatch hatexplain              # the five HateXplain cells
python compare.py --results results
```

One job per model, five per corpus.
The jobs split by model because the cost differs by model and not by seed.
From the paper's appendix, a seed takes ~8 min for a baseline on Toy and
~36 min for GenSPP, ~4 and ~78 on HateXplain.
A table's four cheap cells therefore do not wait behind its slow one.

`run.sbatch` names `--partition=l40s`, since the cluster's default `sbuild` is
the image builder and has no GPU.
Override it per submission with `sbatch --partition=<name>
cluster/run.sbatch ...`.

`build.sbatch` builds the image from `cluster/env.def`, stages the toy corpus
under `$SCRATCH/cache/pyhighlights`, and fetches GloVe once for HateXplain.
`sbatch cluster/build.sbatch --skip-glove` stops before that 1.4 GB file, and
a later submission picks it up.
Neither job writes its working files on scratch, which is a network share:
each runs on the node's own disk and copies its results tree up when the job
ends, however it ends.

The search is a population of 50 over 100 generations
at a selection rate of 0.5 trains 5050 candidates per seed.
Nothing resumes and `results.json` is written once after the last seed, so a
cell killed on its fifth loses all five.

## Results

`compare.py` prints two tables per corpus.
The first is the paper's numbers beside the run's.
The second is what the run cost, from the `cost_` columns every seed writes,
which the paper reports none of.

| Column | What it means |
|---|---|
| `runtime/model` | Wall clock per model trained, which is the column to compare rows on. |
| `runtime/seed` | Wall clock per seed, which for a search counts every candidate it trained. |
| `inference/batch` | Milliseconds per batch at inference. |
| `inference/pass` | Wall clock for one pass over the test split. |
| `memory/peak` | High-water mark for the seed, the larger of this process and the largest single worker reaped. |
| `parameters` | Parameters of the model as it was scored, split into `trainable` and `frozen`. |
| `models trained` | Models a seed trained, which is one for a baseline and 5050 for a search. |
| `at once` | Candidates scored in parallel, one per worker. |

A baseline trains one model per seed.
GenSPP trains its founders and every generation's children, then reports the
winner, so its `runtime/seed` covers 5050 models where a baseline's covers one.
`runtime/model` divides that out as `runtime × at-once / models-trained`, and
for a baseline it is the seed's own wall clock.

`memory/peak` is a ceiling rather than a share.
It is not the sum of the workers, since summing would count a forked page once
per worker that never wrote to it.
There is no per-model memory to divide out either.
Read it as what a machine has to have.

A cell reads `-` where a model has not been run, or where the tree predates the
library reporting costs.

## Reproducibility

Table 1 reproduces.
All five models on both corpora, over the paper's five seeds
`[2023, 15451, 1337, 2001, 2080]`.
The four columns it reports are macro F1, token-level highlight F1, selection
rate and selection size.

Table 2 does not, except one row.
The skew experiment needs a selector pre-trained to select the first token and
then injected into the initial population.
`GenSPPTrainer` builds every founder at random, with no way to seed one.
`GenSPP (G = 150)` is the exception, since it is `n_generations=150` and
nothing else.
Tracked as pyhighlights' open point 7.

The `**` significance markers are Wilcoxon over seeds against the best
baseline, and nothing here computes them.

## Differences from the release

This reproduction was checked against the reference implementation file by
file, and the corpora, the training settings and the search parameters match
it.
Five differences remain, three of them in how the data is prepared and two in
how the search runs.
The table below is the whole of what a number produced here does not share
with a number produced there.
Each row is documented at the point it matters, in the configurations and in
the library's `docsrc/source/benchmarks.rst`.

| Difference | Here | In the release |
|---|---|---|
| HateXplain source | Parsed from upstream, 13507 rows, every one agreeing with the release on tokens, label and highlight | Read from the release's own pickles |
| Split scheme | One scheme serves all five models, so the five numbers are comparable to each other | A scheme per model |
| Validation rows | Held out of training | Trained on, by the genetic code |
| Batch order | One order per search, seen by every candidate | An order per candidate |
| Mutation | Uniform at 0.05, which explores the selector's decision threshold at 71% of the release's rate | Half that rate at the threshold |

## Corpora

There is no loader here for the toy corpus.
It is `pyhighlights.components.loaders.ToyLoader` with a `url`.
One loader generates, saves and reads, so a published toy corpus is a URL and
a digest in a configuration rather than a class somebody writes per dataset.

The artifact this configuration names is
[10.5281/zenodo.22828019](https://doi.org/10.5281/zenodo.22828019),
`pyhighlights-genspp-toy-v2.zip`.
It holds the corpus in the library's own columns, converted when the artifact
was built, so nothing converts it on the way in.
`pyhighlights/tools/build_datasets.py --skip-r2a` reproduces the published
bytes from the release's pickle, and
`genspp2025/configurations/toy/datasets.py` pins their digest.

`genspp2025/components/corpora.py` holds a `ReleasedToyLoader` for that
original pickle, which the first version of the record still carries and the
reference implementation ships.
That file stores `structure_indexes`, the positions a highlight marks, where
the library stores a vector, so the proxy fills in that column and hands the
rest to `ToyLoader`.

```python
from genspp2025.components.corpora import ReleasedToyLoader

splits = ReleasedToyLoader(url="toy_dataset.pkl").load()
```

Nothing registers it.
A test pins that it and a plain `ToyLoader` over the converted file return the
same rows.

HateXplain needs GloVe, which `build.sbatch` fetches.
`run.sbatch` refuses to start without it, and so does the task.
`requires_embeddings` exists because the registered HateXplain task once ran
without its vector file and reported numbers for a two-word vocabulary.

## Contact

Questions about a component or a result are best raised as an
[issue](https://github.com/nlp-unibo/pyhighlights-genspp2025/issues).
For anything else, write to Federico Ruggeri, federico.ruggeri6@unibo.it.
