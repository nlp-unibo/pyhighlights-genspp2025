# Interlocking-free Selective Rationalization Through Genetic-based Learning, ACL 2025

A reproduction of Ruggeri and Signorelli, 2025, built on
[pyhighlights](https://github.com/nlp-unibo/pyhighlights).

It runs two corpora, a synthetic one and HateXplain, against the five
architectures that paper compares.
A **cell** is one of those models on one corpus over the paper's five seeds,
and it is the unit everything here counts in.

[Paper](https://aclanthology.org/2025.acl-long.59/) ·
[Reference implementation](https://github.com/nlp-unibo/gen-spp) ·
[Library](https://github.com/nlp-unibo/pyhighlights)

## Architectures

Every model is defined in pyhighlights.
Paper's configurations are provided and registered in this repository.

| Model | Reference |
|---|---|
| **FR** | Liu, Wang, Wang, Li, Yue and Zhang, 2022, *FR: Folded Rationalization with a Unified Encoder*, NeurIPS 2022. [10.52202/068431-0504](https://doi.org/10.52202/068431-0504) |
| **MGR** | Liu, Wang, Wang, Li, Li, Zhang and Qiu, 2023, *MGR: Multi-generator Based Rationalization*, ACL 2023, pages 12771-12787. [10.18653/v1/2023.acl-long.715](https://doi.org/10.18653/v1/2023.acl-long.715) |
| **MCD** | Liu, Wang, Wang, Li, Deng, Zhang and Qiu, 2023, *D-Separation for Causal Self-Explanation*, NeurIPS 2023. [10.52202/075280-1890](https://doi.org/10.52202/075280-1890) |
| **G-RAT** | Hu and Yu, 2024, *Learning Robust Rationales for Model Explainability: A Guidance-Based Approach*, AAAI 2024, pages 18243-18251. [10.1609/aaai.v38i16.29783](https://doi.org/10.1609/aaai.v38i16.29783) |
| **GenSPP** | This paper. Ruggeri and Signorelli, 2025, ACL 2025, pages 1175-1191. [10.18653/v1/2025.acl-long.59](https://doi.org/10.18653/v1/2025.acl-long.59) |

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
uv run pytest                        # 16 tests, seconds
uv run python run.py toy --smoke     # one seed, one batch, minutes
```

`--smoke` shortens a baseline by setting `trainer_args`, which Lightning
reads.
The GenSPP cell does not train one model.
It runs a genetic search that trains a fresh predictor for every candidate it
scores.
That search builds its own trainer each time, so `trainer_args` never reaches
it.
`--smoke` gives the cell a smaller search instead: two candidates over one
generation, every other setting the paper's.

`--released-threshold` searches Toy GenSPP the way the released
implementation mutates rather than the way its paper reports it.
The release perturbs the selector's output bias at 0.10 where every other
gene takes 0.05, and the paper gives a single N(0.0, 0.05).
The flag is registered for Toy alone, and it writes where `--results` says,
which has to be somewhere other than the paper-faithful run: both carry the
same task name, and `MetricsAnalyzer` reports the newest run under a name.

```bash
uv run python run.py toy --model genspp --released-threshold --results results-released
```

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

One job per model, five per corpus, so a job produces a cell.
The jobs split by model because the cost differs by model and not by seed.
From the paper's appendix, a seed takes ~8 min for a baseline on Toy and
~36 min for GenSPP, ~4 and ~78 on HateXplain.
The four cheap cells of a corpus therefore do not wait behind its slow one.

`run.sbatch` names `--partition=l40s`, since the cluster's default `sbuild` is
the image builder and has no GPU.
Override it per submission with `sbatch --partition=<name>
cluster/run.sbatch ...`.

`build.sbatch` builds the image from `cluster/env.def`, stages the toy corpus
under `$SCRATCH/cache/pyhighlights`, and fetches GloVe once for HateXplain.
`sbatch cluster/build.sbatch --skip-glove` stops before that 1.4 GB file, and
a later submission picks it up.
Scratch is a network share, so neither job writes its working files there.
Each runs on the node's own disk and copies its results tree up when the job
ends, however it ends.

The search is a population of 50 over 100 generations at a selection rate of
0.5.
That is 5050 candidates trained per seed.
Nothing resumes and `results.json` is written once after the last seed, so a
cell killed on its fifth loses all five.
`--time=24:00:00` is the cluster's ceiling, and a GenSPP cell has to finish
inside it.

## Results

### Reproduced against published

Not filled yet.
The experiments are still running, and this section holds the published
numbers beside the reproduced ones once all ten cells have finished.
It reports Table 1, which is macro F1, token-level highlight F1, selection
rate and selection size, for each of the five models on each corpus.
Until then the numbers this repository stands behind are the published ones in
`published.py`.

`python compare.py --results results` produces it, since its first table per
corpus is this comparison.

### What `compare.py` prints

`compare.py` prints two tables per corpus.
The first is the paper's numbers beside the run's.
The second is what the run cost, taken from the `cost_` columns every seed
writes.
The paper reports none of it.

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
GenSPP trains its initial population and every generation's children, then
reports the winner.
Its `runtime/seed` therefore covers 5050 models, and a baseline's covers one.
`runtime/model` divides that out as `runtime × at-once / models-trained`, and
for a baseline it is the seed's own wall clock.

`memory/peak` is what a machine has to have for the seed, not what one model
uses.
It is not the sum of the workers, since summing would count a forked page once
per worker that never wrote to it.
There is no per-model memory to divide out either.

An entry reads `-` where a model has not been run, or where the tree predates
the library reporting costs.

## Reproducibility

Table 1 reproduces.
It runs all five models on both corpora, over the paper's five seeds
`[2023, 15451, 1337, 2001, 2080]`.
The four columns it reports are macro F1, token-level highlight F1, selection
rate and selection size.

Table 2 does not, except one row.
The skew experiment needs a selector pre-trained to select the first token and
then injected into the initial population.
`GenSPPTrainer` builds every founder at random, with no way to seed one.
`GenSPP (G = 150)` is the exception, since it is `n_generations=150` and
nothing else.
Tracked upstream as
[pyhighlights issue 124](https://github.com/nlp-unibo/pyhighlights/issues/124).

The `**` significance markers are Wilcoxon over seeds against the best
baseline, and nothing here computes them.

## Differences from the release

This reproduction was checked against the reference implementation file by
file, and the corpora, the training settings and the search parameters match
it.
Five differences remain, three of them in how the data is prepared and two in
how the search runs.
The table below is complete, so nothing outside it differs.
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

The toy corpus is read by `pyhighlights.components.loaders.ToyLoader` with a
`url`, so this repository defines no loader of its own.
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
the library stores a vector.
`ReleasedToyLoader` fills in that column and hands the rest to `ToyLoader`.

```python
from genspp2025.components.corpora import ReleasedToyLoader

splits = ReleasedToyLoader(url="toy_dataset.pkl").load()
```

`ReleasedToyLoader` is registered nowhere, so no configuration reaches it by
key.
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
