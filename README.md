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
| `selection.py` | what the selections of trained Toy models contain |
| `run.py` | run one cell, or a corpus's five |
| `cluster/` | the Slurm jobs, and the image they build |

## Installation

The reproduction requires Python 3.10 or later.

```bash
uv venv && uv pip install -e ".[dev]"
```

## Usage

```bash
uv run pytest                        # 19 tests, seconds
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
The jobs set no `--chdir`, so each runs in the directory it was submitted
from.
They stage into `$SCRATCH`, which defaults to `/scratch.hpc/$USER` and can be
set to any other path before submitting.

```bash
mkdir -p logs
sbatch cluster/build.sbatch                       # image, registry, corpora, GloVe
sbatch cluster/run.sbatch toy --smoke             # four baseline smokes on GPU
sbatch cluster/run-genspp.sbatch toy --smoke      # GenSPP smoke on CPU
sbatch cluster/run.sbatch toy                     # four Toy baselines
sbatch cluster/run-genspp.sbatch toy              # Toy GenSPP
sbatch cluster/run.sbatch hatexplain              # four HateXplain baselines
sbatch cluster/run-genspp.sbatch hatexplain       # HateXplain GenSPP
sbatch cluster/run-genspp.sbatch toy --seeds 1-20 # Toy GenSPP, seeds 1 to 20
python compare.py --results results
```

`--seeds` writes under `results/seeds-1-20/`, so the table keeps reading the
five-seed run.
Twenty GenSPP seeds on Toy take about nine hours, inside the 24-hour limit.

One job per model, five per corpus, so a job produces a cell.
The jobs split by model because the cost differs by model and not by seed.
From the paper's appendix, a seed takes ~8 min for a baseline on Toy and
~36 min for GenSPP, ~4 and ~78 on HateXplain.
The four cheap cells of a corpus therefore do not wait behind its slow one.

Both run scripts use `l40s`. The baseline array requests one GPU per job;
`run-genspp.sbatch` does not, because its eight candidates run in spawned CPU
worker processes, and the search uses processes only when every device is a
CPU. Override either partition
per submission with `sbatch --partition=<name> cluster/<script> ...` after
checking live cluster state.

`build.sbatch` builds the image from `cluster/env.def`, stages both corpora
under `$SCRATCH/cache/pyhighlights`, and fetches GloVe once for HateXplain.
`sbatch cluster/build.sbatch --skip-glove` stops before that 1.4 GB file, and
a later submission picks it up.
Scratch is a network share, so neither job writes its working files there.
Each runs on the node's own disk and copies its results tree up when the job
ends, however it ends.

The search is a population of 50 over 100 generations at a selection rate of
0.5.
That is 5050 candidates trained per seed.
`results.json` is updated after every finished seed, so an interrupted cell
keeps completed seeds and names them under `completed`. Runs cannot resume yet:
resubmitting starts every seed again. That is tracked in
[pyhighlights issue 134](https://github.com/nlp-unibo/pyhighlights/issues/134).
`--time=24:00:00` is the cluster's ceiling, and all five seeds of a GenSPP
cell have to finish inside it because a later job cannot resume them.

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
It is not the sum of the workers, since summing would count the shared corpus
storage once per worker.
There is no per-model memory to divide out either.

An entry reads `-` where a model has not been run, or where the tree predates
the library reporting costs.

### Toy: GenSPP beside its release

The reference implementation was rerun on Toy at commit `2b447a2`, with its
own configuration and its own seeds, 1 to 5.
The rerun used the versions that its `genetic/requirements.txt` pins, which
are torch 2.3.1 and numpy 2.0.0, on an L40S GPU.
The `cluster` branch of the reference repository holds the Slurm jobs that
ran it.
The reproduction row comes from pyhighlights `bd0ad26`, over the same seeds
1 to 5.
Every entry is a mean and a sample standard deviation over five seeds.
Clf-F1 is macro F1, Hl-F1 is token-level highlight F1, and R is the selection
rate, all in percent.

| GenSPP on Toy | Clf-F1 | Hl-F1 | R |
|---|---|---|---|
| Published | 99.00 ± 0.25 | 76.02 ± 0.64 | 11.47 ± 0.49 |
| Release, rerun | 99.61 ± 0.23 | 69.96 ± 0.70 | 9.94 ± 0.27 |
| This reproduction | 99.54 ± 0.31 | 65.09 ± 4.42 | 8.36 ± 0.97 |

The release does not reach the published highlight F1 either.
Its rerun is 6.1 points below the published value.
The released GenSPP code was last changed in August 2024.
This repository cannot tell whether the published numbers came from that code
under other seeds or another environment, or from an earlier version.

The reproduction is 4.9 points below the release, and its standard deviation
is six times larger over these five seeds.
Over twenty seeds, the release spreads as much as the reproduction does, as
the end of this section shows.
The spread comes from which code the search finds.
Every run selects a short code per class rather than the highlight, as the
next section shows.
The codes differ only in length.

| Seed | `aba` | `baa` | `abc` | Characters | Hl-F1 | R |
|---|---|---|---|---|---|---|
| 1 | `ba` | `a` | `b` | 4 | 60.82 | 7.54 |
| 3 | `ba` | `a` | `b` | 4 | 62.18 | 7.64 |
| 5 | `ba` | `a` | `b` | 4 | 62.77 | 7.79 |
| 2 | `ba` | `a` | `bc` | 5 | 69.24 | 9.50 |
| 4 | `ba` | `aa` | `b` | 5 | 70.46 | 9.32 |

A five-character code overlaps the gold highlight on one more character than
a four-character code.
That character is worth about seven points of highlight F1.
Once the task loss is below its limit, the fitness prefers the sparser of two
selections, so the four-character code is the better optimum of the search.
The highlight F1 is therefore lower in exactly the runs where the search
reaches a better fitness.

The release selects the five-character code `ba`, `aa` and `b` in all five of
its runs.
The reproduction found the four-character code in 8 of 15 runs: 3 of 5 here,
and 2 and 3 of 5 in two earlier runs at pyhighlights `07cbc16`.
Those two runs used the seeds `[2023, 15451, 1337, 2001, 2080]`, one at the
paper's mutation rate and one at the release's threshold rate.
The threshold rate and the batch order fixed in `bd0ad26` therefore do not
explain the difference.

The `probe/variance` branch of the reference repository reruns the release
over seeds 1 to 20 on an L40S GPU, unchanged and in two variants.
Each variant removes one difference in the table at the end of this README.
The first trains each candidate's predictor without the validation rows,
which the release trains on.
The second seeds each candidate's batch order by its chromosome, where the
release draws it from one generator seeded once per run.

| Release on Toy, seeds 1 to 20 | Clf-F1 | Hl-F1 | R | Four-character codes |
|---|---|---|---|---|
| Unchanged | 99.53 ± 0.28 | 68.64 ± 4.12 | 9.39 ± 1.19 | 4 of 20 |
| Without the validation rows | 99.30 ± 0.51 | 68.74 ± 2.99 | 9.79 ± 1.00 | 2 of 20 |
| Batch order seeded by the chromosome | 99.51 ± 0.34 | 69.00 ± 4.00 | 9.55 ± 1.09 | 5 of 20 |

Seeds 1 to 5 of the unchanged release give a highlight F1 of 70.09 ± 0.72,
which agrees with the rerun above.
Over twenty seeds, the unchanged release finds the four-character code in 4
runs.
Its standard deviation of 4.12 is close to the reproduction's 4.42.
At that rate, five seeds miss the four-character code with a probability of
0.33.
The small spread of the rerun is therefore a property of seeds 1 to 5 rather
than of the release.
The reproduction finds the four-character code in 8 of 15 runs, and the
release in 4 of 20.
A two-sided Fisher exact test gives p = 0.071.
The reproduction may still find it more often, and these runs cannot tell.

Neither variant changes the outcome.
Against the unchanged release, the highlight F1 does not move, with
Mann-Whitney p = 0.229 and 0.797.
Its spread does not move either, with Levene p = 0.516 and 0.904.
The share of four-character codes gives Fisher p = 0.661 and 1.000.
Training on the validation rows and the batch order therefore do not explain
the difference either.
A third variant copied the initial weights that a reused model slot is reset
to, and its twenty runs are byte-identical to the unchanged ones.
On a GPU the release already restores those weights, because moving a
candidate to the device gives its parameters new storage.

The release aggregates its runs with `np.mean` and `np.std`, and `np.std` is
the population standard deviation.
Over five runs, it is 0.894 times the sample standard deviation this README
reports, which cannot account for a spread six times smaller.
None of the twenty runs reaches the published highlight F1 of 76.02, and the
highest is 73.9.

### Toy: what the selections contain

The Toy corpus was designed so that only the whole highlight determines the
class.
Each class has one highlight, `aba`, `baa` or `abc`, and the generator was meant
to contaminate every text with fragments of the other two.
The end of this section measures how far the released corpus holds to that
design.
`selection.py` checks whether the test selections of trained models contain
that evidence.
It uses two measures, and neither consults the model's own predictor.

The complete-highlight rate is the share of test examples whose selection
contains the whole gold highlight.
In the positional view, every gold position has to be selected.
Additionally, no other class highlight may appear as a contiguous run of
selected characters.
The string view keeps the selected characters in text order without their
positions.
It is the only view available for the release, whose masks files store no
positions.

The complete-highlight rate is not the `sufficiency` of
`pyhighlights.components.faithfulness`.
That measure follows ERASER and compares the model's own predictions with and
without the rest of the input.
A predictor that has learned to decode a label from a partial selection then
reports that selection as sufficient.
The complete-highlight rate asks the corpus instead.

| Toy, test split | Own accuracy | Complete, positional | Complete, string |
|---|---|---|---|
| FR | 96.37 ± 6.50 | 11.31 ± 14.75 | 11.46 ± 14.68 |
| MGR | 99.45 ± 0.49 | 9.86 ± 10.58 | 10.02 ± 10.60 |
| MCD | 98.76 ± 1.95 | 22.63 ± 16.33 | 22.93 ± 16.41 |
| G-RAT | 99.77 ± 0.39 | 17.67 ± 22.21 | 17.69 ± 22.20 |
| GenSPP | 99.54 ± 0.32 | 2.24 ± 0.92 | 2.78 ± 1.23 |
| GenSPP, release rerun | 99.61 ± 0.23 | - | 6.59 ± 0.43 |

Every model classifies at 96% or above, while at most 23% of its selections
contain the evidence.
The predictors therefore decode the label from something other than the whole
highlight.
The two views differ by less than one point, so selections that spell a
highlight by accident are rare.

`python selection.py --per-class` shows what the selections contain.
GenSPP selects a short code for each class in all of its runs.
In the reproduction, `aba` is marked by `ba` in 81% to 88% of its test
examples.
Four runs mark `abc` by a lone `b` in 87% to 94% of its examples, and one run
marks it by `bc` in 86%.
Four runs mark `baa` by a lone `a` in 68% to 77% of its examples, and one run
marks it by `aa` in 80%.
The release uses the same code, with `b` in 72% to 82%, `ba` in 65% to 78% and
`aa` in 64% to 75%.
A lone `b` occurs in every highlight, so it is no evidence for `abc`.

The other two parts of that code are evidence under the corpus itself.
Outside the highlight, each of `ab`, `ba`, `aa` and `bc` occurs in only 1.7%
to 3.0% of the documents of every class.
That is the rate random filler produces, so the contamination adds almost no
fragments.
Of the first 8,000 documents, 89.5% of those containing `aa` are `baa`
documents, and 96.0% of those containing `bc` are `abc` documents.
A rule that reads `aa` as `baa`, `bc` as `abc` and anything else as `aba`
classifies 94.0% of the 2,000 test documents.
For `aba`, both contiguous fragments occur in a second highlight.
With its middle character hidden, `a?a` still marks `aba` in 85.6% of the
documents that contain it.
A lone `c` marks `abc` in 50.2% of the documents that contain it, against a
third by chance.
The design therefore holds only for `a` and `b`, which occur in every
highlight.
A code of `aa` or `bc` reads as evidence, and a lone `b` does not.

The baselines mostly mark a class by the size of the selection.
In 18 of their 20 runs, one class is selected as a single character in at
least 90% of its test examples.
That character is often outside the highlight.
The other two classes receive three to five characters, which keeps the
average selection size near the sparsity target of three.

The contamination rule therefore protects only against a predictor that has to
find the highlight itself.
A generator reads the whole text, so it can locate the highlight and pass on
any selection that names the class.
Highlight F1 and the complete-highlight rate measure different things as a
result.
Highlight F1 gives partial credit for partial overlap, while the
complete-highlight rate scores a selection as either complete or not.
Across the 20 baseline runs, the Pearson correlation between the two is 0.69.
GenSPP lies outside that trend, with a highlight F1 between 0.61 and 0.70 and
a mean complete-highlight rate of 2.24%.
It ranks first on highlight F1 and last on the complete-highlight rate.

The script also trains five independent readers per view.
Each reader is a GRU trained on windows of the training texts that contain the
gold highlight.
Its accuracy on a model's selections is not reported as a result.
Codes such as a lone `b` never occur in its training data, so each reader maps
them arbitrarily.
On GenSPP's selections, reader accuracy ranges from 21% to 76% across the
five reader seeds and both views.

These results cover Toy alone.
HateXplain has no designed evidence to check a selection against.

## Reproducibility

Table 1 reproduces.
It runs all five models on both corpora over five seeds.
The paper reports five seed runs and does not list their seeds.
The baselines use `[2023, 15451, 1337, 2001, 2080]`, which are the seeds of
the release's baselines.
GenSPP uses 1 to 5, which are the seeds of the release's genetic code.
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

Each model here follows its own paper and its authors' implementation.
GenSPP's original implementation is the gen-spp release itself.
The setup of Ruggeri and Signorelli (2025) is reproduced as published: corpora, splits, hyperparameters, seeds and metrics.
Where the release departs from an original implementation, the difference is documented here and not reproduced.
An error in an original implementation is documented in the library's model pages, under `docsrc/source/models/`.
[Issue 45](https://github.com/nlp-unibo/pyhighlights-genspp2025/issues/45) tracks every difference and how it was resolved.

### Data

| Difference | Here | In the release |
|---|---|---|
| HateXplain source | Parsed from upstream, 13507 rows, every one agreeing with the release on tokens, label and highlight | Read from the release's own pickles |
| Split scheme | One scheme serves all five models, so the five numbers are comparable to each other. Toy reads the release baselines' split from the artifact's `split` column | A scheme per model |
| Validation rows | Held out of training | Trained on, by the genetic code |
| HateXplain embeddings | Every token of every split has a row of its own: its GloVe vector, or a `uniform(-0.05, 0.05)` row from a fixed seed | The baselines map a token GloVe lacks to `<PAD>`, a zero row. GenSPP shares one uniform vector over such training tokens and embeds a token unseen in training as zeros |
| Toy one-hot table | 24 columns | 25 in the baselines, 26 in GenSPP |

### Baselines

| Difference | Here | In the release |
|---|---|---|
| Predictor pooling | Max over the kept positions alone | Max over every valid position, so the predictor reads the states of dropped words |
| Recurrence over padding | Packed, so a document is encoded alike in any batch | An unpacked bidirectional GRU, whose backward direction reads padding first (HateXplain) |
| Penalties over padding | Sparsity and contiguity over words alone | Over padding as well, since the sampled mask is not masked there (HateXplain) |
| G-RAT pretraining | The rationalizer steps from epoch 10, as the original does | From epoch 12 |
| G-RAT early stopping | Patience and checkpoints count from the end of pretraining, as in the original | From epoch 0 |
| MGR predictor | No layer normalisation, as the original | A `LayerNorm` in no optimizer group, which still normalises with its initial weights |

### GenSPP

| Difference | Here | In the release |
|---|---|---|
| Predictor pooling | Max over the kept positions alone | Max of the states multiplied by the selection over every position, so a dimension whose kept states are all negative pools to zero |
| Fitness | Cross entropy and selection rate weighted by sample | Averaged over per-batch means, so a partial last batch counts as a full one |
| Predictor initialisation | One initial state shared by the whole search | One model per slot, reset to its initial weights on a GPU but not on a CPU, where `state_dict()` aliases the live tensors |
| Random streams | Seeded streams of the library's own | Python `random` for roulette selection, `np.random` for survival and mutation |
| HateXplain numpy seed | Each run's own seed | `np.random.seed(seed=100)` in every run, inside `GloveEmbedder` |
| Compilation | Eager | `torch.compile` |
| Batch order | Reshuffled at every epoch, seeded by a hash of the chromosome | Reshuffled at every epoch by the global generator |
| Mutation | Gaussian with deviation 0.05 on every gene, as the paper reports. `--released-threshold` sets 0.10 on the threshold gene | Gaussian with deviation 0.10 on the threshold gene and 0.05 on every other gene |

The release pads every HateXplain row to the training maximum length, and a batch here is cut to its longest row.
This changes no number, because the release's GenSPP encoder is a unidirectional GRU and the pooling here reads kept positions alone.

### Reporting

| Difference | Here | In the release |
|---|---|---|
| Standard deviation | Sample deviation over seeds | Population deviation, `np.std` |
| GenSPP metrics | The library's metrics, as for every baseline | Code of its own: macro and binary micro class F1, per-document and pooled mask F1, a sparsity averaged over batch means, and no selection size |
| HateXplain highlight F1 | Every token of a document | A document up to its first all-zero embedding, so a test token unseen in training ends the scored part |

## Corpora

The toy corpus is read by `pyhighlights.components.loaders.ToyLoader` with a
`url`, so this repository defines no loader of its own.
One loader generates, saves and reads, so a published toy corpus is a URL and
a digest in a configuration rather than a class somebody writes per dataset.

The artifact this configuration names is
[10.5281/zenodo.23570345](https://doi.org/10.5281/zenodo.23570345),
`pyhighlights-genspp-toy-v3.zip`.
It holds the corpus in the library's own columns, converted when the artifact
was built, so nothing converts it on the way in.
It also stores each row's split in a `split` column, which holds the released baselines' rows.
The first 80% is train, a fifth of it is held out under seed 15000, and the rest is test.
The split is therefore data, not a draw that depends on a library version.
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
