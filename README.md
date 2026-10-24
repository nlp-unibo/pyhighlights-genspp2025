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
| `truncated.py` | GenSPP's HateXplain highlight F1 as the release scores it |
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
The flag is registered for Toy alone.

```bash
uv run python run.py toy --model genspp --released-threshold
```

Every run writes `results/<task>/<stamp>`, whatever its flags.
`compare.py` reads each run's manifest and puts a run in the table only when
it ran the paper's seeds and a search registered for a published row.
A smoke run, a run of other seeds and a released-threshold search sit beside
the paper's run without entering the table.
Two runs of one row resolve to the newest, and `compare.py` names both on
stderr.

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

A run of other seeds is not a table row, so the table keeps reading the
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

This section reports Table 1, which is macro F1, token-level highlight F1,
selection rate and selection size, for each of the five models on each corpus.
Both corpora ran with pyhighlights `d245d98`.
Toy reads the artifact `pyhighlights-genspp-toy-v3.zip`, and HateXplain reads
`glove.twitter.27B.25d.txt`.
Every entry is a mean and a sample standard deviation over five seeds.
Clf-F1 is macro F1, Hl-F1 is token-level highlight F1, and R is the selection
rate, all in percent.
The selection size counts characters on Toy and words on HateXplain.
Each cell gives the reproduced value first and the published value second.
The G-RAT rows on both corpora come from runs that took the argmax of the
selection at evaluation.
G-RAT now samples it as the other baselines do, and those rows await a rerun.

| Toy | Clf-F1 | Hl-F1 | R | Size |
|---|---|---|---|---|
| FR | 91.47 ± 13.71 / 99.78 ± 0.20 | 30.15 ± 12.85 / 54.07 ± 4.02 | 14.99 ± 0.37 / 14.80 ± 0.24 | 3.00 ± 0.07 / 2.96 ± 0.05 |
| MGR | 99.81 ± 0.14 / 99.92 ± 0.04 | 50.74 ± 4.74 / 50.34 ± 11.23 | 14.70 ± 0.23 / 15.05 ± 0.80 | 2.94 ± 0.05 / 3.01 ± 0.16 |
| MCD | 99.86 ± 0.20 / 99.90 ± 0.04 | 63.35 ± 1.49 / 65.70 ± 3.76 | 15.11 ± 0.39 / 15.18 ± 0.17 | 3.02 ± 0.08 / 3.04 ± 0.03 |
| G-RAT | 99.67 ± 0.58 / 99.36 ± 0.82 | 55.92 ± 16.68 / 50.22 ± 7.78 | 14.32 ± 1.22 / 14.81 ± 0.51 | 2.86 ± 0.24 / 2.96 ± 0.10 |
| GenSPP | 99.56 ± 0.30 / 99.00 ± 0.25 | 66.00 ± 4.72 / 76.02 ± 0.64 | 8.42 ± 1.12 / 11.47 ± 0.49 | 1.68 ± 0.22 / 2.29 ± 0.08 |

MGR, MCD and G-RAT reproduce within about one published standard deviation
on highlight F1, selection rate and selection size.
Their macro F1 differs from the published one by at most 0.31 points.
For MGR, that gap of 0.11 points is 2.7 published standard deviations, since
the published spread is 0.04.
G-RAT reaches this mean with a wide spread, since its highlight F1 ranges from
39.5 to 79.3 across seeds.

FR does not reproduce.
Seed 2001 reaches a macro F1 of 67.3 and a highlight F1 of 14.7.
Seed 1337 classifies at 93.5 but reaches a highlight F1 of 17.7.
The other three seeds classify at 97.9 or above and reach a highlight F1
between 37.8 and 41.5, which is still below the published 54.07.
The released FR has not been rerun.
These runs therefore cannot tell whether the gap comes from this
implementation, from FR itself, or from the published numbers.

GenSPP classifies as published but selects fewer characters.
Its selection rate is 3.05 points below the published one, and its highlight
F1 is 10.02 points below.
The section on the release below explains this gap.

| HateXplain | Clf-F1 | Hl-F1 | R | Size |
|---|---|---|---|---|
| FR | 73.24 ± 1.35 / 72.14 ± 1.12 | 35.67 ± 0.64 / 31.15 ± 2.56 | 23.22 ± 1.04 / 25.55 ± 0.72 | 3.36 ± 0.14 / 3.46 ± 0.11 |
| MGR | 70.66 ± 1.21 / 71.14 ± 1.16 | 29.40 ± 1.33 / 29.38 ± 4.83 | 24.06 ± 0.72 / 25.30 ± 1.04 | 3.43 ± 0.11 / 3.42 ± 0.06 |
| MCD | 64.67 ± 9.12 / 70.37 ± 1.06 | 25.35 ± 6.69 / 27.92 ± 1.66 | 23.11 ± 1.67 / 25.07 ± 1.52 | 3.40 ± 0.23 / 3.50 ± 0.20 |
| G-RAT | 74.26 ± 0.71 / 73.85 ± 1.05 | 38.25 ± 0.99 / 36.17 ± 1.62 | 24.31 ± 0.57 / 24.68 ± 0.86 | 3.35 ± 0.08 / 3.34 ± 0.08 |
| GenSPP | 67.97 ± 1.69 / 69.71 ± 0.40 | 37.53 ± 3.40 / 42.62 ± 0.73 | 7.41 ± 0.63 / 6.51 ± 0.58 | 0.99 ± 0.07 / 0.75 ± 0.05 |

MGR and G-RAT reproduce within about one published standard deviation on
every column.
FR classifies as published, but its highlight F1 is 4.52 points above the
published one while its selection rate is 2.33 points below.

MCD reproduces on four of its five seeds.
Seed 2023 reaches a macro F1 of 48.4 and a highlight F1 of 14.2.
The other four seeds classify between 68.4 and 69.1 and reach a mean
highlight F1 of 28.1, against the published 27.92.

GenSPP does not reproduce.
Its macro F1 is 1.74 points below the published one, and its highlight F1 is
5.09 points below.
It selects more words than published, where on Toy it selects fewer.
Seeds 1, 2 and 4 reach a highlight F1 between 38.6 and 40.9, and seeds 3 and
5 reach 34.3 and 33.6.
No seed reaches the published 42.62.
The release scores the GenSPP highlight F1 differently on HateXplain, as the
table of reporting differences at the end of this README states.
It scores a test post only up to its first token unseen in training and
validation, which keeps 66.5% of the test tokens.
`truncated.py` scores the same checkpoints that way, and reaches a highlight
F1 of 40.80 ± 3.20.
The scoring therefore accounts for 3.27 of the 5.09 points.
The remaining 1.82 points are 2.5 published standard deviations, against a
spread of 3.20 over these five seeds.
The release has not been rerun on HateXplain, so these runs cannot tell
whether it reaches the published value under its own scoring.

`python compare.py --results results` produces this comparison, since its
first table per corpus is Table 1.
A GenSPP seed takes 27.4 minutes on Toy and 72.8 minutes on HateXplain, on
CPU workers.
That is 2.6 and 6.9 seconds per candidate trained, respectively.
A baseline seed takes 5.9 to 14.1 minutes on Toy and 8.0 to 14.2 minutes on
HateXplain, on average.

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
The reproduction row comes from pyhighlights `d245d98`, over the same seeds
1 to 5.
Every entry is a mean and a sample standard deviation over five seeds.
Clf-F1 is macro F1, Hl-F1 is token-level highlight F1, and R is the selection
rate, all in percent.

| GenSPP on Toy | Clf-F1 | Hl-F1 | R |
|---|---|---|---|
| Published | 99.00 ± 0.25 | 76.02 ± 0.64 | 11.47 ± 0.49 |
| Release, rerun | 99.61 ± 0.23 | 69.96 ± 0.70 | 9.94 ± 0.27 |
| This reproduction | 99.56 ± 0.30 | 66.00 ± 4.72 | 8.42 ± 1.12 |

The release does not reach the published highlight F1 either.
Its rerun is 6.1 points below the published value.
The released GenSPP code was last changed in August 2024.
This repository cannot tell whether the published numbers came from that code
under other seeds or another environment, or from an earlier version.

The reproduction is 4.0 points below the release, and its standard deviation
is nearly seven times larger over these five seeds.
Over twenty seeds, the release spreads as much as the reproduction does, as
the end of this section shows.
The spread comes from which code the search finds.
Every run selects a short code per class rather than the highlight, as the
next section shows.
The codes differ only in length.

| Seed | `aba` | `baa` | `abc` | Characters | Hl-F1 | R |
|---|---|---|---|---|---|---|
| 1 | `ba` | `a` | `b` | 4 | 62.53 | 7.52 |
| 2 | `ba` | `a` | `b` | 4 | 64.29 | 8.04 |
| 4 | `ba` | `a` | `b` | 4 | 61.17 | 7.35 |
| 3 | `ba` | `aa` | `b` | 5 | 70.55 | 9.27 |
| 5 | `ba` | `aa` | `b` | 5 | 71.48 | 9.90 |

A five-character code overlaps the gold highlight on one more character than
a four-character code.
That character is worth about eight points of highlight F1.
Once the task loss is below its limit, the fitness prefers the sparser of two
selections, so the four-character code is the better optimum of the search.
The highlight F1 is therefore lower in exactly the runs where the search
reaches a better fitness.

The release selects the five-character code `ba`, `aa` and `b` in all five of
its runs over seeds 1 to 5.
The reproduction selects the four-character code in three of them.

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
Its standard deviation of 4.12 is close to the reproduction's 4.72.
At that rate, five seeds miss the four-character code with a probability of
0.33.
The small spread of the rerun is therefore a property of seeds 1 to 5 rather
than of the release.
The reproduction reaches a highlight F1 of 66.00 ± 4.72 over seeds 1 to 5,
and the unchanged release reaches 68.64 ± 4.12 over seeds 1 to 20.
A two-sided exact Mann-Whitney test between the two gives p = 0.62.
The reproduction finds the four-character code in 3 of 5 runs, and the
release in 4 of 20.
A two-sided Fisher exact test gives p = 0.113.
Neither test detects a difference between the reproduction and the release,
although five runs detect only a large one.

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
| FR | 92.51 ± 11.22 | 8.94 ± 8.97 | 9.16 ± 8.97 |
| MGR | 99.80 ± 0.13 | 21.58 ± 10.15 | 21.78 ± 10.06 |
| MCD | 99.83 ± 0.22 | 55.75 ± 6.91 | 55.72 ± 6.90 |
| G-RAT | 99.67 ± 0.58 | 32.97 ± 20.45 | 33.24 ± 20.52 |
| GenSPP | 99.56 ± 0.30 | 3.23 ± 2.23 | 3.90 ± 2.57 |
| GenSPP, release rerun | 99.61 ± 0.23 | - | 6.59 ± 0.43 |

Every model except FR classifies at 99% or above, and FR classifies at 92.5%.
At most 56% of a model's selections contain the evidence, and only MCD
exceeds a third.
Every predictor therefore decodes the label from something other than the
whole highlight in at least 44% of the test examples.
The two views differ by less than one point, so selections that spell a
highlight by accident are rare.

`python selection.py --per-class` shows what the selections contain.
GenSPP selects a short code for each class in all of its runs.
In the reproduction, `aba` is marked by `ba` in 69% to 91% of its test
examples.
Every run marks `abc` by a lone `b`, in 73% to 94% of its examples.
Three runs mark `baa` by a lone `a` in 57% to 83% of its examples, and two
runs mark it by `aa` in 77% and 78%.
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
In 16 of their 20 runs, one class receives an empty selection in at least
90% of its test examples.
In 15 of those 16 runs, the other two classes receive 2.6 to 6.1 characters
on average, which keeps the average selection size near the sparsity target of
three.

The contamination rule therefore protects only against a predictor that has to
find the highlight itself.
A generator reads the whole text, so it can locate the highlight and pass on
any selection that names the class.
Highlight F1 and the complete-highlight rate measure different things as a
result.
Highlight F1 gives partial credit for partial overlap, while the
complete-highlight rate scores a selection as either complete or not.
Across the 20 baseline runs, the Pearson correlation between the two is 0.74.
GenSPP lies outside that trend, with a highlight F1 between 0.61 and 0.71 and
a mean complete-highlight rate of 3.23%.
It ranks first on highlight F1 and last on the complete-highlight rate.

The script also trains five independent readers per view.
Each reader is a GRU trained on windows of the training texts that contain the
gold highlight.
Its accuracy on a model's selections is not reported as a result.
Codes such as a lone `b` never occur in its training data, so each reader maps
them arbitrarily.
On GenSPP's selections, reader accuracy ranges from 21% to 81% across the
five reader seeds and both views.

These results cover Toy alone.
HateXplain has no designed evidence to check a selection against.

## Reproducibility

Table 1 can be rerun, and the results section compares its numbers with the
published ones.
The rerun covers all five models on both corpora over five seeds.
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
| G-RAT guide over padding | The guide target is scaled by its mean over words, and its cross entropy averages over words alone | The mean is taken over the padded width, and the cross entropy also scores padding against a target of zero (HateXplain) |
| MCD validation loss | One selection for every term | One sample for the predictor's cross entropy and another for sparsity, contiguity and the discrepancy |
| MGR predictor | No layer normalisation, as the original | A `LayerNorm` in no optimizer group, which still normalises with its initial weights |

Every baseline samples its selection at validation and test, as the release
does.
The original G-RAT thresholds its selection at 0.5 there instead
([`model/model_st.py`](https://github.com/shuaibo919/g-rat/blob/d279932/model/model_st.py)),
and its paper gives no reason for it.
This repository samples, so that the four baselines are scored under one
inference strategy.

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
| GenSPP metrics | The library's metrics, as for every baseline | Code of its own. Its `Binary Micro F1 Mask` is the highlight F1 the paper reports, and it pools true positives, false positives and false negatives over every test token as the library does. Its selection rate, `Sparsity`, is averaged over batch means. It computes no selection size |
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
