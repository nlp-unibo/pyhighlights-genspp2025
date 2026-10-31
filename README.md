# Interlocking-free Selective Rationalization Through Genetic-based Learning, ACL 2025

A reproduction of Ruggeri and Signorelli, 2025, built on [pyhighlights](https://github.com/nlp-unibo/pyhighlights).
It runs the five architectures that paper compares on two corpora, a synthetic one called Toy and HateXplain.
A **cell** is one of those models on one corpus over the paper's five seeds, and it is the unit everything here counts in.
The **release** is the authors' implementation, [gen-spp](https://github.com/nlp-unibo/gen-spp).

[Paper](https://aclanthology.org/2025.acl-long.59/) ·
[Release](https://github.com/nlp-unibo/gen-spp) ·
[Library](https://github.com/nlp-unibo/pyhighlights)

## Contents

1. [Architectures](#architectures)
2. [Layout](#layout)
3. [Usage](#usage)
4. [Cluster](#cluster)
5. [Results](#results)
   1. [Table 1: published, release and reproduction](#table-1-published-release-and-reproduction)
   2. [Toy](#toy)
   3. [HateXplain](#hatexplain)
   4. [GenSPP on Toy over twenty seeds](#genspp-on-toy-over-twenty-seeds)
   5. [Toy: what the selections contain](#toy-what-the-selections-contain)
   6. [Cost](#cost)
6. [Reproducibility](#reproducibility)
7. [Differences from the release](#differences-from-the-release)
8. [Corpora](#corpora)
9. [Contact](#contact)

## Architectures

Every model is defined in pyhighlights.
This repository registers the paper's configurations of them.

| Model | Reference |
|---|---|
| **FR** | Liu, Wang, Wang, Li, Yue and Zhang, 2022, *FR: Folded Rationalization with a Unified Encoder*, NeurIPS 2022. [10.52202/068431-0504](https://doi.org/10.52202/068431-0504) |
| **MGR** | Liu, Wang, Wang, Li, Li, Zhang and Qiu, 2023, *MGR: Multi-generator Based Rationalization*, ACL 2023, pages 12771-12787. [10.18653/v1/2023.acl-long.715](https://doi.org/10.18653/v1/2023.acl-long.715) |
| **MCD** | Liu, Wang, Wang, Li, Deng, Zhang and Qiu, 2023, *D-Separation for Causal Self-Explanation*, NeurIPS 2023. [10.52202/075280-1890](https://doi.org/10.52202/075280-1890) |
| **G-RAT** | Hu and Yu, 2024, *Learning Robust Rationales for Model Explainability: A Guidance-Based Approach*, AAAI 2024, pages 18243-18251. [10.1609/aaai.v38i16.29783](https://doi.org/10.1609/aaai.v38i16.29783) |
| **GenSPP** | This paper. Ruggeri and Signorelli, 2025, ACL 2025, pages 1175-1191. [10.18653/v1/2025.acl-long.59](https://doi.org/10.18653/v1/2025.acl-long.59) |

## Layout

| Path | Contents |
|---|---|
| `genspp2025/configurations/` | The paper's values, one module per kind of thing, one package per corpus |
| `published.py` | Tables 1 and 2, transcribed. Nothing is computed |
| `compare.py` | A results tree beside the published numbers, and what it cost to produce |
| `selection.py` | What the selections of trained Toy models contain |
| `truncated.py` | GenSPP's HateXplain highlight F1, scored as the release scores it |
| `run.py` | Runs one cell, or the five cells of a corpus |
| `cluster/` | The Slurm jobs, and the image they build |

## Usage

The reproduction requires Python 3.10 or later.

```bash
uv venv && uv pip install -e ".[dev]"
uv run pytest                                          # 31 tests, seconds
uv run python run.py toy --smoke                       # one seed, one batch, minutes
uv run python run.py toy --model genspp                # one cell
uv run python run.py hatexplain --embeddings glove.twitter.27B.25d.txt
uv run python run.py toy --model genspp --seeds 1-20   # other seeds
uv run python run.py toy --model genspp --g150         # Table 2, GenSPP (G = 150)
uv run python run.py toy --model genspp --released-threshold
```

`--smoke` shortens a baseline through `trainer_args`, which Lightning reads.
The GenSPP cell runs a genetic search that builds a trainer for every candidate, so `trainer_args` never reaches it.
`--smoke` gives that cell a smaller search instead: two candidates over one generation, every other setting the paper's.

`--released-threshold` mutates Toy GenSPP as the release does, not as the paper reports.
The release perturbs the selector's output bias with a deviation of 0.10 and every other gene with 0.05.
The paper gives a single N(0.0, 0.05).
The flag is registered for Toy alone.

Every run writes `results/<task>/<stamp>`, whatever its flags.
`compare.py` reads each run's manifest and puts a run in the table only when it ran the paper's seeds and a search registered for a published row.
A smoke run, a run of other seeds and a released-threshold search therefore stay out of the table.
Two runs of one row resolve to the newest, and `compare.py` names both on stderr.

## Cluster

Submit from the repository root, since the jobs set no `--chdir`.
The jobs stage into `$SCRATCH`, which defaults to `/scratch.hpc/$USER` and can be set before submitting.

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

Each job runs one model, so each job produces one cell.
The jobs split by model because the cost differs by model and not by seed.
The paper's appendix gives about 8 minutes per seed for a baseline on Toy and 36 for GenSPP, and about 4 and 78 on HateXplain.
Twenty GenSPP seeds on Toy take about nine hours.

| Job | Partition | Resources |
|---|---|---|
| `build.sbatch` | `sbuild` | Builds the image from `cluster/env.def`, stages both corpora under `$SCRATCH/cache/pyhighlights`, and fetches GloVe once. `--skip-glove` stops before that 1.4 GB file |
| `run.sbatch` | `l40s` | One GPU per baseline job |
| `run-genspp.sbatch` | `l40s` | No GPU. Eight candidates are scored at once in spawned CPU worker processes, which the search uses only when every device is a CPU |

`sbatch --partition=<name> cluster/<script> ...` overrides a partition for one submission, after checking live cluster state.
Scratch is a network share, so each job works on the node's own disk and copies its results tree up when it ends, however it ends.

The search is a population of 50 over 100 generations at a selection rate of 0.5, which trains 5050 candidates per seed.
`results.json` is rewritten after every finished seed, and it lists the finished seeds under `completed`.
A run cannot resume, so a resubmitted cell starts every seed again ([pyhighlights issue 134](https://github.com/nlp-unibo/pyhighlights/issues/134)).
All five seeds of a GenSPP cell therefore have to finish inside `--time=24:00:00`, the cluster's ceiling.

## Results

Every entry is a mean and a sample standard deviation over five seeds.
**Clf-F1** is macro F1, **Hl-F1** is token-level highlight F1, and **R** is the selection rate, all in percent.
**Size** is the selection size, in characters on Toy and in words on HateXplain.
**Published** is Table 1 of the paper.
**Release** is the release rerun with its own configurations and seeds, as [Reproducibility](#reproducibility) describes.
**Here** is this reproduction, run with pyhighlights `d245d98`.
Toy reads the artifact `pyhighlights-genspp-toy-v3.zip`, and HateXplain reads `glove.twitter.27B.25d.txt`.
`python compare.py --results results` prints the published and reproduced columns.

### Table 1: published, release and reproduction

| Toy | | Clf-F1 | Hl-F1 | R | Size |
|---|---|---|---|---|---|
| **FR** | Published | 99.78 ± 0.20 | 54.07 ± 4.02 | 14.80 ± 0.24 | 2.96 ± 0.05 |
| | Release | 99.90 ± 0.06 | 55.17 ± 4.35 | 14.70 ± 0.51 | 2.94 ± 0.10 |
| | Here | 91.47 ± 13.71 | 30.15 ± 12.85 | 14.99 ± 0.37 | 3.00 ± 0.07 |
| **MGR** | Published | 99.92 ± 0.04 | 50.34 ± 11.23 | 15.05 ± 0.80 | 3.01 ± 0.16 |
| | Release | 99.76 ± 0.26 | 48.55 ± 16.25 | 15.09 ± 1.16 | 3.02 ± 0.23 |
| | Here | 99.81 ± 0.14 | 50.74 ± 4.74 | 14.70 ± 0.23 | 2.94 ± 0.05 |
| **MCD** | Published | 99.90 ± 0.04 | 65.70 ± 3.76 | 15.18 ± 0.17 | 3.04 ± 0.03 |
| | Release | 99.92 ± 0.04 | 66.17 ± 4.46 | 14.96 ± 0.26 | 2.99 ± 0.05 |
| | Here | 99.86 ± 0.20 | 63.35 ± 1.49 | 15.11 ± 0.39 | 3.02 ± 0.08 |
| **G-RAT** | Published | 99.36 ± 0.82 | 50.22 ± 7.78 | 14.81 ± 0.51 | 2.96 ± 0.10 |
| | Release | 99.51 ± 0.84 | 47.26 ± 8.92 | 15.08 ± 0.29 | 3.02 ± 0.06 |
| | Here | 98.81 ± 2.21 | 50.35 ± 11.69 | 14.35 ± 0.33 | 2.87 ± 0.07 |
| **GenSPP** | Published | 99.00 ± 0.25 | 76.02 ± 0.64 | 11.47 ± 0.49 | 2.29 ± 0.08 |
| | Release | 99.61 ± 0.23 | 69.96 ± 0.69 | 9.94 ± 0.27 | - |
| | Here | 99.56 ± 0.30 | 66.00 ± 4.72 | 8.42 ± 1.12 | 1.68 ± 0.22 |

| HateXplain | | Clf-F1 | Hl-F1 | R | Size |
|---|---|---|---|---|---|
| **FR** | Published | 72.14 ± 1.12 | 31.15 ± 2.56 | 25.55 ± 0.72 | 3.46 ± 0.11 |
| | Release | 72.01 ± 1.40 | 32.47 ± 3.32 | 24.46 ± 0.33 | 3.37 ± 0.05 |
| | Here | 73.24 ± 1.35 | 35.67 ± 0.64 | 23.22 ± 1.04 | 3.36 ± 0.14 |
| **MGR** | Published | 71.14 ± 1.16 | 29.38 ± 4.83 | 25.30 ± 1.04 | 3.42 ± 0.06 |
| | Release | 72.56 ± 0.66 | 20.79 ± 13.78 | 25.20 ± 0.74 | 3.50 ± 0.14 |
| | Here | 70.66 ± 1.21 | 29.40 ± 1.33 | 24.06 ± 0.72 | 3.43 ± 0.11 |
| **MCD** | Published | 70.37 ± 1.06 | 27.92 ± 1.66 | 25.07 ± 1.52 | 3.50 ± 0.20 |
| | Release | 71.75 ± 0.83 | 28.24 ± 1.76 | 24.23 ± 0.52 | 3.38 ± 0.14 |
| | Here | 64.67 ± 9.12 | 25.35 ± 6.69 | 23.11 ± 1.67 | 3.40 ± 0.23 |
| **G-RAT** | Published | 73.85 ± 1.05 | 36.17 ± 1.62 | 24.68 ± 0.86 | 3.34 ± 0.08 |
| | Release | 72.41 ± 0.78 | 37.24 ± 1.06 | 24.93 ± 0.82 | 3.40 ± 0.08 |
| | Here | 74.20 ± 0.80 | 37.31 ± 1.09 | 24.80 ± 0.71 | 3.45 ± 0.08 |
| **GenSPP** | Published | 69.71 ± 0.40 | 42.62 ± 0.73 | 6.51 ± 0.58 | 0.75 ± 0.05 |
| | Release | 69.58 ± 1.95 | 43.09 ± 2.73 ¹ | 7.07 ± 1.27 | - |
| | Here | 67.97 ± 1.69 | 37.53 ± 3.40 | 7.41 ± 0.63 | 0.99 ± 0.07 |

¹ Scored as the release scores it, which [HateXplain](#hatexplain) compares.

The release computes no selection size for GenSPP, and its GenSPP selection rate averages batch means.
The release's own files store the population deviation, which is recomputed here as the sample deviation.

A collapsed seed here does not correspond to a collapsed seed in the release.
A seed fixes a random stream, and the two code bases draw from it in a different order and over different shapes.
The Toy one-hot table, the HateXplain vocabulary and the batch order all differ.
Each code base collapses on runs of its own: this reproduction on FR on Toy and on MCD on HateXplain, the release on MGR on HateXplain.

### Toy

MGR, MCD and G-RAT reproduce within about one published standard deviation on Hl-F1, R and Size.
Their Clf-F1 differs from the published value by at most 0.55 points.
For MGR, that gap of 0.11 points is 2.7 published standard deviations, since the published spread is 0.04.
The release's MGR is 0.16 points below on Clf-F1, which is 4.0 published standard deviations.
G-RAT's Hl-F1 ranges from 39.1 to 67.1 across seeds.

FR does not reproduce.
Seed 2001 reaches a Clf-F1 of 67.3 and an Hl-F1 of 14.7.
Seed 1337 classifies at 93.5 but reaches an Hl-F1 of 17.7.
The other three seeds classify at 97.9 or above and reach an Hl-F1 between 37.8 and 41.5, still below the published 54.07.
The release's FR reaches the published Hl-F1, and none of its seeds collapses.
The gap is therefore this reproduction's.

Every Toy text has twenty characters, so no difference over padding applies.
FR then differs from the release only in predictor pooling and the unused one-hot column.
Pooling over the kept positions alone is a choice of the library, which `docsrc/source/models/fr.rst` states.
No run isolates its effect on the collapsed seeds.

GenSPP classifies as published but selects less.
Its R is 3.05 points below the published value, and its Hl-F1 is 10.02 points below.
The release's GenSPP is also 6.1 points below the published Hl-F1, and this reproduction is 4.0 points below the release.
[GenSPP on Toy over twenty seeds](#genspp-on-toy-over-twenty-seeds) explains both gaps.

### HateXplain

MGR and G-RAT reproduce within 1.4 published standard deviations on every column.
FR classifies as published, but its Hl-F1 is 4.52 points above the published value while its R is 2.33 points below.

MCD reproduces on four of its five seeds.
Seed 2023 reaches a Clf-F1 of 48.4 and an Hl-F1 of 14.2.
The other four classify between 68.4 and 69.1 and reach a mean Hl-F1 of 28.1, against the published 27.92.

The release's MGR is 8.59 points below the published Hl-F1, which is 1.8 published standard deviations.
Its seeds 15451 and 1337 reach an Hl-F1 of 5.6 and 5.9 while still selecting 25.5% and 25.7% of the words.
Its other three seeds reach 29.5 to 32.4.
Every other column of the four release baselines lies within 1.7 published standard deviations of the published value.

GenSPP does not reproduce.
Its Clf-F1 is 1.74 points below the published value, and it selects more words than published, where on Toy it selects fewer.
The release scores GenSPP's Hl-F1 on a post only up to the post's first token unseen in training and validation, which keeps 66.5% of the test tokens.
`truncated.py` scores this reproduction's checkpoints that way.

| GenSPP on HateXplain | Scoring | Hl-F1 |
|---|---|---|
| Published | Release's | 42.62 ± 0.73 |
| Release | Release's | 43.09 ± 2.73 |
| Here | Release's | 40.80 ± 3.20 |
| Here | Every token | 37.53 ± 3.40 |

The published value is scored the release's way, and the release lies within one published standard deviation of it.
The scoring accounts for 3.27 of the 5.09 points between this reproduction and the published value.
The remaining 2.29 points, between the release and this reproduction under the same scoring, are within one standard deviation of either set of five seeds.
The two code bases differ in every row of the [Data](#data) and [GenSPP](#genspp) tables, including the validation rows and the embeddings of tokens that GloVe lacks.
Seeds 1, 2 and 4 reach an Hl-F1 between 38.6 and 40.9 under every-token scoring, and seeds 3 and 5 reach 34.3 and 33.6.

### GenSPP on Toy over twenty seeds

The release's GenSPP code was last changed in August 2024.
This repository cannot tell whether the published numbers came from that code under other seeds or another environment, or from an earlier version.

Every GenSPP run selects a short code per class rather than the highlight, as [the next section](#toy-what-the-selections-contain) shows.
The codes differ only in length, and the length decides the Hl-F1.

| Seed, here | `aba` | `baa` | `abc` | Characters | Hl-F1 | R |
|---|---|---|---|---|---|---|
| 1 | `ba` | `a` | `b` | 4 | 62.53 | 7.52 |
| 2 | `ba` | `a` | `b` | 4 | 64.29 | 8.04 |
| 4 | `ba` | `a` | `b` | 4 | 61.17 | 7.35 |
| 3 | `ba` | `aa` | `b` | 5 | 70.55 | 9.27 |
| 5 | `ba` | `aa` | `b` | 5 | 71.48 | 9.90 |

A five-character code overlaps the gold highlight on one more character, which is worth about eight points of Hl-F1.
Once the task loss is below its limit, the fitness prefers the sparser selection, so the four-character code is the better optimum of the search.
The Hl-F1 is therefore lower in exactly the runs where the search reaches a better fitness.
The release selects the five-character code in all five of its runs over seeds 1 to 5, and this reproduction selects the four-character code in three of them.

The `probe/variance` branch of the release reruns it over seeds 1 to 20 on an L40S GPU, unchanged and in two variants.
The first trains each candidate's predictor without the validation rows, which the release trains on.
The second seeds each candidate's batch order by its chromosome, where the release draws it from one generator seeded once per run.

| Toy GenSPP | Seeds | Clf-F1 | Hl-F1 | R | Four-character codes |
|---|---|---|---|---|---|
| Here | 1 to 5 | 99.56 ± 0.30 | 66.00 ± 4.72 | 8.42 ± 1.12 | 3 of 5 |
| Release | 1 to 5 | 99.61 ± 0.23 | 69.96 ± 0.69 | 9.94 ± 0.27 | 0 of 5 |
| Release, unchanged | 1 to 20 | 99.53 ± 0.28 | 68.64 ± 4.12 | 9.39 ± 1.19 | 4 of 20 |
| Release, without the validation rows | 1 to 20 | 99.30 ± 0.51 | 68.74 ± 2.99 | 9.79 ± 1.00 | 2 of 20 |
| Release, batch order seeded by the chromosome | 1 to 20 | 99.51 ± 0.34 | 69.00 ± 4.00 | 9.55 ± 1.09 | 5 of 20 |

Over twenty seeds, the release spreads as much as this reproduction does.
Its seeds 1 to 5 give an Hl-F1 of 70.09 ± 0.72, which agrees with the five-seed rerun.
Four of its twenty runs find a four-character code: `ba`, `a` and `b` in two, and `a`, `aa` and `c` in the other two.
At that rate, five seeds miss every four-character code with a probability of 0.33.
The small spread over seeds 1 to 5 is therefore a property of those seeds rather than of the release.

Neither test detects a difference between this reproduction over seeds 1 to 5 and the unchanged release over seeds 1 to 20, although five runs detect only a large one.
A two-sided exact Mann-Whitney test on Hl-F1 gives p = 0.62.
A two-sided Fisher exact test on the share of four-character codes gives p = 0.113.

Neither variant changes the outcome against the unchanged release.

| Variant against the unchanged release | Hl-F1, Mann-Whitney p | Spread, Levene p | Four-character codes, Fisher p |
|---|---|---|---|
| Without the validation rows | 0.231 | 0.516 | 0.661 |
| Batch order seeded by the chromosome | 0.799 | 0.904 | 1.000 |

A third variant copied the initial weights that a reused model slot is reset to.
Its twenty runs are byte-identical to the unchanged ones, since on a GPU moving a candidate to the device already gives its parameters new storage.

The release reports `np.std`, the population deviation, which over five runs is 0.894 times the sample deviation.
That cannot account for a spread six times smaller.
The highest of the twenty unchanged runs reaches 73.9, below the published 76.02.
One run of the variant with the batch order seeded by the chromosome reaches 76.84, above it.

### Toy: what the selections contain

The Toy corpus was designed so that only the whole highlight determines the class.
Each class has one highlight, `aba`, `baa` or `abc`, and the generator was meant to contaminate every text with fragments of the other two.
`selection.py` checks whether the test selections of trained models contain that evidence, without consulting the model's own predictor.

The **complete-highlight rate** is the share of test examples whose selection contains the whole gold highlight and no other class highlight as a contiguous run of selected characters.
The positional view requires every gold position to be selected.
The string view keeps the selected characters in text order without their positions, and it is the only view available for the release, whose masks files store no positions.
The complete-highlight rate is not the `sufficiency` of `pyhighlights.components.faithfulness`.
That measure compares the model's own predictions with and without the rest of the input.
A predictor that decodes the label from a partial selection therefore reports that selection as sufficient.

The four baselines sample their selection at evaluation.
`selection.py` seeds each run's draw with that run's seed, so its rows reproduce.
They still score another draw than the run's own test metrics.

| Toy, test split | Own accuracy | Complete, positional | Complete, string |
|---|---|---|---|
| FR | 92.59 ± 11.28 | 8.81 ± 8.53 | 8.97 ± 8.55 |
| MGR | 99.74 ± 0.17 | 22.03 ± 10.87 | 22.18 ± 10.67 |
| MCD | 99.83 ± 0.17 | 56.11 ± 6.12 | 56.13 ± 6.02 |
| G-RAT | 98.74 ± 2.57 | 21.91 ± 13.59 | 22.19 ± 13.80 |
| GenSPP | 99.56 ± 0.30 | 3.23 ± 2.23 | 3.90 ± 2.57 |
| GenSPP, release | 99.61 ± 0.23 | - | 6.59 ± 0.43 |

At most 56% of a model's selections contain the evidence, and only MCD exceeds a third.
Every predictor therefore decodes the label from something other than the whole highlight in at least 44% of the test examples.
The two views differ by less than one point, so selections that spell a highlight by accident are rare.

`python selection.py --per-class` shows what the selections contain.

| Class | GenSPP code here | Share of the class's examples | GenSPP code in the release | Share |
|---|---|---|---|---|
| `aba` | `ba` | 69% to 91% | `ba` | 65% to 78% |
| `baa` | `a` in three runs, `aa` in two | 57% to 83%, and 77% and 78% | `aa` | 64% to 75% |
| `abc` | `b` | 73% to 94% | `b` | 72% to 82% |

A lone `b` occurs in every highlight, so it is no evidence for `abc`.
The released corpus does make `aa` and `bc` evidence, against its design.
Outside the highlight, each of `ab`, `ba`, `aa` and `bc` occurs in only 1.7% to 3.0% of the documents of every class, the rate random filler produces.
Of the first 8,000 documents, 89.5% of those containing `aa` are `baa` documents, and 96.0% of those containing `bc` are `abc` documents.
A rule that reads `aa` as `baa`, `bc` as `abc` and anything else as `aba` classifies 94.55% of the 2,000 test documents.

With its middle character hidden, `a?a` still marks `aba` in 85.6% of the documents that contain it.
A lone `c` marks `abc` in 50.2% of the documents that contain it, against a third by chance.
The design therefore holds only for `a` and `b`, which occur in every highlight.

The baselines mostly mark a class by the size of the selection.
In 14 of their 20 runs, one class receives an empty selection in at least 90% of its test examples.
In 13 of those 14 runs, the other two classes receive 2.65 to 6.15 characters on average, which keeps the mean size near the sparsity target of three.

The contamination therefore constrains only a predictor that has to find the highlight itself.
A generator reads the whole text, so it can locate the highlight and pass on any selection that names the class.
Hl-F1 gives partial credit for partial overlap, while the complete-highlight rate scores a selection as complete or not.
Across the 20 baseline runs, the Pearson correlation between the two is 0.76.
GenSPP lies outside that trend: it ranks first on Hl-F1, between 61 and 71, and last on the complete-highlight rate, at 3.23%.

`selection.py` also trains five independent readers per view, each a GRU trained on windows of the training texts that contain the gold highlight.
Their accuracy on a model's selections is not reported as a result.
Codes such as a lone `b` never occur in their training data, so each reader maps them arbitrarily.
On GenSPP's selections, reader accuracy ranges from 21% to 81% across the five reader seeds and both views.

These results cover Toy alone, since HateXplain has no designed evidence to check a selection against.

### Cost

`compare.py` prints a second table per corpus with what each run cost, from the `cost_` columns every seed writes.
The paper reports none of it.

| Column | Meaning |
|---|---|
| `runtime/model` | Wall clock per model trained, `runtime × at-once / models-trained`. The column to compare rows on |
| `runtime/seed` | Wall clock per seed, which for a search covers every candidate trained |
| `inference/batch` | Milliseconds per batch at inference |
| `inference/pass` | Wall clock for one pass over the test split |
| `memory/peak` | The larger of this process's peak and the largest single worker's. It is what a machine needs for the seed, not what one model uses, and it does not sum the workers, which share the corpus |
| `parameters` | Parameters of the scored model, split into `trainable` and `frozen` |
| `models trained` | One for a baseline, 5050 for a search |
| `at once` | Candidates scored in parallel, one per worker |

An entry reads `-` where a model has not been run, or where the tree predates the library reporting costs.

| Seed wall clock, here | Toy | HateXplain |
|---|---|---|
| Baseline | 5.9 to 14.1 min | 8.0 to 14.2 min |
| GenSPP, CPU workers | 27.4 min, 2.6 s per candidate | 72.8 min, 6.9 s per candidate |

## Reproducibility

Table 1 can be rerun: all five models on both corpora over five seeds, reporting Clf-F1, Hl-F1, R and Size.
The paper does not list its seeds.
The baselines use `[2023, 15451, 1337, 2001, 2080]`, the seeds of the release's baselines.
GenSPP uses 1 to 5, the seeds of the release's genetic code.

Table 2 cannot be rerun, except its `GenSPP (G = 150)` row, which is `n_generations=150` and nothing else.
The skew experiment needs a selector pre-trained to select the first token and injected into the initial population.
`GenSPPTrainer` builds every founder at random, with no way to seed one ([pyhighlights issue 124](https://github.com/nlp-unibo/pyhighlights/issues/124)).
The `**` significance markers are Wilcoxon tests over seeds against the best baseline, and nothing here computes them.

The release was rerun from the `cluster` branch of its repository.
The baselines ran at `634c86e`, which changes only logging, data paths and the Slurm jobs.
GenSPP ran from the same branch, whose `genetic/` is unchanged from `2b447a2`, under the versions that `genetic/requirements.txt` pins.
The released baselines read Toy from `toy/toy_sequential_dataset_easy.pkl`, which the release does not ship.
The rerun reads `genetic/data/toy_dataset.pkl` in its place, and the authors confirm that the two files hold the same corpus.
That corpus is the one the released GenSPP reads and this reproduction's artifact holds.

| Numbers | Hardware | Software |
|---|---|---|
| Published | NVIDIA RTX 3060 Ti 8 GB, Intel Core i7-8700K, Ubuntu 24 | CUDA 12.x |
| Release | NVIDIA L40S 46 GB with driver 580.95.05, Intel Xeon Platinum 8480+, Ubuntu 24.04.2 | Baselines: torch 1.13.1 for CUDA 11.6, lightning 1.9.0, torchmetrics 0.11.4. GenSPP: torch 2.3.1 for CUDA 12.1, numpy 2.0.0 |
| Here | The same L40S nodes. GenSPP scores its candidates on CPU workers | torch 2.6.0 for CUDA 12.4, lightning 2.6.6, torchmetrics 1.9.0, pyhighlights `d245d98` |

The paper's appendix A.4 names the GPU alone, and the rest of its row is from the authors.
The RTX 3060 Ti and the L40S differ in architecture, Ampere and Ada, so the same code and seed need not give the same numbers on both.

## Differences from the release

Each model here follows its own paper and its authors' implementation, the **original implementation**, except where this section says otherwise.
GenSPP's original implementation is the release itself.
The setup of the paper is reproduced as published: corpora, splits, hyperparameters, seeds and metrics, except the two hyperparameters below.
Where the release departs from an original implementation, the difference is documented here and not reproduced.
An error in an original implementation is documented in the library's model pages, under `docsrc/source/models/`.
[Issue 45](https://github.com/nlp-unibo/pyhighlights-genspp2025/issues/45) records every difference and how it was resolved.

Two hyperparameters of the release differ from a table of the paper, and the release's value is taken for both.
On Toy, the contiguity coefficient is 2.0, as appendix A.2 and the release set it, where Table 4 gives 1.0.
On HateXplain, the coefficient of G-RAT's Jensen-Shannon divergence term is 1.5, as the release sets it, where Table 5 gives 1.0.

### Data

| Difference | Here | In the release |
|---|---|---|
| HateXplain source | Parsed from upstream, 13507 rows, every one agreeing with the release on tokens, label and highlight | Read from the release's own pickles |
| Split scheme | One scheme serves all five models, so their numbers are comparable. Toy reads the release baselines' split from the artifact's `split` column | A scheme per model |
| Validation rows | Held out of training | Trained on, by the genetic code |
| HateXplain embeddings | Every token of every split has a row of its own: its GloVe vector, or a `uniform(-0.05, 0.05)` row from a fixed seed | The baselines map a token GloVe lacks to `<PAD>`, a zero row. GenSPP shares one uniform vector over such training tokens and embeds a token unseen in training as zeros |
| Toy one-hot table | 24 columns | 25 in the baselines, 26 in GenSPP |

### Baselines

The first three rows are not departures of the release, since the original FR, MGR, MCD and G-RAT pool, recur and penalise as the release does.
They are choices of the library, and its model pages explain them.

| Difference | Here | In the release |
|---|---|---|
| Predictor pooling | Max over the kept positions alone, and an empty selection pools to a zero vector | Max over every valid position, so the predictor reads the recurrent states at dropped words |
| Recurrence over padding | Packed, so a document is encoded alike in any batch | An unpacked bidirectional GRU, whose backward direction reads padding first (HateXplain) |
| Penalties over padding | Sparsity and contiguity over words alone | Over padding as well, since the sampled mask is not masked there (HateXplain) |
| G-RAT pretraining | The rationalizer steps from epoch 10, as the original does | From epoch 12 |
| G-RAT early stopping | Patience and checkpoints count from the end of pretraining. The original saves checkpoints only from then on and has no early stopping | From epoch 0 |
| G-RAT guide over padding | The guide target is scaled by its mean over words, and its cross entropy averages over words alone | The mean is taken over the padded width, and the cross entropy also scores padding against a target of zero (HateXplain) |
| MCD validation loss | One selection for every term | One sample for the predictor's cross entropy and another for sparsity, contiguity and the discrepancy |
| MGR predictor | No layer normalisation, as the original | A `LayerNorm` in no optimizer group, which still normalises with its initial weights |

Every baseline samples its selection at validation and test, as the release does.
The original G-RAT thresholds its selection at 0.5 there instead ([`model/model_st.py`](https://github.com/shuaibo919/g-rat/blob/d279932/model/model_st.py)), and its paper gives no reason for it.
This repository samples, so that the four baselines are scored under one inference strategy.

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
| GenSPP metrics | The library's metrics, as for every baseline | Code of its own. Its `Binary Micro F1 Mask` is the Hl-F1 the paper reports, and it pools true positives, false positives and false negatives over every test token as the library does. Its selection rate, `Sparsity`, is averaged over batch means. It computes no selection size |
| HateXplain Hl-F1 | Every token of a document | A document up to its first all-zero embedding, so a test token unseen in training ends the scored part |

## Corpora

Toy is read by `pyhighlights.components.loaders.ToyLoader` with a `url`, so this repository defines no loader of its own.
One loader generates, saves and reads, so a published toy corpus is a URL and a digest in a configuration rather than a class per dataset.

The artifact is [10.5281/zenodo.23570345](https://doi.org/10.5281/zenodo.23570345), `pyhighlights-genspp-toy-v3.zip`.
It holds the corpus in the library's own columns, converted when the artifact was built.
Its `split` column holds the released baselines' split: the first 80% is train, a fifth of it is held out under seed 15000, and the rest is test.
The split is therefore data, not a draw that depends on a library version.
`pyhighlights/tools/build_datasets.py --skip-r2a` reproduces the published bytes from the release's pickle, and `genspp2025/configurations/toy/datasets.py` pins their digest.

`genspp2025/components/corpora.py` holds a `ReleasedToyLoader` for that original pickle, which the first version of the record still carries and the release ships.
That file stores `structure_indexes`, the positions a highlight marks, where the library stores a vector.
`ReleasedToyLoader` fills in that column and hands the rest to `ToyLoader`.
It is registered nowhere, so no configuration reaches it by key.
A test pins that it and a plain `ToyLoader` over the converted file return the same rows.

```python
from genspp2025.components.corpora import ReleasedToyLoader

splits = ReleasedToyLoader(url="toy_dataset.pkl").load()
```

HateXplain needs GloVe, which `build.sbatch` fetches.
`run.sbatch` refuses to start without it, and so does the task.
`requires_embeddings` exists because the registered HateXplain task once ran without its vector file and reported numbers for a two-word vocabulary.

## Contact

Questions about a component or a result are best raised as an [issue](https://github.com/nlp-unibo/pyhighlights-genspp2025/issues).
For anything else, write to Federico Ruggeri, federico.ruggeri6@unibo.it.
