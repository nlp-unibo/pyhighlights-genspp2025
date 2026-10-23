# Codebase schema

## cluster/build.sbatch

`build.sbatch` selects node-local storage for `APPTAINER_CACHEDIR` and `APPTAINER_TMPDIR` when at least 20 GB is free. It falls back to persistent cache and per-job temporary directories under `SCRATCH`. It calls `apptainer build` with `cluster/env.def`, then calls `apptainer exec` to build the Cinnamon registry and fetch both corpora. Lastly, it downloads, validates, and stages GloVe when required.

## selection.py

`selection.py` depends on `genspp2025.configurations.toy.keys`, `cinnamon.registry`, `pyhighlights`, and `torch`.
`main` calls `Registry.build`, then `reproduction`, `release`, and either `per_class` or `Readers`.
`reproduction` builds each Toy task with `Registry.from_key`, then calls `build_model` and `loaders` on it.
`release` parses the reference implementation's `masks_<name>_<seed>.txt` files, where `name` is `toy` or a probe arm's `toy_probe-<arm>`.
`per_class` prints each class's most frequent selections and the code a run settled on.

`main` scores each source with `complete_in_view`, `complete_in_string`, and `Readers.accuracy`, then prints its means with `spread`.
`Readers` trains `Reader` instances through `Readers._train`, which calls `gold` and `Readers.encode`.
`tests/test_genspp2025.py` calls `complete_in_string` and `complete_in_view`.

## compare.py

The module depends on `published`, `genspp2025.configurations` keys, and `pyhighlights.components.analyzers`.
`main` calls `compare` and `costs`, and both read through `rows_of` and `analyzed`.
`analyzed` builds `PublishedRows`, a `MetricsAnalyzer` whose `reports` calls `row_of` on every run and `latest_runs` on the runs that measure a row.
`row_of` reads a run's `manifest.json` and maps its search key through `SEARCHES` and its seeds against `SEEDS` and `GENSPP_SEEDS`.
`PublishedRows.reports` calls `conflict` for a row that more than one run measures.

## run.py

The module depends on `genspp2025.configurations` and `cinnamon.registry`.
`main` calls `Registry.build`, `keys`, and `seed_list`, then runs each chosen task with `Registry.from_key`.
`keys`, `smoke_search` and `g150_search` call `corpus_keys`.
`main` replaces the GenSPP task's `search` with `smoke_search`, `released_threshold_search`, or `g150_search`.
