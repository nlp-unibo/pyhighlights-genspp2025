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
