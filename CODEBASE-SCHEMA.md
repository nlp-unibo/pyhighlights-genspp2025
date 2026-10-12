# Codebase schema

## cluster/build.sbatch

`build.sbatch` selects node-local storage for `APPTAINER_CACHEDIR` and `APPTAINER_TMPDIR` when at least 20 GB is free. It falls back to persistent cache and per-job temporary directories under `SCRATCH`. It calls `apptainer build` with `cluster/env.def`, then calls `apptainer exec` to build the Cinnamon registry and fetch both corpora. Lastly, it downloads, validates, and stages GloVe when required.
