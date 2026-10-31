"""Read a results tree and set it beside the numbers the paper published.

Usage::

    python compare.py --results results --corpus toy
    python compare.py --results results            # both corpora

Two tables per corpus. **Table 1** is the paper's, measured beside published,
with Table 2's ``GenSPP (G = 150)`` row under it.
**Cost** is what those numbers took to produce: runtime, inference, memory
in mebibytes, parameters trainable and frozen, none of which the paper
report and which is the other axis a reader compares a rationalizer on. It has
no published half; it is filled by running the experiments.

The library already turns a results tree into a table:
:class:`pyhighlights.components.analyzers.MetricsAnalyzer` does the walking,
the per-seed aggregation and the ``mean ± std``. All this adds is the other
half of each row and the distance between them.

**Which run is a row.** Every run writes ``<task>/<stamp>``, so a smoke run,
a run of other seeds and a longer search sit beside the paper's under one task
name. `row_of` reads each run's manifest to say which published row, if any,
it measures, and two runs of one row resolve to the newest, named on stderr.

**Units.** The paper reports percentages for the three F1-like columns and for
the selection rate; the library reports those as fractions, and the selection
size as a count either way. The conversion is here rather than in the library:
a metric should report one quantity, and which multiple of it a paper prints
is the paper's business.

**What the distance means.** ``d`` is the gap in units of the *published*
standard deviation over its five seeds. It is a readability aid, not a test:
five seeds estimate a standard deviation badly, and the paper's own
significance claims are Wilcoxon over seeds rather than anything derived from
these summaries. Read a large ``d`` as "look at this row", not as "this
failed".
"""

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd
from pyhighlights.components.analyzers import MetricsAnalyzer, latest_runs

from genspp2025.configurations.hatexplain import keys as hatexplain_keys
from genspp2025.configurations.hatexplain.tasks import (
    HateXplainFRTaskConfig,
    HateXplainGenSPPTaskConfig,
    HateXplainGRATTaskConfig,
    HateXplainMCDTaskConfig,
    HateXplainMGRTaskConfig,
)
from genspp2025.configurations.toy import keys as toy_keys
from genspp2025.configurations.toy.tasks import (
    ToyFRTaskConfig,
    ToyGenSPPTaskConfig,
    ToyGRATTaskConfig,
    ToyMCDTaskConfig,
    ToyMGRTaskConfig,
)
from published import COLUMNS, TABLE_1, TABLE_2

#: The library's column name for each of the paper's, and what to multiply it
#: by to land in the paper's units.
MEASURED = {
    "clf_f1": ("f1", 100.0),
    "hl_f1": ("highlight_f1", 100.0),
    "rate": ("selection_rate", 100.0),
    "size": ("selection_size", 1.0),
}

MODELS = ("fr", "mgr", "mcd", "grat", "genspp")

#: The published row each registered GenSPP search measures. A search missing
#: from here, the smoke one or the release's threshold mutation, measures none.
SEARCHES = {
    str(toy_keys.TOY_GENSPP_TRAINER): "genspp",
    str(toy_keys.TOY_GENSPP_G150_TRAINER): "genspp-g150",
    str(hatexplain_keys.HATEXPLAIN_GENSPP_TRAINER): "genspp",
    str(hatexplain_keys.HATEXPLAIN_GENSPP_G150_TRAINER): "genspp-g150",
}

#: Each paper task's seeds, by the task name a run's manifest records. A run
#: measures a row only on the seeds its task registers.
PAPER_SEEDS = {
    config.model_fields["name"].default: list(config.model_fields["seeds"].default)
    for config in (
        ToyFRTaskConfig,
        ToyMGRTaskConfig,
        ToyMCDTaskConfig,
        ToyGRATTaskConfig,
        ToyGenSPPTaskConfig,
        HateXplainFRTaskConfig,
        HateXplainMGRTaskConfig,
        HateXplainMCDTaskConfig,
        HateXplainGRATTaskConfig,
        HateXplainGenSPPTaskConfig,
    )
}

#: The rows the comparison prints, and the table each is published in.
ROWS = {**{model: TABLE_1 for model in MODELS}, "genspp-g150": TABLE_2}

#: The cost table: the library's column name, and how to print it. Runtimes
#: are per model trained rather than per seed, which is what makes a search
#: comparable to a model trained once. See `cost_runtime_per_run_s` in the
#: library. Nothing is published to set these beside.
COSTS = {
    "runtime/model": ("runtime_per_run_s", "seconds"),
    "runtime/seed": ("runtime_s", "seconds"),
    "inference/batch": ("inference_batch_s", "milliseconds"),
    "inference/pass": ("inference_epoch_s", "seconds"),
    "memory/peak": ("memory_mib", "mebibytes"),
    "parameters": ("parameters", "parameters"),
    "trainable": ("trainable_parameters", "parameters"),
    "frozen": ("frozen_parameters", "parameters"),
    "models trained": ("models", "count"),
    "at once": ("concurrency", "count"),
}


def shown(value: tuple[float, float] | str, unit: str) -> str:
    """One cell of the cost table, in the unit that reads best.

    A seed of GenSPP is hours and a batch of inference is milliseconds, so a
    single format would print either as zero or as a wall of digits.
    """
    if not isinstance(value, tuple):
        return "-"
    mean, deviation = value
    if unit == "count":
        return f"{mean:,.0f}" if not deviation else f"{mean:,.0f} ± {deviation:,.0f}"
    if unit == "parameters":
        # A toy backbone is thousands and a transformer is hundreds of
        # millions, and neither reads in the other's unit. The threshold is
        # on what will be printed rather than on the mean, so 999,990 reads
        # as `1.00M` rather than as `1000.0k`.
        def sized(value: float) -> str:
            if round(value / 1e3, 1) >= 1000:
                return f"{value / 1e6:.2f}M"
            return f"{value / 1e3:.1f}k" if value >= 1e3 else f"{value:,.0f}"

        # A search settles on whatever candidate won, so a parameter count is
        # a spread across seeds like every other column here.
        return sized(mean) if not deviation else f"{sized(mean)} ± {sized(deviation)}"
    if unit == "milliseconds":
        return f"{mean * 1e3:.1f} ± {deviation * 1e3:.1f}"
    if unit == "mebibytes":
        return f"{mean:,.0f} ± {deviation:,.0f}"
    # Seconds, until an hour makes them unreadable.
    if mean >= 3600:
        return f"{mean / 3600:.2f}h ± {deviation / 3600:.2f}"
    if mean >= 60:
        return f"{mean / 60:.1f}m ± {deviation / 60:.1f}"
    return f"{mean:.2f}s ± {deviation:.2f}"


def costs(directory: Path, corpus: str) -> pd.DataFrame:
    """One row per model, one column per cost, ``-`` where a run has none.

    A results tree written before the library reported costs has no ``cost_``
    columns at all, and says so rather than printing zeros.
    """
    frame = rows_of(directory, corpus, "cost", [name for name, _ in COSTS.values()])
    frame = frame.set_index("task") if "task" in frame.columns else frame
    rows = []
    for model in ROWS:
        task = f"{corpus}-{model}"
        found = frame.loc[task] if task in frame.index else None
        row = {"model": model}
        for label, (name, unit) in COSTS.items():
            row[label] = "-" if found is None else shown(found[name], unit)
        rows.append(row)
    return pd.DataFrame(rows)


def row_of(run: Path) -> str | None:
    """The published row a run measures, read off its own manifest.

    Every run of a task writes ``<task>/<stamp>``, a smoke run, a run of other
    seeds or a longer search as much as the paper's. What tells them apart is
    the configuration the run recorded: the search it ran and the seeds it ran
    over. A run on the paper's seeds and a registered search, or on no search
    at all, measures a row. Anything else measures none, so ``None``.
    """
    manifest = run / "manifest.json"
    if not manifest.exists():
        return None
    settings = json.loads(manifest.read_text()).get("settings", {})
    trainer_args = settings.get("trainer_args") or {}
    if any(name.startswith("limit_") for name in trainer_args):
        return None
    model = settings.get("name", "").partition("-")[2]
    if "search" in settings:
        model = SEARCHES.get(settings["search"].get("@key"))
    if model not in ROWS:
        return None
    seeds = PAPER_SEEDS.get(settings.get("name"))
    return model if list(settings.get("seeds", [])) == seeds else None


@lru_cache(maxsize=None)
def conflict(name: str, stamps: tuple[str, ...]) -> None:
    """Name the runs one row had to choose between, once however many splits
    read them."""
    print(
        f"{name}: {len(stamps)} runs, reading {stamps[-1]} of {list(stamps)}",
        file=sys.stderr,
    )


class PublishedRows(MetricsAnalyzer):
    """`MetricsAnalyzer`, over the runs that measure a published row.

    A run is reported under ``<corpus>-<row>`` rather than its task name, so
    a G = 150 search and the paper's search are two rows of one task. Two
    runs of one row are a conflict, resolved as the library does, to the
    newest, and named on stderr so neither is lost silently.
    """

    def reports(self):
        candidates = []
        for path in sorted(self.directory.rglob("results.json")):
            row = row_of(path.parent)
            if row is None:
                continue
            report = json.loads(path.read_text())
            corpus = report.get("name", "").partition("-")[0]
            name = f"{corpus}-{row}"
            candidates.append((name, path.parent.name, {**report, "name": name}))
        for name, stamp, _ in candidates:
            others = [other for found, other, _ in candidates if found == name]
            if len(others) > 1 and stamp == max(others):
                conflict(name, tuple(sorted(others)))
        return latest_runs(
            (name, stamp, {**report, "run": stamp})
            for name, stamp, report in candidates
        )


@lru_cache(maxsize=None)
def analyzed(directory: Path, split: str, metrics: tuple[str, ...]) -> pd.DataFrame:
    """One walk of the results tree per split, however many tables read it.

    The analyzer re-reads every ``results.json`` under the tree on each call
    and filters by corpus afterwards, so a two-corpus run walked it four times
    over.
    """
    return PublishedRows(
        directory=directory, metrics=list(metrics), split=split, pairs=True
    ).analyze()


def rows_of(directory: Path, corpus: str, split: str, metrics) -> pd.DataFrame:
    """The runs of one corpus, or an empty frame where there are none.

    A results tree with no ``results.json`` in it, a directory made by hand or
    a batch of jobs that all failed, analyzes to a frame with no columns at
    all, which has no ``task`` to filter on.
    """
    frame = analyzed(directory, split, tuple(metrics))
    if "task" not in frame.columns:
        return frame
    return frame[frame["task"].str.startswith(f"{corpus}-")]


def measured(directory: Path, corpus: str) -> pd.DataFrame:
    """Every run under ``directory`` whose task name starts with ``corpus``."""
    return rows_of(directory, corpus, "test", [name for name, _ in MEASURED.values()])


def compare(directory: Path, corpus: str) -> pd.DataFrame:
    frame = measured(directory, corpus)
    frame = frame.set_index("task") if "task" in frame.columns else frame
    rows = []
    for model, table in ROWS.items():
        published = table[corpus][model]
        row = {"model": model}
        task = f"{corpus}-{model}"
        found = frame.loc[task] if task in frame.index else None
        for column in COLUMNS:
            name, scale = MEASURED[column]
            expected, deviation = published[column]
            row[f"{column}_published"] = f"{expected:.2f} ± {deviation:.2f}"
            if found is None or not isinstance(found[name], tuple):
                row[f"{column}_measured"] = "-"
                row[f"{column}_d"] = "-"
                continue
            mean, std = (value * scale for value in found[name])
            row[f"{column}_measured"] = f"{mean:.2f} ± {std:.2f}"
            # A published standard deviation of zero would divide by nothing;
            # none is zero in either table, and a guard is cheaper than a note.
            distance = (mean - expected) / deviation if deviation else None
            row[f"{column}_d"] = f"{distance:+.1f}" if distance is not None else "-"
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results", type=Path)
    parser.add_argument("--corpus", choices=("toy", "hatexplain"), default=None)
    arguments = parser.parse_args()

    if not arguments.results.exists():
        raise SystemExit(f"no results tree at {arguments.results}")

    for corpus in [arguments.corpus] if arguments.corpus else ("toy", "hatexplain"):
        frame = compare(arguments.results, corpus)
        print(f"\n=== {corpus}: Table 1, and Table 2's genspp-g150 ===")
        for column in COLUMNS:
            print(f"\n{column}")
            table = frame[
                ["model", f"{column}_measured", f"{column}_published", f"{column}_d"]
            ]
            table.columns = ["model", "measured", "published", "d"]
            print(table.to_string(index=False))

        print(f"\n=== {corpus}: cost ===")
        print(costs(arguments.results, corpus).to_string(index=False))


if __name__ == "__main__":
    main()
