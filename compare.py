"""Read a results tree and set it beside the numbers the paper published.

Usage::

    python compare.py --results results --corpus toy
    python compare.py --results results            # both corpora

The library already turns a results tree into a table --
:class:`pyhighlights.components.analyzers.MetricsAnalyzer` does the walking,
the per-seed aggregation and the ``mean ± std``. All this adds is the other
half of each row and the distance between them.

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
from pathlib import Path

import pandas as pd
from pyhighlights.components.analyzers import MetricsAnalyzer

from published import COLUMNS, TABLE_1

#: The library's column name for each of the paper's, and what to multiply it
#: by to land in the paper's units.
MEASURED = {
    "clf_f1": ("f1", 100.0),
    "hl_f1": ("highlight_f1", 100.0),
    "rate": ("selection_rate", 100.0),
    "size": ("selection_size", 1.0),
}

MODELS = ("fr", "mgr", "mcd", "grat", "genspp")


def measured(directory: Path, corpus: str) -> pd.DataFrame:
    """Every run under ``directory`` whose task name starts with ``corpus``."""
    frame = MetricsAnalyzer(
        directory=directory,
        metrics=[name for name, _ in MEASURED.values()],
        split="test",
        pairs=True,
    ).analyze()
    return frame[frame["task"].str.startswith(f"{corpus}-")]


def compare(directory: Path, corpus: str) -> pd.DataFrame:
    frame = measured(directory, corpus).set_index("task")
    rows = []
    for model in MODELS:
        published = TABLE_1[corpus][model]
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
        print(f"\n=== {corpus} — Table 1 ===")
        for column in COLUMNS:
            print(f"\n{column}")
            shown = frame[
                ["model", f"{column}_measured", f"{column}_published", f"{column}_d"]
            ]
            shown.columns = ["model", "measured", "published", "d"]
            print(shown.to_string(index=False))


if __name__ == "__main__":
    main()
