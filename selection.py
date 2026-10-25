"""What the selections of trained Toy models contain.

Usage::

    python selection.py --results results
    python selection.py --results results --release <gen-spp>/genetic/results/outputs
    python selection.py --results results --per-class
    python selection.py --release <outputs> --release-name toy_probe-noval --per-class

The Toy corpus is designed so that a class is determined only by its whole
highlight, one of ``aba``, ``baa`` and ``abc``. This script checks whether a
model's test selections contain that evidence, with two measures that do not
use the model's own predictor.

**Complete-highlight rate** is the share of test examples whose selection
contains the whole gold highlight. In the positional view, every gold position
has to be selected, and no other class highlight may appear as a contiguous
run of selected characters. The string view keeps the selected characters in
text order without their positions, which is all that the release's masks
files store.

**Independent reader accuracy** is the accuracy of a GRU trained on windows of
the training texts that contain the gold highlight, from the bare highlight to
the whole text. Window length carries no label, so a reader can learn only
content. Five readers are trained with different seeds. A model's own
accuracy minus the reader accuracy is the part of its accuracy that its
selections do not explain.

The complete-highlight rate is not the ``sufficiency`` of
:mod:`pyhighlights.components.faithfulness`. That measure follows ERASER and
asks the model's own predictor, which can decode a label from a selection
that does not contain the evidence.

``--release`` reads the ``masks_toy_<seed>.txt`` files that the reference
implementation writes. Those files store no positions, so the release is
measured in the string view only.
"""

from __future__ import annotations

import argparse
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator, List, NamedTuple, Optional, Sequence

import pyhighlights
import torch as th
from cinnamon.registry import Registry

import genspp2025
from genspp2025.configurations.toy import keys

HIGHLIGHTS = ("aba", "baa", "abc")
MODELS = ("fr", "mgr", "mcd", "grat", "genspp")
BLANK = "_"
READER_SEEDS = range(5)


class Selections(NamedTuple):
    """One trained model's test selections."""

    model: str
    seed: str
    own_accuracy: float
    strings: List[str]
    truths: List[str]
    #: The text with every unselected character blanked, or ``None`` when
    #: the source stores no positions.
    views: Optional[List[str]]
    gold_positions: Optional[List[List[int]]]


def complete_in_string(selected: str, truth: str) -> bool:
    """The selected characters spell the gold highlight and no other one."""
    return {h for h in HIGHLIGHTS if h in selected} == {truth}


def complete_in_view(view: str, gold_positions: Sequence[int], truth: str) -> bool:
    """Every gold position is selected and no other highlight is a selected run."""
    if any(view[i] == BLANK for i in gold_positions):
        return False
    runs = view.split(BLANK)
    return {h for run in runs for h in HIGHLIGHTS if h in run} == {truth}


def gold(row) -> str:
    return "".join(t for t, h in zip(row["tokens"], row["highlights"]) if h == 1)


class Reader(th.nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        self.embedding = th.nn.Embedding(vocabulary_size, 16, padding_idx=0)
        self.gru = th.nn.GRU(16, 32, batch_first=True, bidirectional=True)
        self.out = th.nn.Linear(64, len(HIGHLIGHTS))

    def forward(self, ids: th.Tensor) -> th.Tensor:
        states, _ = self.gru(self.embedding(ids))
        states = states.masked_fill((ids == 0).unsqueeze(-1), -1e4)
        return self.out(states.max(dim=1).values)


class Readers:
    """Five readers per view, trained on windows around the gold highlight."""

    def __init__(self, train_frame):
        alphabet = sorted({c for tokens in train_frame["tokens"] for c in tokens})
        self.index = {BLANK: 1, **{c: i + 2 for i, c in enumerate(alphabet)}}
        self.string = [self._train(train_frame, s, False) for s in READER_SEEDS]
        self.positional = [self._train(train_frame, s, True) for s in READER_SEEDS]

    def encode(self, texts: Sequence[str]) -> th.Tensor:
        ids = th.zeros(len(texts), max(1, max(len(t) for t in texts)), dtype=th.long)
        for i, text in enumerate(texts):
            for j, c in enumerate(text):
                ids[i, j] = self.index.get(c, 0)
        return ids

    def _train(self, frame, seed: int, positional: bool) -> Reader:
        rng = random.Random(seed)
        th.manual_seed(seed)
        reader = Reader(len(self.index) + 1)
        optimizer = th.optim.Adam(reader.parameters(), lr=3e-3)
        for _ in range(30):
            texts, labels = [], []
            for _, row in frame.iterrows():
                tokens = row["tokens"]
                start = list(row["highlights"]).index(1)
                left = rng.randint(0, start)
                right = rng.randint(start + 3, len(tokens))
                window = "".join(tokens[left:right])
                if positional:
                    window = BLANK * left + window + BLANK * (len(tokens) - right)
                texts.append(window)
                labels.append(HIGHLIGHTS.index(gold(row)))
            labels = th.tensor(labels)
            order = th.randperm(len(texts))
            for begin in range(0, len(texts), 128):
                batch = order[begin : begin + 128]
                logits = reader(self.encode([texts[i] for i in batch]))
                loss = th.nn.functional.cross_entropy(logits, labels[batch])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return reader.eval()

    def accuracy(self, reader: Reader, texts, truths) -> float:
        with th.no_grad():
            predicted = reader(self.encode(texts)).argmax(-1).tolist()
        return statistics.mean(HIGHLIGHTS[p] == t for p, t in zip(predicted, truths))


def reproduction(results: Path, splits) -> Iterator[Selections]:
    """The newest run of each model under ``results``, one entry per seed."""
    frame = splits["test"].set_index("sample_id")
    for model_name in MODELS:
        directory = results / f"toy-{model_name}"
        if not directory.is_dir():
            continue
        task = Registry.from_key(getattr(keys, f"TOY_{model_name.upper()}_TASK"))
        loader = task.loaders(splits)["test"]
        run = sorted(path for path in directory.iterdir() if path.is_dir())[-1]
        for checkpoint in sorted(run.glob("seed=*/*.ckpt")):
            seed = checkpoint.parent.name.split("=")[1]
            model = task.build_model()
            state = th.load(checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(state["state_dict"], strict=True)
            model.eval()
            # The four baselines sample their selection at evaluation, so the
            # draw is seeded with the run's seed to make a row reproducible.
            # It is still not the draw the run's own test metrics scored.
            th.manual_seed(int(seed))
            strings, truths, views, positions, correct = [], [], [], [], 0
            with th.no_grad():
                for batch in loader:
                    output = model.reported(model(batch))
                    predicted = output.class_logits.argmax(-1)
                    for r in range(batch.y_true.shape[0]):
                        row = frame.loc[int(batch.sample_ids[r])]
                        keep = output.highlight_mask[r].bool().tolist()
                        pairs = list(zip(row["tokens"], keep))
                        strings.append("".join(t for t, k in pairs if k))
                        views.append("".join(t if k else BLANK for t, k in pairs))
                        positions.append(
                            [i for i, h in enumerate(row["highlights"]) if h == 1]
                        )
                        truths.append(gold(row))
                        correct += int(predicted[r] == batch.y_true[r])
            yield Selections(
                model_name,
                seed,
                correct / len(truths),
                strings,
                truths,
                views,
                positions,
            )


def release(outputs: Path, name: str = "toy") -> Iterator[Selections]:
    """The reference implementation's ``masks_<name>_<seed>.txt`` files.

    ``name`` is ``toy`` for the release as published, and
    ``toy_probe-<arm>`` for a run of its variance probe branch.
    """
    model = "genspp-release"
    if name != "toy":
        model += "-" + name.split("-", 1)[1]
    for path in sorted(outputs.glob(f"masks_{name}_[0-9]*.txt")):
        strings, truths, correct, record = [], [], 0, {}
        for line in [*open(path, encoding="utf-8"), "\n"]:
            line = line.rstrip("\n")
            key, _, value = line.partition(":")
            if key == "Masked":
                record["selected"] = "".join(value.split())
            elif key == "True mask 1":
                record["truth"] = "".join(value.split())
            elif key == "Pred. label":
                record["predicted"] = int(value)
            elif key == "Real label":
                record["label"] = int(value)
            elif not line and record:
                strings.append(record["selected"])
                truths.append(record["truth"])
                correct += int(record["predicted"] == record["label"])
                record = {}
        seed = re.search(rf"masks_{re.escape(name)}_(\d+)", path.name).group(1)
        yield Selections(
            model, seed, correct / len(truths), strings, truths, None, None
        )


def per_class(selections: Selections) -> None:
    by_truth = defaultdict(list)
    for string, truth in zip(selections.strings, selections.truths):
        by_truth[truth].append(string)
    print(f"== {selections.model} seed {selections.seed}")
    code = {}
    for truth in HIGHLIGHTS:
        strings = by_truth[truth]
        sizes = dict(sorted(Counter(len(s) for s in strings).items()))
        common = Counter(strings).most_common(6)
        code[truth] = common[0][0]
        print(f"  {truth}: n {len(strings)}, sizes {sizes}")
        print(f"    most selected {common}")
    # The most frequent selection per class, and its length summed over the
    # classes: the code a run settled on, compared across runs by its length.
    entries = " ".join(f"{truth}={selected!r}" for truth, selected in code.items())
    print(f"  code: {entries}, {sum(map(len, code.values()))} characters")


def spread(values: Sequence[float]) -> str:
    if any(v != v for v in values):
        return "-"
    mean = 100 * statistics.mean(values)
    deviation = 100 * statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:6.2f} ± {deviation:5.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results", type=Path)
    parser.add_argument("--release", default=None, type=Path)
    parser.add_argument("--release-name", default="toy")
    parser.add_argument("--per-class", action="store_true")
    arguments = parser.parse_args()

    Registry.build(
        directory=Path(genspp2025.__file__).parent,
        external_directories=[Path(pyhighlights.__file__).parent],
    )
    splits = Registry.from_key(keys.TOY_FR_TASK).splits()
    sources = list(reproduction(arguments.results, splits))
    if arguments.release is not None:
        sources += list(release(arguments.release, arguments.release_name))
    if arguments.per_class:
        for selections in sources:
            per_class(selections)
        return

    readers = Readers(splits["train"])
    columns = (
        "own accuracy",
        "complete, positional",
        "complete, string",
        "reader, positional",
        "reader, string",
    )
    rows = defaultdict(list)
    per_reader = defaultdict(lambda: defaultdict(list))
    for s in sources:
        string_reads = [
            readers.accuracy(r, s.strings, s.truths) for r in readers.string
        ]
        complete_string = statistics.mean(map(complete_in_string, s.strings, s.truths))
        if s.views is None:
            positional_reads = [float("nan")] * len(READER_SEEDS)
            complete_view = float("nan")
        else:
            positional_reads = [
                readers.accuracy(r, s.views, s.truths) for r in readers.positional
            ]
            complete_view = statistics.mean(
                map(complete_in_view, s.views, s.gold_positions, s.truths)
            )
        for k, value in enumerate(string_reads):
            per_reader[(s.model, "string")][k].append(value)
        for k, value in enumerate(positional_reads):
            per_reader[(s.model, "positional")][k].append(value)
        rows[s.model].append(
            (
                s.own_accuracy,
                complete_view,
                complete_string,
                statistics.mean(positional_reads),
                statistics.mean(string_reads),
            )
        )

    print(f"{'model':15}" + "".join(f"{c:>22}" for c in columns))
    for model_name, values in rows.items():
        cells = "".join(f"{spread(column):>22}" for column in zip(*values))
        print(f"{model_name:15}{cells}")

    print("\nreader accuracy per reader seed, averaged over a model's runs")
    for (model_name, view), seeds in per_reader.items():
        means = [statistics.mean(v) for v in seeds.values()]
        if any(m != m for m in means):
            continue
        print(
            f"{model_name:15}{view:>11}: "
            + " ".join(f"{100 * m:5.1f}" for m in means)
            + f"   {spread(means)}"
        )


if __name__ == "__main__":
    main()
