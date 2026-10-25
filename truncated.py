"""GenSPP's HateXplain highlight F1 as the release scores it.

Usage::

    python truncated.py --results results --embeddings glove.twitter.27B.25d.txt

The release builds its GenSPP vocabulary from the training and validation
posts, and embeds a test token outside it as a zero vector. Its
``get_texts_len`` then ends a post at the first all-zero embedding, so the
release scores a test post only up to its first token unseen in training. This
script scores the newest ``hatexplain-genspp`` run both ways from its
checkpoints: over every token, which is the library's score and must equal the
reported one, and over that prefix.

The baselines need no rescoring, because the release scores them over every
token as the library does.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Collection, Sequence, Tuple

import pyhighlights
import torch as th
from cinnamon.registry import Registry

import genspp2025
from genspp2025.configurations.hatexplain import keys


def scored_length(tokens: Sequence[str], seen: Collection[str]) -> int:
    """How many leading tokens the release scores: up to the first unseen."""
    return next((i for i, token in enumerate(tokens) if token not in seen), len(tokens))


def counts(
    predicted: Sequence[bool], gold: Sequence[int], end: int
) -> Tuple[int, int, int]:
    """True positives, false positives and false negatives over ``[0, end)``."""
    pairs = list(zip(predicted[:end], gold[:end]))
    return (
        sum(p and g == 1 for p, g in pairs),
        sum(p and g != 1 for p, g in pairs),
        sum(not p and g == 1 for p, g in pairs),
    )


def f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / (2 * tp + fp + fn)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results", type=Path)
    parser.add_argument("--embeddings", required=True)
    arguments = parser.parse_args()

    Registry.build(
        directory=Path(genspp2025.__file__).parent,
        external_directories=[Path(pyhighlights.__file__).parent],
    )
    task = Registry.from_key(
        keys.HATEXPLAIN_GENSPP_TASK, embeddings=arguments.embeddings
    )
    splits = task.splits()
    seen = {
        token
        for name, frame in splits.items()
        if name != "test"
        for tokens in frame["tokens"]
        for token in tokens
    }
    frame = splits["test"].set_index("sample_id")
    lengths = [len(tokens) for tokens in frame["tokens"]]
    kept = [scored_length(tokens, seen) for tokens in frame["tokens"]]
    print(f"the release scores {sum(kept) / sum(lengths):.1%} of the test tokens")
    loader = task.loaders(splits)["test"]

    directory = arguments.results / "hatexplain-genspp"
    run = sorted(path for path in directory.iterdir() if path.is_dir())[-1]
    stored = json.loads((run / "results.json").read_text())
    reported = {
        str(seed): values["test_highlight_f1"]
        for seed, values in zip(stored["completed"], stored["runs"])
    }

    whole, truncated = [], []
    for checkpoint in sorted(run.glob("seed=*/*.ckpt")):
        seed = checkpoint.parent.name.split("=")[1]
        model = task.build_model()
        state = th.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["state_dict"], strict=True)
        model.eval()
        every, prefix = [0, 0, 0], [0, 0, 0]
        with th.no_grad():
            for batch in loader:
                output = model.reported(model(batch))
                for r in range(batch.y_true.shape[0]):
                    row = frame.loc[int(batch.sample_ids[r])]
                    tokens, gold = list(row["tokens"]), list(row["highlights"])
                    predicted = output.highlight_mask[r].bool().tolist()
                    for total, end in (
                        (every, len(tokens)),
                        (prefix, scored_length(tokens, seen)),
                    ):
                        for i, n in enumerate(counts(predicted, gold, end)):
                            total[i] += n
        whole.append(f1(*every))
        truncated.append(f1(*prefix))
        # The library's own score, recomputed: a mismatch means the model or
        # the vocabulary was rebuilt differently from the run.
        if abs(whole[-1] - reported[seed]) > 1e-4:
            raise SystemExit(
                f"seed {seed}: recomputed {whole[-1]:.4f}, "
                f"reported {reported[seed]:.4f}"
            )
        print(
            f"seed {seed}: every token {whole[-1]:.4f}, "
            f"release prefix {truncated[-1]:.4f}"
        )

    for name, values in (("every token", whole), ("release prefix", truncated)):
        mean = 100 * statistics.mean(values)
        deviation = 100 * statistics.stdev(values)
        print(f"{name}: {mean:.2f} ± {deviation:.2f}")


if __name__ == "__main__":
    main()
