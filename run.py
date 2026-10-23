"""Run one cell of the reproduction, or a whole corpus's table.

    python run.py toy                       # the five models, in order
    python run.py toy --model genspp        # one cell
    python run.py hatexplain --embeddings glove.twitter.27B.25d.txt
    python run.py toy --smoke               # one seed, one batch, minutes
    python run.py toy --model genspp --released-threshold
    python run.py toy --model genspp --seeds 1-20
    python run.py toy --model genspp --g150

`--smoke` bounds a baseline with ``trainer_args`` and the GenSPP cell with a
search of its own, since a search reads no trainer argument.

The registry is built over this package with the library beside it, so an edit
here runs without reinstalling anything, which is why the container installs
`pyhighlights` and not this repository.

`--released-threshold` searches Toy GenSPP the way the released implementation
mutates rather than the way its paper reports. `--seeds` replaces the cell's
seeds. `--g150` searches over 150 generations, Table 2's `GenSPP (G = 150)`
row.

Every run writes `<results>/<task>/<stamp>`, whatever its flags. A smoke run, a
run of other seeds and the paper's run share a task name, and `compare.py`
tells them apart by the search and the seeds each run's manifest records.
"""

import argparse
from pathlib import Path

import pyhighlights
from cinnamon.registry import Registry

import genspp2025

MODELS = ("fr", "mgr", "mcd", "grat", "genspp")

#: One batch and one seed: enough to say every cell builds, trains, scores and
#: writes its files, and not enough to mean anything.
SMOKE = {
    "seeds": (2023,),
    "trainer_args": {
        "accelerator": "auto",
        "devices": 1,
        "max_epochs": 1,
        "limit_train_batches": 1,
        "limit_val_batches": 1,
        "deterministic": True,
    },
}


def keys(corpus: str):
    """The corpus's task keys, by model name, imported only when asked for.

    Importing a corpus's keys imports its configurations, and the point of the
    package registering nothing on import is that a run says which half it
    wants.
    """
    module = corpus_keys(corpus)
    prefix = "TOY" if corpus == "toy" else "HATEXPLAIN"
    return {
        model: getattr(module, f"{prefix}_{model.upper()}_TASK") for model in MODELS
    }


def corpus_keys(corpus: str):
    if corpus == "toy":
        from genspp2025.configurations.toy import keys as module
    else:
        from genspp2025.configurations.hatexplain import keys as module
    return module


def smoke_search(corpus: str):
    """The search a smoke run uses instead of the paper's.

    ``--smoke`` bounds a baseline with ``trainer_args``, which Lightning
    reads. The search reads none of it, since it builds a trainer per candidate,
    so the GenSPP cell ran its fifty candidates over a hundred generations
    under a flag that promises minutes. The key below is two candidates and one
    generation, and every other setting is the paper's.
    """
    module = corpus_keys(corpus)
    prefix = "TOY" if corpus == "toy" else "HATEXPLAIN"
    return getattr(module, f"{prefix}_GENSPP_SMOKE_TRAINER")


def released_threshold_search(corpus: str):
    """The search that mutates the threshold as the release mutates it.

    The released GenSPP perturbs the selector's output bias at 0.10 where
    every other gene takes 0.05, which its paper does not report. Registered
    for Toy alone, which is the cell the difference is measured on.
    """
    if corpus != "toy":
        raise SystemExit("--released-threshold is registered for toy alone")
    from genspp2025.configurations.toy.keys import (
        TOY_GENSPP_RELEASED_THRESHOLD_TRAINER,
    )

    return TOY_GENSPP_RELEASED_THRESHOLD_TRAINER


def g150_search(corpus: str):
    """The search over 150 generations, Table 2's ``GenSPP (G = 150)`` row."""
    module = corpus_keys(corpus)
    prefix = "TOY" if corpus == "toy" else "HATEXPLAIN"
    return getattr(module, f"{prefix}_GENSPP_G150_TRAINER")


def seed_list(spec: str) -> list:
    """``1-20`` or ``1,2,5`` as a list of seeds."""
    if "-" in spec:
        first, last = (int(part) for part in spec.split("-"))
        if first > last:
            raise argparse.ArgumentTypeError(f"{spec} is an empty range")
        return list(range(first, last + 1))
    return [int(part) for part in spec.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", choices=("toy", "hatexplain"))
    parser.add_argument("--model", choices=MODELS, default=None)
    parser.add_argument(
        "--embeddings",
        default=None,
        help="path to glove.twitter.27B.25d.txt; HateXplain refuses without it",
    )
    parser.add_argument("--results", default=Path("results"), type=Path)
    parser.add_argument(
        "--seeds",
        default=None,
        help="seeds to run instead of the cell's, as 1-20 or 1,2,5",
    )
    # Either flag replaces the GenSPP cell's search, so naming both would have
    # to mean one of them silently losing.
    search = parser.add_mutually_exclusive_group()
    search.add_argument("--smoke", action="store_true")
    search.add_argument(
        "--released-threshold",
        action="store_true",
        help="search Toy GenSPP with the release's threshold mutation, not the paper's",
    )
    search.add_argument(
        "--g150",
        action="store_true",
        help="search GenSPP over 150 generations, Table 2's G = 150 row",
    )
    arguments = parser.parse_args()

    if arguments.corpus == "hatexplain" and not arguments.embeddings:
        raise SystemExit(
            "HateXplain needs --embeddings: the 1.4 GB GloVe file is a download "
            "the paper expects you to fetch, so the task takes a path"
        )

    valid, invalid = Registry.build(
        directory=Path(genspp2025.__file__).parent,
        external_directories=[Path(pyhighlights.__file__).parent],
    )
    if invalid:
        raise SystemExit(f"invalid registrations: {sorted(str(k) for k in invalid)}")

    if arguments.seeds and arguments.smoke:
        raise SystemExit("--smoke runs one seed of its own, so it takes no --seeds")
    extra = dict(SMOKE) if arguments.smoke else {}
    if arguments.seeds:
        try:
            extra["seeds"] = seed_list(arguments.seeds)
        except (ValueError, argparse.ArgumentTypeError) as error:
            raise SystemExit(f"--seeds {arguments.seeds}: {error}") from error
    if arguments.g150:
        if arguments.model != "genspp":
            raise SystemExit("--g150 is a GenSPP search, so it needs --model genspp")
    if arguments.embeddings:
        extra["embeddings"] = arguments.embeddings

    chosen = keys(arguments.corpus)
    wanted = [arguments.model] if arguments.model else list(MODELS)
    for model in wanted:
        print(f"=== {arguments.corpus}-{model} ===", flush=True)
        cell = dict(extra)
        if model == "genspp" and arguments.smoke:
            cell["search"] = smoke_search(arguments.corpus)
        elif model == "genspp" and arguments.released_threshold:
            cell["search"] = released_threshold_search(arguments.corpus)
        elif model == "genspp" and arguments.g150:
            cell["search"] = g150_search(arguments.corpus)
        Registry.from_key(chosen[model], save_path=str(arguments.results), **cell).run()


if __name__ == "__main__":
    main()
