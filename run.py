"""Run one cell of the reproduction, or a whole corpus's table.

    python run.py toy                       # the five models, in order
    python run.py toy --model genspp        # one cell
    python run.py hatexplain --embeddings glove.twitter.27B.25d.txt
    python run.py toy --smoke               # one seed, one batch, minutes

The registry is built over this package with the library beside it, so an edit
here runs without reinstalling anything -- which is why the container installs
`pyhighlights` and not this repository.

`--smoke` writes under `results/smoke/` rather than beside the real runs.
`MetricsAnalyzer` reports the newest run per task name, and a smoke result
carries the same names, so writing it alongside would take a real number out
of the table.
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
    "trainer_args": {"max_epochs": 1, "limit_train_batches": 1, "limit_val_batches": 1},
}


def keys(corpus: str):
    """The corpus's task keys, by model name, imported only when asked for.

    Importing a corpus's keys imports its configurations, and the point of the
    package registering nothing on import is that a run says which half it
    wants.
    """
    if corpus == "toy":
        from genspp2025.configurations.toy import keys as module

        return {model: getattr(module, f"TOY_{model.upper()}_TASK") for model in MODELS}
    from genspp2025.configurations.hatexplain import keys as module

    return {
        model: getattr(module, f"HATEXPLAIN_{model.upper()}_TASK") for model in MODELS
    }


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
    parser.add_argument("--smoke", action="store_true")
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

    save_path = arguments.results / ("smoke" if arguments.smoke else "")
    extra = dict(SMOKE) if arguments.smoke else {}
    if arguments.embeddings:
        extra["embeddings"] = arguments.embeddings

    chosen = keys(arguments.corpus)
    wanted = [arguments.model] if arguments.model else list(MODELS)
    for model in wanted:
        print(f"=== {arguments.corpus}-{model} ===", flush=True)
        Registry.from_key(chosen[model], save_path=str(save_path), **extra).run()


if __name__ == "__main__":
    main()
