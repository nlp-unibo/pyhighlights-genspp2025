"""Settings both corpora share, and the metric sets they report.

Every baseline trains the same way in the released implementation: Adam at
1e-3, batches of 64, up to 500 epochs, early stopping on validation loss after
30 worse ones, and five seeds. Only the corpus and the model change.

The genetic search of both corpora shares its worker pool. A smoke task cuts
any of these tasks to one seed and one batch.
"""

from typing import Any, Dict, List, Sequence

from cinnamon.configuration import Configuration, Param
from cinnamon.registry import RegistrationKey, register_class
from lightning.pytorch.callbacks import Callback
from pyhighlights.components.loaders import HighlightLoader
from pyhighlights.components.models.base import Model
from pyhighlights.components.models.spp.genspp import GenSPPTrainer
from pyhighlights.components.preprocessors import Preprocessor
from pyhighlights.configurations.genspp import GRUGenSPPTrainerConfig
from pyhighlights.configurations.keys import (
    F1_METRIC,
    HIGHLIGHT_F1_METRIC,
    MULTICLASS_F1_METRIC,
    SELECTION_RATE_METRIC,
    SELECTION_SIZE_METRIC,
)
from pyhighlights.configurations.tasks import GenSPPTaskConfig, TaskConfig
from pyhighlights.utility.metrics import BoundMetric

from genspp2025.configurations.keys import (
    PAPER_CHECKPOINT,
    PAPER_EARLY_STOPPING,
)

BINARY_METRICS = [
    F1_METRIC,
    HIGHLIGHT_F1_METRIC,
    SELECTION_RATE_METRIC,
    SELECTION_SIZE_METRIC,
]

THREE_CLASS_METRICS = [
    MULTICLASS_F1_METRIC,
    HIGHLIGHT_F1_METRIC,
    SELECTION_RATE_METRIC,
    SELECTION_SIZE_METRIC,
]


class PaperTaskConfig(TaskConfig):
    """How every baseline in the paper is trained.

    Left abstract: a corpus half fills in ``loader`` and a model half fills in
    ``model``. The fields below are the same for all of them, and every other
    field keeps the library's default. ``save_path`` is left null, so a
    benchmark tells each task to write inside its own directory.
    """

    #: The corpus, as distributed. What the paper does to it is
    #: ``preprocessor``'s business.
    loader: RegistrationKey[HighlightLoader] = Param(None)
    #: The architecture under test, the one thing a row of the paper's table
    #: varies.
    model: RegistrationKey[Model] = Param(None)
    #: Filtering, label folding and vote aggregation. ``None`` for a corpus
    #: the paper takes as it comes.
    preprocessor: RegistrationKey[Preprocessor] | None = Param(None)
    #: Scored at the end of every epoch, and what the early stopping rule
    #: watches through ``val_loss``.
    val_metrics: List[RegistrationKey[BoundMetric]] = Param(BINARY_METRICS)
    #: Scored once, on the epoch the checkpoint kept. These are the numbers in
    #: the table.
    test_metrics: List[RegistrationKey[BoundMetric]] = Param(BINARY_METRICS)
    #: Five runs per cell, on the seeds the release's baselines use. A single
    #: run of a select-then-predict model says very little, so the spread is
    #: part of the result.
    seeds: Sequence[int] = Param([2023, 15451, 1337, 2001, 2080])
    #: Sixty-four, for every architecture and both corpora.
    batch_size: int = Param(64, ge=1)
    #: Early stopping and checkpointing on the validation loss, with the
    #: paper's patience of thirty rather than the library's five. Both monitor
    #: the same quantity, so the epoch that stops a run is the epoch it is
    #: scored on.
    callbacks: List[RegistrationKey[Callback]] = Param(
        [PAPER_EARLY_STOPPING, PAPER_CHECKPOINT]
    )
    #: Five hundred epochs is a budget, not a schedule: the early stopping
    #: rule above ends a run long before it. Deterministic, as every released
    #: baseline runnable sets it, so a seed reruns to the same numbers.
    trainer_args: Dict[str, Any] = Param(
        {"accelerator": "auto", "devices": 1, "max_epochs": 500, "deterministic": True}
    )


class PaperGenSPPTaskConfig(GenSPPTaskConfig):
    """GenSPP's own task: a search, not five hundred epochs of descent.

    The fields it shares with :class:`PaperTaskConfig` mean the same things.
    It names no ``model``, because the search names its own. A corpus half
    fills in ``loader`` and ``search``.
    """

    #: The genetic search, which carries the model key and every setting of
    #: the search itself.
    search: RegistrationKey[GenSPPTrainer] = Param(None)
    loader: RegistrationKey[HighlightLoader] = Param(None)
    #: Nothing is monitored across epochs when the generator is searched
    #: rather than trained, so a run records no callbacks.
    callbacks: List[RegistrationKey[Callback]] | None = Param(None)

    val_metrics: List[RegistrationKey[BoundMetric]] = Param(BINARY_METRICS)
    test_metrics: List[RegistrationKey[BoundMetric]] = Param(BINARY_METRICS)
    #: The release's ``train.py`` seeds run ``i`` with ``i + 1``, on both
    #: corpora. These differ from the baselines' seeds.
    seeds: Sequence[int] = Param([1, 2, 3, 4, 5])
    batch_size: int = Param(64, ge=1)
    trainer_args: Dict[str, Any] = Param({"accelerator": "auto", "devices": 1})


class SmokeTaskConfig(Configuration):
    """A task cut to one seed and one batch, placed before a paper task's base.

    Enough to say a cell builds, trains, scores and writes its files, and not
    enough to mean anything. Every other setting is the paper task's. A search
    reads none of these trainer arguments, so a smoke GenSPP task also names
    a smoke search.
    """

    seeds: Sequence[int] = Param([2023])
    trainer_args: Dict[str, Any] = Param(
        {
            "accelerator": "auto",
            "devices": 1,
            "max_epochs": 1,
            "limit_train_batches": 1,
            "limit_val_batches": 1,
            "deterministic": True,
        }
    )


class PaperGenSPPTrainerConfig(GRUGenSPPTrainerConfig):
    """The genetic search's settings both corpora share."""

    #: Candidates a genetic search scores at once, one per worker, as the
    #: release's own pool does and as ``cluster/run.sbatch`` allocates
    #: (``--cpus-per-task=8``). Every entry is a CPU device, which is what lets
    #: pyhighlights score them in spawned worker processes. The search uses
    #: processes only when every device is a CPU, so one CUDA device here would
    #: put the search back on threads. It buys less than eight times the speed,
    #: since a candidate is small and most of its cost is the training loop
    #: stepping from Python. The library measured sixteen candidates of the toy
    #: search on a 24-core machine at 1442 ms a candidate sequentially, 832 ms
    #: on eight threads and 293 ms on eight processes.
    devices: List[str] = Param(["cpu"] * 8)


@register_class(
    key=PAPER_EARLY_STOPPING,
    component="pyhighlights.components.callbacks.WarmupEarlyStopping",
)
class PaperEarlyStoppingConfig(Configuration):
    """Thirty epochs of patience, which is the paper's and not the library's."""

    monitor: str = Param("val_loss")
    mode: str = Param("min")
    patience: int = Param(30, ge=0)


@register_class(
    key=PAPER_CHECKPOINT,
    component="pyhighlights.components.callbacks.WarmupModelCheckpoint",
)
class PaperCheckpointConfig(Configuration):
    """The same quantity the run stops on, so the two cannot disagree."""

    monitor: str = Param("val_loss")
    mode: str = Param("min")
