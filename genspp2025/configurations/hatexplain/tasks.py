"""One task per architecture: what a run of this corpus is."""

from cinnamon.configuration import Configuration, Param
from cinnamon.registry import RegistrationKey, register_class
from pyhighlights.components.models.base import Model
from pyhighlights.components.preprocessors import Preprocessor
from pyhighlights.configurations.keys import (
    HATEXPLAIN,
)

from genspp2025.configurations.common import (
    PaperGenSPPTaskConfig,
    PaperTaskConfig,
    SmokeTaskConfig,
)
from genspp2025.configurations.hatexplain.keys import (
    HATEXPLAIN_FR,
    HATEXPLAIN_FR_SMOKE_TASK,
    HATEXPLAIN_FR_TASK,
    HATEXPLAIN_GENSPP_SMOKE_TASK,
    HATEXPLAIN_GENSPP_SMOKE_TRAINER,
    HATEXPLAIN_GENSPP_TASK,
    HATEXPLAIN_GENSPP_TRAINER,
    HATEXPLAIN_GRAT,
    HATEXPLAIN_GRAT_SMOKE_TASK,
    HATEXPLAIN_GRAT_TASK,
    HATEXPLAIN_MCD,
    HATEXPLAIN_MCD_SMOKE_TASK,
    HATEXPLAIN_MCD_TASK,
    HATEXPLAIN_MGR,
    HATEXPLAIN_MGR_SMOKE_TASK,
    HATEXPLAIN_MGR_TASK,
    HATEXPLAIN_PIPELINE,
)

SPP_TASK_COMPONENT = "pyhighlights.components.tasks.SPPTask"


class HateXplainCorpusConfig(Configuration):
    """What every task over HateXplain reads, baseline or search."""

    loader: RegistrationKey = Param(HATEXPLAIN)
    preprocessor: RegistrationKey[Preprocessor] | None = Param(HATEXPLAIN_PIPELINE)
    #: The GloVe file, which the task is given rather than fetching. Building
    #: without it is refused.
    embeddings: str | None = Param(None)
    #: Every token of every split gets a row of its own: its GloVe vector, or
    #: a uniform(-0.05, 0.05) row. The released collator maps a token GloVe
    #: lacks to ``<PAD>``, a zero row, which is a release error documented in
    #: the package docstring rather than reproduced.
    vocabulary_from: str = Param("splits")
    pretrained_tokens_only: bool = Param(False)


class HateXplainTaskConfig(HateXplainCorpusConfig, PaperTaskConfig):
    """One baseline over HateXplain, with the paper's preprocessing."""


@register_class(key=HATEXPLAIN_FR_TASK, component=SPP_TASK_COMPONENT, run_method="run")
class HateXplainFRTaskConfig(HateXplainTaskConfig):
    name: str = Param("hatexplain-fr")
    model: RegistrationKey[Model] = Param(HATEXPLAIN_FR)


@register_class(key=HATEXPLAIN_MGR_TASK, component=SPP_TASK_COMPONENT, run_method="run")
class HateXplainMGRTaskConfig(HateXplainTaskConfig):
    name: str = Param("hatexplain-mgr")
    model: RegistrationKey[Model] = Param(HATEXPLAIN_MGR)


@register_class(key=HATEXPLAIN_MCD_TASK, component=SPP_TASK_COMPONENT, run_method="run")
class HateXplainMCDTaskConfig(HateXplainTaskConfig):
    name: str = Param("hatexplain-mcd")
    model: RegistrationKey[Model] = Param(HATEXPLAIN_MCD)


@register_class(
    key=HATEXPLAIN_GRAT_TASK, component=SPP_TASK_COMPONENT, run_method="run"
)
class HateXplainGRATTaskConfig(HateXplainTaskConfig):
    name: str = Param("hatexplain-grat")
    model: RegistrationKey[Model] = Param(HATEXPLAIN_GRAT)


@register_class(
    key=HATEXPLAIN_GENSPP_TASK,
    component="pyhighlights.components.tasks.GenSPPTask",
    run_method="run",
)
class HateXplainGenSPPTaskConfig(HateXplainCorpusConfig, PaperGenSPPTaskConfig):
    """GenSPP over HateXplain, embedded as the four baselines are.

    The genetic half of the release builds its vocabulary from the training
    split and looks a token id back up in a detokenizer built from it
    (``genetic/src/preprocessing/dataset_reader/dataset.py``). Every training
    token GloVe lacks shares one ``uniform(-0.05, 0.05)`` vector, and a token
    the training split never saw resolves to ``None`` and is embedded as
    zeros. Both collapses are release errors. Here every token of every split
    gets a row of its own, as for the baselines.
    """

    name: str = Param("hatexplain-genspp")
    search: RegistrationKey = Param(HATEXPLAIN_GENSPP_TRAINER)
    requires_embeddings: bool = Param(True)


@register_class(
    key=HATEXPLAIN_FR_SMOKE_TASK,
    component=SPP_TASK_COMPONENT,
    run_method="run",
)
class HateXplainFRSmokeTaskConfig(SmokeTaskConfig, HateXplainFRTaskConfig):
    pass


@register_class(
    key=HATEXPLAIN_MGR_SMOKE_TASK,
    component=SPP_TASK_COMPONENT,
    run_method="run",
)
class HateXplainMGRSmokeTaskConfig(SmokeTaskConfig, HateXplainMGRTaskConfig):
    pass


@register_class(
    key=HATEXPLAIN_MCD_SMOKE_TASK,
    component=SPP_TASK_COMPONENT,
    run_method="run",
)
class HateXplainMCDSmokeTaskConfig(SmokeTaskConfig, HateXplainMCDTaskConfig):
    pass


@register_class(
    key=HATEXPLAIN_GRAT_SMOKE_TASK,
    component=SPP_TASK_COMPONENT,
    run_method="run",
)
class HateXplainGRATSmokeTaskConfig(SmokeTaskConfig, HateXplainGRATTaskConfig):
    pass


@register_class(
    key=HATEXPLAIN_GENSPP_SMOKE_TASK,
    component="pyhighlights.components.tasks.GenSPPTask",
    run_method="run",
)
class HateXplainGenSPPSmokeTaskConfig(SmokeTaskConfig, HateXplainGenSPPTaskConfig):
    search: RegistrationKey = Param(HATEXPLAIN_GENSPP_SMOKE_TRAINER)
