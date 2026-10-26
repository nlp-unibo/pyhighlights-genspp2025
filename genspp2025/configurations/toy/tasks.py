"""One task per architecture: what a run of this corpus is."""

from typing import List

from cinnamon.configuration import Param
from cinnamon.registry import RegistrationKey, register_class
from pyhighlights.components.models.base import Model

from genspp2025.configurations.common import (
    THREE_CLASS_METRICS,
    PaperGenSPPTaskConfig,
    PaperTaskConfig,
)
from genspp2025.configurations.toy import (
    EMBEDDING_DIM,
    VOCABULARY_SIZE,
)
from genspp2025.configurations.toy.keys import (
    TOY,
    TOY_FR,
    TOY_FR_TASK,
    TOY_GENSPP_TASK,
    TOY_GENSPP_TRAINER,
    TOY_GRAT,
    TOY_GRAT_TASK,
    TOY_MCD,
    TOY_MCD_TASK,
    TOY_MGR,
    TOY_MGR_TASK,
)

SPP_TASK_COMPONENT = "pyhighlights.components.tasks.SPPTask"


class ToyTaskConfig(PaperTaskConfig):
    """One baseline over the toy corpus."""

    loader: RegistrationKey = Param(TOY)
    val_metrics: List[RegistrationKey] = Param(THREE_CLASS_METRICS)
    test_metrics: List[RegistrationKey] = Param(THREE_CLASS_METRICS)
    vocabulary_size: int = Param(VOCABULARY_SIZE, ge=2)
    #: One-hot inputs, as the release reads this corpus. One column per
    #: character, matching the backbone's ``embedding_dim``.
    one_hot_embeddings: int = Param(EMBEDDING_DIM, ge=1)


@register_class(key=TOY_FR_TASK, component=SPP_TASK_COMPONENT, run_method="run")
class ToyFRTaskConfig(ToyTaskConfig):
    name: str = Param("toy-fr")
    model: RegistrationKey[Model] = Param(TOY_FR)


@register_class(key=TOY_MGR_TASK, component=SPP_TASK_COMPONENT, run_method="run")
class ToyMGRTaskConfig(ToyTaskConfig):
    name: str = Param("toy-mgr")
    model: RegistrationKey[Model] = Param(TOY_MGR)


@register_class(key=TOY_MCD_TASK, component=SPP_TASK_COMPONENT, run_method="run")
class ToyMCDTaskConfig(ToyTaskConfig):
    name: str = Param("toy-mcd")
    model: RegistrationKey[Model] = Param(TOY_MCD)


@register_class(key=TOY_GRAT_TASK, component=SPP_TASK_COMPONENT, run_method="run")
class ToyGRATTaskConfig(ToyTaskConfig):
    name: str = Param("toy-grat")
    model: RegistrationKey[Model] = Param(TOY_GRAT)


@register_class(
    key=TOY_GENSPP_TASK,
    component="pyhighlights.components.tasks.GenSPPTask",
    run_method="run",
)
class ToyGenSPPTaskConfig(PaperGenSPPTaskConfig):
    name: str = Param("toy-genspp")
    loader: RegistrationKey = Param(TOY)
    search: RegistrationKey = Param(TOY_GENSPP_TRAINER)
    val_metrics: List[RegistrationKey] = Param(THREE_CLASS_METRICS)
    test_metrics: List[RegistrationKey] = Param(THREE_CLASS_METRICS)
    vocabulary_size: int = Param(VOCABULARY_SIZE, ge=2)
    #: The same table the baselines read. The genetic half of the release
    #: declares 26 columns for the same twenty-four characters.
    one_hot_embeddings: int = Param(EMBEDDING_DIM, ge=1)
