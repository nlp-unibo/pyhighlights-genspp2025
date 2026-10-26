"""The genetic search over GenSPP's generator."""

from typing import List

from cinnamon.configuration import Param
from cinnamon.registry import RegistrationKey, register_class
from pyhighlights.components.models.spp.genspp import GenSPP
from pyhighlights.configurations.genspp import GRUGenSPPTrainerConfig

from genspp2025.configurations.hatexplain.keys import (
    HATEXPLAIN_GENSPP,
    HATEXPLAIN_GENSPP_G150_TRAINER,
    HATEXPLAIN_GENSPP_SMOKE_TRAINER,
    HATEXPLAIN_GENSPP_TRAINER,
)
from genspp2025.configurations.keys import WORKERS


@register_class(
    key=HATEXPLAIN_GENSPP_TRAINER,
    component="pyhighlights.components.models.spp.genspp.GenSPPTrainer",
)
class HateXplainGenSPPTrainerConfig(GRUGenSPPTrainerConfig):
    """The genetic search over GenSPP's generator, at the paper's settings."""

    model: RegistrationKey[GenSPP] = Param(HATEXPLAIN_GENSPP)
    #: Cross entropy above which a candidate scores the fitness floor whatever
    #: it selected, so sparsity cannot be bought with a model that has stopped
    #: classifying. Real-world text does not reach the loss a synthetic corpus
    #: does, so the bar is set where a working classifier actually sits.
    task_loss_limit: float = Param(0.6, ge=0.0)
    devices: List[str] = Param(WORKERS)


@register_class(
    key=HATEXPLAIN_GENSPP_SMOKE_TRAINER,
    component="pyhighlights.components.models.spp.genspp.GenSPPTrainer",
)
class HateXplainGenSPPSmokeTrainerConfig(HateXplainGenSPPTrainerConfig):
    """The search, cut to the smallest one that still exercises every step.

    The search reads no ``trainer_args``, so nothing ``--smoke`` passes ever
    reached it. The toy half's smoke search says the rest.
    """

    n_generations: int = Param(1, ge=0)
    population_size: int = Param(2, ge=2)
    predictor_epochs: int = Param(1, ge=1)


@register_class(
    key=HATEXPLAIN_GENSPP_G150_TRAINER,
    component="pyhighlights.components.models.spp.genspp.GenSPPTrainer",
)
class HateXplainGenSPPG150TrainerConfig(HateXplainGenSPPTrainerConfig):
    """The search over 150 generations, the ``GenSPP (G = 150)`` row of Table 2.

    The paper's search in every other respect.
    """

    n_generations: int = Param(150, ge=0)
