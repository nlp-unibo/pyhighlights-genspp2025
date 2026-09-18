"""The genetic search over GenSPP's generator."""

from typing import List

from cinnamon.configuration import Param
from cinnamon.registry import RegistrationKey, register_class
from pyhighlights.components.models.spp.genspp import GenSPP
from pyhighlights.configurations.genspp import GRUGenSPPTrainerConfig

from genspp2025.configurations.keys import (
    NAMESPACE,
    WORKERS,
)
from genspp2025.configurations.toy.keys import (
    TOY_GENSPP,
)


@register_class(
    name="trainer",
    tags={"genspp", "toy"},
    namespace=NAMESPACE,
    component="pyhighlights.components.models.spp.genspp.GenSPPTrainer",
)
class ToyGenSPPTrainerConfig(GRUGenSPPTrainerConfig):
    """The search as released: the expected cross entropy is 0.1 here."""

    model: RegistrationKey[GenSPP] = Param(TOY_GENSPP)
    task_loss_limit: float = Param(0.1, ge=0.0)
    devices: List[str] = Param(WORKERS)


@register_class(
    name="trainer",
    tags={"genspp", "toy", "smoke"},
    namespace=NAMESPACE,
    component="pyhighlights.components.models.spp.genspp.GenSPPTrainer",
)
class ToyGenSPPSmokeTrainerConfig(ToyGenSPPTrainerConfig):
    """The search, cut to the smallest one that still exercises every step.

    ``--smoke`` bounds a baseline through ``trainer_args``, which is what
    Lightning reads. A search reads none of it: it builds a trainer per
    candidate, so the paper's fifty candidates over a hundred generations ran
    in full under a flag that promises minutes. Two candidates and one
    generation still cross, mutate, score and serialize.
    """

    n_generations: int = Param(1, ge=0)
    population_size: int = Param(2, ge=2)
    predictor_epochs: int = Param(1, ge=1)
