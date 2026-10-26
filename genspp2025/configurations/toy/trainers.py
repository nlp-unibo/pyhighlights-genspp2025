"""The genetic search over GenSPP's generator."""

from typing import List

from cinnamon.configuration import Param
from cinnamon.registry import RegistrationKey, register_class
from pyhighlights.components.models.spp.genspp import GenSPP
from pyhighlights.configurations.genspp import GRUGenSPPTrainerConfig

from genspp2025.configurations.keys import WORKERS
from genspp2025.configurations.toy.keys import (
    TOY_GENSPP,
    TOY_GENSPP_G150_TRAINER,
    TOY_GENSPP_RELEASED_THRESHOLD_TRAINER,
    TOY_GENSPP_SMOKE_TRAINER,
    TOY_GENSPP_TRAINER,
)


@register_class(
    key=TOY_GENSPP_TRAINER,
    component="pyhighlights.components.models.spp.genspp.GenSPPTrainer",
)
class ToyGenSPPTrainerConfig(GRUGenSPPTrainerConfig):
    """The search as released: the expected cross entropy is 0.1 here."""

    model: RegistrationKey[GenSPP] = Param(TOY_GENSPP)
    task_loss_limit: float = Param(0.1, ge=0.0)
    devices: List[str] = Param(WORKERS)


@register_class(
    key=TOY_GENSPP_RELEASED_THRESHOLD_TRAINER,
    component="pyhighlights.components.models.spp.genspp.GenSPPTrainer",
)
class ToyGenSPPReleasedThresholdTrainerConfig(ToyGenSPPTrainerConfig):
    """The search with the threshold explored at the rate the release uses.

    The release mutates the selector's output bias at 0.10 where every other
    gene takes 0.05, which its paper does not report. This key exists to
    measure what that difference is worth on Toy.
    """

    threshold_mutation_std: float = Param(0.10, gt=0.0)


@register_class(
    key=TOY_GENSPP_SMOKE_TRAINER,
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


@register_class(
    key=TOY_GENSPP_G150_TRAINER,
    component="pyhighlights.components.models.spp.genspp.GenSPPTrainer",
)
class ToyGenSPPG150TrainerConfig(ToyGenSPPTrainerConfig):
    """The search over 150 generations, the ``GenSPP (G = 150)`` row of Table 2.

    The paper's search in every other respect.
    """

    n_generations: int = Param(150, ge=0)
