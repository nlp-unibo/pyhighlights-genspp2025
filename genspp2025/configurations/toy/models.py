"""The five architectures the paper puts on this corpus, and the parts they share."""

from typing import List

from cinnamon.configuration import Param
from cinnamon.registry import RegistrationKey, register_class
from pyhighlights.components.models.spp.base import SPPBackbone, SPPPredictor
from pyhighlights.configurations.backbones import MLPPredictorConfig
from pyhighlights.configurations.fr import GRUFRConfig
from pyhighlights.configurations.genspp import GRUGenSPPConfig
from pyhighlights.configurations.grat import GRUGRATConfig, GuidanceGuiderConfig
from pyhighlights.configurations.mcd import GRUMCDConfig
from pyhighlights.configurations.mgr import GRUMGRConfig

from genspp2025.configurations.toy import (
    EMBEDDING_DIM,
    VOCABULARY_SIZE,
)
from genspp2025.configurations.toy.keys import (
    TOY_BACKBONE,
    TOY_FR,
    TOY_GENSPP,
    TOY_GENSPP_BACKBONE,
    TOY_GRAT,
    TOY_GUIDER,
    TOY_MCD,
    TOY_MGR,
    TOY_PREDICTOR,
    TOY_UNNORMALISED_BACKBONE,
)


@register_class(
    key=TOY_PREDICTOR,
    component="pyhighlights.components.models.spp.implementations.MLPPredictor",
)
class ToyPredictorConfig(MLPPredictorConfig):
    num_classes: int = Param(3, ge=2)


@register_class(
    key=TOY_GUIDER, component="pyhighlights.components.models.spp.grat.GuidanceGuider"
)
class ToyGuiderConfig(GuidanceGuiderConfig):
    """The original guidance module, sized as the baselines' encoder."""

    vocab_size: int = Param(VOCABULARY_SIZE, ge=3)
    embedding_dim: int = Param(EMBEDDING_DIM, ge=1)
    hidden_size: int = Param(8, ge=1)
    predictor: RegistrationKey[SPPPredictor] = Param(TOY_PREDICTOR)
    freeze_embeddings: bool | None = Param(True)


@register_class(key=TOY_FR, component="pyhighlights.components.models.spp.fr.FR")
class ToyFRConfig(GRUFRConfig):
    selector_backbones: RegistrationKey[SPPBackbone] = Param(TOY_BACKBONE)
    predictor: RegistrationKey[SPPPredictor] = Param(TOY_PREDICTOR)


@register_class(key=TOY_MGR, component="pyhighlights.components.models.spp.mgr.MGR")
class ToyMGRConfig(GRUMGRConfig):
    selector_backbones: List[RegistrationKey[SPPBackbone]] = Param(
        [TOY_BACKBONE, TOY_BACKBONE, TOY_BACKBONE]
    )
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(TOY_UNNORMALISED_BACKBONE)
    predictor: RegistrationKey[SPPPredictor] = Param(TOY_PREDICTOR)


@register_class(key=TOY_MCD, component="pyhighlights.components.models.spp.mcd.MCD")
class ToyMCDConfig(GRUMCDConfig):
    selector_backbones: RegistrationKey[SPPBackbone] = Param(TOY_BACKBONE)
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(TOY_UNNORMALISED_BACKBONE)
    predictor: RegistrationKey[SPPPredictor] = Param(TOY_PREDICTOR)
    #: The setup's validation total leaves out the full input's cross entropy.
    monitored_losses: List[str] | None = Param(
        ["classification", "sparsity", "contiguity", "discrepancy"]
    )


@register_class(key=TOY_GRAT, component="pyhighlights.components.models.spp.grat.GRAT")
class ToyGRATConfig(GRUGRATConfig):
    selector_backbones: RegistrationKey[SPPBackbone] = Param(TOY_BACKBONE)
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(TOY_BACKBONE)
    predictor: RegistrationKey[SPPPredictor] = Param(TOY_PREDICTOR)
    guider: RegistrationKey = Param(TOY_GUIDER)
    guide_decay: float = Param(1e-5, ge=0.0)
    pretrain_epochs: int = Param(10, ge=0)
    #: Sampled at validation and test, as FR, MGR and MCD are, so the four
    #: baselines are scored under one inference strategy. The original
    #: G-RAT thresholds its selection at 0.5 there, and the release samples.
    sample_at_evaluation: bool = Param(True)


@register_class(
    key=TOY_GENSPP, component="pyhighlights.components.models.spp.genspp.GenSPP"
)
class ToyGenSPPConfig(GRUGenSPPConfig):
    selector_backbones: RegistrationKey[SPPBackbone] = Param(TOY_GENSPP_BACKBONE)
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(TOY_GENSPP_BACKBONE)
    predictor: RegistrationKey[SPPPredictor] = Param(TOY_PREDICTOR)
