"""The five architectures the paper puts on this corpus, and the parts they share."""

from typing import List

from cinnamon.configuration import Param
from cinnamon.registry import RegistrationKey, register_class
from pyhighlights.components.models.spp.base import SPPBackbone
from pyhighlights.configurations.fr import GRUFRConfig
from pyhighlights.configurations.genspp import GRUGenSPPConfig
from pyhighlights.configurations.grat import GRUGRATConfig, GuidanceGuiderConfig
from pyhighlights.configurations.keys import (
    CLASSIFICATION_LOSS,
)
from pyhighlights.configurations.mcd import GRUMCDConfig
from pyhighlights.configurations.mgr import GRUMGRConfig
from pyhighlights.utility.losses import Loss

from genspp2025.configurations.hatexplain.keys import (
    HATEXPLAIN_BACKBONE,
    HATEXPLAIN_GENSPP_BACKBONE,
    HATEXPLAIN_GUIDE_LOSS,
    HATEXPLAIN_GUIDER,
    HATEXPLAIN_JSD_LOSS,
    HATEXPLAIN_SPARSITY_LOSS,
    HATEXPLAIN_UNNORMALISED_BACKBONE,
)
from genspp2025.configurations.keys import (
    NAMESPACE,
)


@register_class(
    name="guider",
    tags={"hatexplain"},
    namespace=NAMESPACE,
    component="pyhighlights.components.models.spp.grat.GuidanceGuider",
)
class HateXplainGuiderConfig(GuidanceGuiderConfig):
    """The original guidance module, sized as the baselines' encoder."""

    #: A placeholder, as in :class:`HateXplainBackboneConfig`: the task
    #: replaces the table with GloVe.
    vocab_size: int = Param(3, ge=3)
    embedding_dim: int = Param(25, ge=1)
    hidden_size: int = Param(16, ge=1)
    freeze_embeddings: bool | None = Param(True)


@register_class(
    name="model",
    tags={"fr", "hatexplain"},
    namespace=NAMESPACE,
    component="pyhighlights.components.models.spp.fr.FR",
)
class HateXplainFRConfig(GRUFRConfig):
    selector_backbones: RegistrationKey[SPPBackbone] = Param(HATEXPLAIN_BACKBONE)
    losses: List[RegistrationKey[Loss]] = Param(
        [CLASSIFICATION_LOSS, HATEXPLAIN_SPARSITY_LOSS]
    )


@register_class(
    name="model",
    tags={"hatexplain", "mgr"},
    namespace=NAMESPACE,
    component="pyhighlights.components.models.spp.mgr.MGR",
)
class HateXplainMGRConfig(GRUMGRConfig):
    selector_backbones: List[RegistrationKey[SPPBackbone]] = Param(
        [HATEXPLAIN_BACKBONE, HATEXPLAIN_BACKBONE, HATEXPLAIN_BACKBONE]
    )
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(
        HATEXPLAIN_UNNORMALISED_BACKBONE
    )
    losses: List[RegistrationKey[Loss]] = Param(
        [CLASSIFICATION_LOSS, HATEXPLAIN_SPARSITY_LOSS]
    )


@register_class(
    name="model",
    tags={"hatexplain", "mcd"},
    namespace=NAMESPACE,
    component="pyhighlights.components.models.spp.mcd.MCD",
)
class HateXplainMCDConfig(GRUMCDConfig):
    selector_backbones: RegistrationKey[SPPBackbone] = Param(HATEXPLAIN_BACKBONE)
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(
        HATEXPLAIN_UNNORMALISED_BACKBONE
    )
    #: Sparsity alone, since HateXplain has no contiguity term. The
    #: classification losses stay in the predictor phase, as the original
    #: `train_decouple_causal2` scores them, so the generator never trains on
    #: the predictor's cross entropy.
    shared_losses: List[RegistrationKey[Loss]] = Param([HATEXPLAIN_SPARSITY_LOSS])
    #: The setup's validation total leaves out the full input's cross entropy.
    monitored_losses: List[str] | None = Param(
        ["classification", "sparsity", "discrepancy"]
    )


@register_class(
    name="model",
    tags={"grat", "hatexplain"},
    namespace=NAMESPACE,
    component="pyhighlights.components.models.spp.grat.GRAT",
)
class HateXplainGRATConfig(GRUGRATConfig):
    selector_backbones: RegistrationKey[SPPBackbone] = Param(HATEXPLAIN_BACKBONE)
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(HATEXPLAIN_BACKBONE)
    guider: RegistrationKey = Param(HATEXPLAIN_GUIDER)
    losses: List[RegistrationKey[Loss]] = Param(
        [
            CLASSIFICATION_LOSS,
            HATEXPLAIN_SPARSITY_LOSS,
            HATEXPLAIN_GUIDE_LOSS,
            HATEXPLAIN_JSD_LOSS,
        ]
    )
    guide_decay: float = Param(1e-5, ge=0.0)
    pretrain_epochs: int = Param(10, ge=0)


@register_class(
    name="model",
    tags={"genspp", "hatexplain"},
    namespace=NAMESPACE,
    component="pyhighlights.components.models.spp.genspp.GenSPP",
)
class HateXplainGenSPPConfig(GRUGenSPPConfig):
    selector_backbones: RegistrationKey[SPPBackbone] = Param(HATEXPLAIN_GENSPP_BACKBONE)
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(HATEXPLAIN_GENSPP_BACKBONE)
