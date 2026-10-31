"""The encoders: one GRU for the baselines, one for GenSPP."""

from cinnamon.configuration import Param
from cinnamon.registry import register_class
from pyhighlights.configurations.backbones import GRUBackboneConfig

from genspp2025.configurations.hatexplain import EMBEDDING_DIM
from genspp2025.configurations.hatexplain.keys import (
    HATEXPLAIN_BACKBONE,
    HATEXPLAIN_GENSPP_BACKBONE,
    HATEXPLAIN_UNNORMALISED_BACKBONE,
)

GRU_BACKBONE_COMPONENT = (
    "pyhighlights.components.models.spp.implementations.GRUBackbone"
)


@register_class(key=HATEXPLAIN_BACKBONE, component=GRU_BACKBONE_COMPONENT)
class HateXplainBackboneConfig(GRUBackboneConfig):
    """The baselines' encoder, embedding from the GloVe table the task reads."""

    #: A placeholder. The task reads GloVe and hands the matrix to the model,
    #: and :meth:`GRUBackbone.load_embeddings` **replaces** the table rather
    #: than copying into it, so the size declared here is never the size used.
    #: ``embedding_dim`` below is not a placeholder: the replacement is refused
    #: if its width disagrees.
    vocab_size: int = Param(2, ge=1)
    embedding_dim: int = Param(EMBEDDING_DIM, ge=1)
    hidden_size: int = Param(16, ge=1)
    freeze_embeddings: bool = Param(True)
    dropout_rate: float = Param(0.0, ge=0.0, lt=1.0)


@register_class(key=HATEXPLAIN_UNNORMALISED_BACKBONE, component=GRU_BACKBONE_COMPONENT)
class HateXplainUnnormalisedBackboneConfig(HateXplainBackboneConfig):
    """The MCD and MGR predictors' encoder, unnormalised as in their originals."""

    layer_norm: bool = Param(False)


@register_class(key=HATEXPLAIN_GENSPP_BACKBONE, component=GRU_BACKBONE_COMPONENT)
class HateXplainGenSPPBackboneConfig(HateXplainBackboneConfig):
    bidirectional: bool = Param(False)
    #: The released GenSPP encoder passes the GRU states on unnormalised.
    layer_norm: bool = Param(False)
