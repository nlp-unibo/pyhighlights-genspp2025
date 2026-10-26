"""Keys the HateXplain half of the reproduction is addressed by.

Split from the shared ones so neither corpus imports a module carrying the
other's.
"""

from genspp2025.configurations.keys import NAMESPACE

HATEXPLAIN_LENGTH_FILTER = NAMESPACE.key("preprocessor", "length", "hatexplain")
HATEXPLAIN_LABEL_MAPPER = NAMESPACE.key("preprocessor", "labels", "hatexplain")
HATEXPLAIN_AGGREGATOR = NAMESPACE.key("preprocessor", "aggregator", "hatexplain")
HATEXPLAIN_PIPELINE = NAMESPACE.key("preprocessor", "pipeline", "hatexplain")
HATEXPLAIN_SPARSITY = NAMESPACE.key("criterion", "sparsity", "hatexplain")
HATEXPLAIN_SPARSITY_LOSS = NAMESPACE.key("loss", "sparsity", "hatexplain")
HATEXPLAIN_GUIDE_LOSS = NAMESPACE.key("loss", "guide", "hatexplain")
HATEXPLAIN_JSD_LOSS = NAMESPACE.key("loss", "jsd", "hatexplain")
HATEXPLAIN_BACKBONE = NAMESPACE.key("backbone", "hatexplain")
HATEXPLAIN_GENSPP_BACKBONE = NAMESPACE.key("backbone", "hatexplain", "genspp")
HATEXPLAIN_UNNORMALISED_BACKBONE = NAMESPACE.key(
    "backbone", "hatexplain", "unnormalised"
)
HATEXPLAIN_GUIDER = NAMESPACE.key("guider", "hatexplain")
HATEXPLAIN_FR = NAMESPACE.key("model", "fr", "hatexplain")
HATEXPLAIN_MGR = NAMESPACE.key("model", "mgr", "hatexplain")
HATEXPLAIN_MCD = NAMESPACE.key("model", "mcd", "hatexplain")
HATEXPLAIN_GRAT = NAMESPACE.key("model", "grat", "hatexplain")
HATEXPLAIN_GENSPP = NAMESPACE.key("model", "genspp", "hatexplain")
HATEXPLAIN_GENSPP_TRAINER = NAMESPACE.key("trainer", "genspp", "hatexplain")
HATEXPLAIN_GENSPP_SMOKE_TRAINER = NAMESPACE.key(
    "trainer", "genspp", "hatexplain", "smoke"
)
HATEXPLAIN_GENSPP_G150_TRAINER = NAMESPACE.key(
    "trainer", "genspp", "hatexplain", "g150"
)
HATEXPLAIN_FR_TASK = NAMESPACE.key("task", "fr", "hatexplain")
HATEXPLAIN_MGR_TASK = NAMESPACE.key("task", "mgr", "hatexplain")
HATEXPLAIN_MCD_TASK = NAMESPACE.key("task", "mcd", "hatexplain")
HATEXPLAIN_GRAT_TASK = NAMESPACE.key("task", "grat", "hatexplain")
HATEXPLAIN_GENSPP_TASK = NAMESPACE.key("task", "genspp", "hatexplain")
HATEXPLAIN_BENCHMARK = NAMESPACE.key("benchmark", "hatexplain")
