"""Keys the Toy half of the reproduction is addressed by.

Split from the shared ones so neither corpus imports a module carrying the
other's.
"""

from genspp2025.configurations.keys import NAMESPACE

TOY = NAMESPACE.key("dataset", "toy")
TOY_BACKBONE = NAMESPACE.key("backbone", "toy")
TOY_GENSPP_BACKBONE = NAMESPACE.key("backbone", "toy", "genspp")
TOY_UNNORMALISED_BACKBONE = NAMESPACE.key("backbone", "toy", "unnormalised")
TOY_PREDICTOR = NAMESPACE.key("predictor", "toy")
TOY_GUIDER = NAMESPACE.key("guider", "toy")
TOY_FR = NAMESPACE.key("model", "fr", "toy")
TOY_MGR = NAMESPACE.key("model", "mgr", "toy")
TOY_MCD = NAMESPACE.key("model", "mcd", "toy")
TOY_GRAT = NAMESPACE.key("model", "grat", "toy")
TOY_GENSPP = NAMESPACE.key("model", "genspp", "toy")
TOY_GENSPP_TRAINER = NAMESPACE.key("trainer", "genspp", "toy")
TOY_GENSPP_RELEASED_THRESHOLD_TRAINER = NAMESPACE.key(
    "trainer", "genspp", "toy", "released-threshold"
)
TOY_GENSPP_SMOKE_TRAINER = NAMESPACE.key("trainer", "genspp", "toy", "smoke")
TOY_GENSPP_G150_TRAINER = NAMESPACE.key("trainer", "genspp", "toy", "g150")
TOY_FR_TASK = NAMESPACE.key("task", "fr", "toy")
TOY_MGR_TASK = NAMESPACE.key("task", "mgr", "toy")
TOY_MCD_TASK = NAMESPACE.key("task", "mcd", "toy")
TOY_GRAT_TASK = NAMESPACE.key("task", "grat", "toy")
TOY_GENSPP_TASK = NAMESPACE.key("task", "genspp", "toy")
TOY_FR_SMOKE_TASK = NAMESPACE.key("task", "fr", "toy", "smoke")
TOY_MGR_SMOKE_TASK = NAMESPACE.key("task", "mgr", "toy", "smoke")
TOY_MCD_SMOKE_TASK = NAMESPACE.key("task", "mcd", "toy", "smoke")
TOY_GRAT_SMOKE_TASK = NAMESPACE.key("task", "grat", "toy", "smoke")
TOY_GENSPP_SMOKE_TASK = NAMESPACE.key("task", "genspp", "toy", "smoke")
TOY_BENCHMARK = NAMESPACE.key("benchmark", "toy")
