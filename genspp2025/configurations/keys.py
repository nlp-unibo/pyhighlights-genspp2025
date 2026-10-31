"""Keys the GenSPP reproduction is addressed by.

Its own namespace, so the paper's HateXplain, filtered to short posts and
folded to two classes, never stands in for the corpus as distributed.
"""

from cinnamon.registry import Namespace

NAMESPACE = Namespace("genspp2025")

#: The paper waits thirty epochs before giving up on a five-hundred-epoch
#: budget, which is the reproduction's policy rather than the library's.
PAPER_EARLY_STOPPING = NAMESPACE.key("callback", "early_stopping")
PAPER_CHECKPOINT = NAMESPACE.key("callback", "checkpoint")
