"""Keys the GenSPP reproduction is addressed by.

Its own namespace, so the paper's HateXplain, filtered to short posts and
folded to two classes, never stands in for the corpus as distributed.
"""

from cinnamon.registry import Namespace

NAMESPACE = Namespace("genspp2025")

#: The five seeds every baseline number averages over, which are the ones the
#: release's baselines use.
SEEDS = [2023, 15451, 1337, 2001, 2080]

#: The five seeds every GenSPP number averages over. The release's ``train.py``
#: seeds run ``i`` with ``i + 1``, on both corpora.
GENSPP_SEEDS = [1, 2, 3, 4, 5]

#: Candidates a genetic search scores at once, one per worker, as the release's
#: own pool does and as ``cluster/run.sbatch`` allocates
#: (``--cpus-per-task=8``). Every entry is a CPU device, which is what lets
#: pyhighlights score them in spawned worker processes: the search uses
#: processes only when every device is a CPU, so one CUDA device here would put
#: the search back on threads. It buys less than eight times the speed, since a
#: candidate is small and most of its cost is the training loop stepping from
#: Python. The library measured sixteen candidates of the toy search on a
#: 24-core machine at 1442 ms a candidate sequentially, 832 ms on eight
#: threads and 293 ms on eight processes.
WORKERS = ["cpu"] * 8


#: The paper waits thirty epochs before giving up on a five-hundred-epoch
#: budget, which is the reproduction's policy rather than the library's.
PAPER_EARLY_STOPPING = NAMESPACE.key("callback", "early_stopping")
PAPER_CHECKPOINT = NAMESPACE.key("callback", "checkpoint")
