"""Keys the GenSPP reproduction is addressed by.

Its own namespace, so the paper's HateXplain -- filtered to short posts, folded
to two classes -- never stands in for the corpus as distributed.
"""

from cinnamon.registry import RegistrationKey

NAMESPACE = "genspp2025"

#: The five seeds every reported number averages over.
SEEDS = [2023, 15451, 1337, 2001, 2080]

#: Candidates a genetic search scores at once, one per worker, as the release's
#: own CPU thread pool does and as ``cluster/run.sbatch`` allocates
#: (``--cpus-per-task=8``). It buys less than eight times the speed: a
#: candidate is small, so most of its cost is building a Lightning trainer and
#: stepping it from Python. Measured on eight candidates of the toy search --
#: one worker 14.1 s, eight workers 10.0 s, and 7.4 s with torch held to one
#: thread each, which is the machine, not the setting.
WORKERS = ["cpu"] * 8


def key(name: str, *tags: str) -> RegistrationKey:
    return RegistrationKey(name=name, tags=set(tags), namespace=NAMESPACE)


#: The paper waits thirty epochs before giving up on a five-hundred-epoch
#: budget, which is the reproduction's policy rather than the library's.
PAPER_EARLY_STOPPING = key("callback", "early_stopping")
PAPER_CHECKPOINT = key("callback", "checkpoint")
