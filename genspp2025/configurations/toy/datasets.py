"""The released toy corpus, as this reproduction reads it."""

from cinnamon.configuration import Param
from cinnamon.registry import register_class
from pyhighlights.configurations.datasets import LoaderConfig

from genspp2025.configurations.keys import (
    NAMESPACE,
)

#: The Zenodo **version** record the artifact is fetched from, rather than the
#: concept one: the digest below pins these exact bytes, and a concept DOI
#: resolves to whatever is newest.
#:
#: `10.5281/zenodo.23570345 <https://doi.org/10.5281/zenodo.23570345>`_, the
#: third version of concept `10.5281/zenodo.22711448
#: <https://doi.org/10.5281/zenodo.22711448>`_. The first version holds
#: ``pyhighlights-genspp-toy-v1.zip``, the release's own pickle, which needs
#: :class:`~genspp2025.components.corpora.ReleasedToyLoader` to read. The
#: second holds the corpus in the library's columns, and this one adds the
#: released baselines' split as a ``split`` column.
RECORD = "23570345"
#: The artifact this configuration reads, as the build names it.
ARCHIVE = "pyhighlights-genspp-toy-v3.zip"
URL = f"https://zenodo.org/api/records/{RECORD}/files/{ARCHIVE}/content"
#: Digest of the deposited artifact, checked against the published file. The
#: build is deterministic, so ``build_datasets.py --skip-r2a`` reproduces these
#: bytes from the released pickle.
SHA256 = "f44ed02017cfde635b8ba3bc11bcf468e1d467d83634a96d698505574127431f"


@register_class(
    name="dataset",
    tags={"toy"},
    namespace=NAMESPACE,
    component="pyhighlights.components.loaders.ToyLoader",
)
class ToyConfig(LoaderConfig):
    """The released corpus, not a regenerated one.

    A plain :class:`~pyhighlights.components.loaders.ToyLoader` with a ``url``.
    The artifact holds the corpus in the library's own columns, so nothing
    converts it on the way in. It stores each row's split, the released
    baselines' one, so nothing draws it either. A copy of the *original*
    pickle, which stores ``structure_indexes`` instead of a highlight vector,
    is read with :class:`genspp2025.components.corpora.ReleasedToyLoader`.

    ``url`` is set rather than left null on purpose. Null, the loader
    generates, and a corpus of the same shape and different content is
    precisely what a reproduction must not quietly get.
    """

    url: str | None = Param(URL)
    sha256: str | None = Param(SHA256)
    member: str = Param("corpus.pkl")
    archive_name: str | None = Param(ARCHIVE)
