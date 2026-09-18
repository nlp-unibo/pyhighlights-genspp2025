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
#: **Still the v1 record, and v2 is not deposited yet.** The record published
#: at `10.5281/zenodo.22711449 <https://doi.org/10.5281/zenodo.22711449>`_
#: holds ``pyhighlights-genspp-toy-v1.zip``, which is the release's own pickle
#: and needs :class:`~genspp2025.components.corpora.ReleasedToyLoader` to read.
#: What this configuration names is the converted artifact,
#: ``pyhighlights-genspp-toy-v2.zip``, which
#: ``pyhighlights/tools/build_datasets.py --skip-r2a`` builds byte for byte and
#: :data:`SHA256` already pins. Zenodo cannot add a file to a published record,
#: so depositing it means a new version of concept
#: `10.5281/zenodo.22711448 <https://doi.org/10.5281/zenodo.22711448>`_ -- and
#: a new version gets its own record id. Set it here once that is published;
#: until then a run of this corpus fetches a 404.
RECORD = "22711449"
#: The artifact this configuration reads, as the build names it.
ARCHIVE = "pyhighlights-genspp-toy-v2.zip"
URL = f"https://zenodo.org/api/records/{RECORD}/files/{ARCHIVE}/content"
#: Digest of the built artifact, which is deterministic: two builds of the same
#: released pickle produce the same bytes, so this pins the deposit in advance.
SHA256 = "d527ff51dd143e51d89251895f6b691f8bc91c707011abce7298b374d6b868f9"


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
    converts it on the way in; a copy of the *original* pickle, which stores
    ``structure_indexes`` instead of a highlight vector, is read with
    :class:`genspp2025.components.corpora.ReleasedToyLoader`.

    ``url`` is set rather than left null on purpose. Null, the loader generates
    -- and a corpus of the same shape and different content is precisely what a
    reproduction must not quietly get.
    """

    url: str | None = Param(URL)
    sha256: str | None = Param(SHA256)
    member: str = Param("corpus.pkl")
    archive_name: str | None = Param(ARCHIVE)
    #: The released baselines' scheme, which the artifact does not store: the
    #: first 80% train, a fifth of it held out, the rest test.
    train_ratio: float = Param(0.8, gt=0.0, lt=1.0)
    val_ratio: float = Param(0.2, ge=0.0, lt=1.0)
    split_seed: int = Param(15000)
