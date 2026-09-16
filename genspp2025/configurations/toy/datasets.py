"""The released toy corpus, as this reproduction reads it."""

from cinnamon.configuration import Param
from cinnamon.registry import register_class
from pyhighlights.configurations.datasets import LoaderConfig

from genspp2025.configurations.keys import (
    NAMESPACE,
)

#: The published artifact, `10.5281/zenodo.22711449
#: <https://doi.org/10.5281/zenodo.22711449>`_, released under CC-BY-4.0 by
#: both authors of the paper. The version record rather than the concept one,
#: because the digest pins these exact bytes and a concept DOI resolves to
#: whatever is newest.
URL = (
    "https://zenodo.org/api/records/22711449/files/"
    "pyhighlights-genspp-toy-v2.zip/content"
)
#: Digest of the artifact :data:`URL` names.
SHA256 = "TO-BE-FILLED-ON-PUBLICATION"


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
    archive_name: str | None = Param("pyhighlights-genspp-toy-v2.zip")
    #: The released baselines' scheme, which the artifact does not store: the
    #: first 80% train, a fifth of it held out, the rest test.
    train_ratio: float = Param(0.8, gt=0.0, lt=1.0)
    val_ratio: float = Param(0.2, ge=0.0, lt=1.0)
    split_seed: int = Param(15000)
