"""Reading the corpus the paper released, in the schema it released it in.

The released ``toy_dataset.pkl`` stores ``structure_indexes`` -- the positions
a highlight marks -- where pyhighlights stores a ``highlights`` vector, and it
stores no ``tokens`` because its tokens are the characters of its ``text``.
That is a difference in serialisation rather than in what the corpus is, so it
is a conversion rather than a loader of its own:
:class:`~pyhighlights.components.loaders.ToyLoader` does the fetching, the
digest check, the splitting and the rest, and this fills in the two columns
the file predates.

**Kept although the published artifact no longer needs it.** The Zenodo record
holds the corpus already converted, so the reproduction's own configuration
reads it with a plain ``ToyLoader``. This stays for anyone holding the
*original* pickle -- from the reference implementation, or from a copy made
before the record was converted -- who would otherwise have nothing to read it
with::

    from genspp2025.components.corpora import ReleasedToyLoader

    splits = ReleasedToyLoader(url="toy_dataset.pkl").load()

It is a standalone component: nothing here is registered, and it composes with
whatever ``ToyLoader`` accepts.
"""

from __future__ import annotations

import pandas as pd
from pyhighlights.components.loaders import ToyLoader

__all__ = ["ReleasedToyLoader"]


class ReleasedToyLoader(ToyLoader):
    """``ToyLoader``, reading the release's schema instead of the library's."""

    def parse(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        # A list of marked positions becomes a vector as long as the text. The
        # release guarantees one pattern per row, so every index is in range;
        # an index that is not would be a corpus this cannot label, and
        # `ToyLoader.parse` is what then says which column it could not build.
        frame["highlights"] = [
            [1 if position in set(marked) else 0 for position in range(len(text))]
            for marked, text in zip(frame["structure_indexes"], frame["text"])
        ]
        return super().parse(frame)
