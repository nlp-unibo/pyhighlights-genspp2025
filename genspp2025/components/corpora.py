"""Reading the corpus the paper released, in the schema it released it in.

The released ``toy_dataset.pkl`` stores ``structure_indexes``, the positions
a highlight marks, where pyhighlights stores a ``highlights`` vector, and it
stores no ``tokens`` because its tokens are the characters of its ``text``.
The Zenodo record holds the corpus already converted, so the reproduction's
own configuration reads it with a plain ``ToyLoader``. This stays for anyone
holding the *original* pickle, from the reference implementation or from a copy
made before the record was converted, who would otherwise have nothing to read
it with::

    from genspp2025.components.corpora import ReleasedToyLoader

    splits = ReleasedToyLoader(url="toy_dataset.pkl").load()

"""

from __future__ import annotations

import pandas as pd
from pyhighlights.components.loaders import ToyLoader

__all__ = ["ReleasedToyLoader"]


class ReleasedToyLoader(ToyLoader):
    """``ToyLoader``, reading the release's schema instead of the library's."""

    def parse(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        frame["highlights"] = [
            [1 if position in set(marked) else 0 for position in range(len(text))]
            for marked, text in zip(frame["structure_indexes"], frame["text"])
        ]
        return super().parse(frame)
