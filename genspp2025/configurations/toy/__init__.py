"""The synthetic corpus: three hidden patterns over a twenty-character string.

Tokens are characters, and the corpus turns out to use twenty-four of them,
with no ``i`` and no ``x``. With separate unknown and padding ids that is a
vocabulary of twenty-six ids, and a one-hot code over the characters needs
**twenty-four** columns: ids zero and one are zero rows and carry none.

The vocabulary and embedding widths are defined in
:mod:`genspp2025.configurations.toy.constants`.
"""
