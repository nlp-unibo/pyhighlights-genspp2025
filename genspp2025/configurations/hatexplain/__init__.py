"""HateXplain as the paper prepares it, which is not as it is distributed.

Three choices turn the corpus into the paper's benchmark, and all three change
the numbers:

* posts over thirty tokens are **dropped**, which is how the released code
  bounds its compute;
* ``offensive`` is folded into ``hatespeech`` **before** the annotators are
  counted, leaving a two-class task. Fold it afterwards and a post the three
  annotators split three ways gets a different label;
* the surviving votes and rationales are reduced by majority.

Tokens are embedded with GloVe ``twitter.27B`` at 25 dimensions, frozen.

**Every token of every split has a row of its own.** A token GloVe covers
gets its vector, and any other token gets its own ``uniform(-0.05, 0.05)``
row from a fixed seed. That is ``vocabulary_from="splits"`` with
``pretrained_tokens_only=False``, for the four baselines and for GenSPP. The
table is frozen, so a row no training token uses is never trained.

The release collapses tokens instead, and this reproduction documents that
rather than reproducing it. The released baseline collator is built with
``use_pretrained_only=True``, which takes the whole of ``twitter.27B`` as its
vocabulary and maps any other token to ``<PAD>``, a zero row. The released
GenSPP shares one uniform vector over the training tokens GloVe lacks and
embeds every token unseen in training as zeros. Fitting on the training split
alone would leave 5.4% of validation tokens and 5.6% of test tokens unknown.

That file is a 1.4 GB download the paper expects you to fetch yourself, so it
is a path the task is given rather than a URL it fetches::

    Registry.from_key(HATEXPLAIN_FR_TASK, embeddings="glove.twitter.27B.25d.txt")

Leaving it out is refused rather than run: without the file there is no
vocabulary to take.
"""

#: GloVe ``twitter.27B`` at 25 dimensions. The task's replacement table is
#: refused when its width disagrees with a backbone's or a guider's.
EMBEDDING_DIM = 25
