"""The embedding width every HateXplain configuration shares."""

#: GloVe ``twitter.27B`` at 25 dimensions. The task's replacement table is
#: refused when its width disagrees with a backbone's or a guider's.
EMBEDDING_DIM = 25
