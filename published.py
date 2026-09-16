"""The numbers this reproduction is trying to land on.

Transcribed from Ruggeri and Signorelli, 2025, *Interlocking-free Selective
Rationalization Through Genetic-based Learning*, ACL 2025, Tables 1 and 2 --
<https://aclanthology.org/2025.acl-long.59/>.

Every entry is ``(mean, std)`` over the paper's five seeds, in the paper's own
units: percentages for the three F1-like columns, and ``R`` a percentage of
the document. Nothing here is computed; it is a transcription, and the only
thing it is good for is being compared against.

Columns, as the caption defines them:

``clf_f1``
    Macro F1 over the classification task. Three classes on Toy, two on
    HateXplain after ``offensive`` is folded into ``hatespeech``.
``hl_f1``
    Binary token-level F1 of the highlight against the annotation.
``rate``
    Selection rate: the share of the document the selector kept.
``size``
    Selection size: how many tokens it kept.
"""

from typing import Dict, Tuple

#: What a row holds, in order.
COLUMNS = ("clf_f1", "hl_f1", "rate", "size")

Row = Dict[str, Tuple[float, float]]


def _row(clf_f1, hl_f1, rate, size) -> Row:
    return dict(zip(COLUMNS, (clf_f1, hl_f1, rate, size)))


#: Table 1, the benchmark evaluation. ``**`` in the paper marks Wilcoxon
#: significance at 0.01 against the best baseline, which is not recorded here:
#: it is a claim about a comparison rather than a number a cell holds.
TABLE_1: Dict[str, Dict[str, Row]] = {
    "toy": {
        "fr": _row((99.78, 0.20), (54.07, 4.02), (14.80, 0.24), (2.96, 0.05)),
        "mgr": _row((99.92, 0.04), (50.34, 11.23), (15.05, 0.80), (3.01, 0.16)),
        "mcd": _row((99.90, 0.04), (65.70, 3.76), (15.18, 0.17), (3.04, 0.03)),
        "grat": _row((99.36, 0.82), (50.22, 7.78), (14.81, 0.51), (2.96, 0.10)),
        "genspp": _row((99.00, 0.25), (76.02, 0.64), (11.47, 0.49), (2.29, 0.08)),
    },
    "hatexplain": {
        "fr": _row((72.14, 1.12), (31.15, 2.56), (25.55, 0.72), (3.46, 0.11)),
        "mgr": _row((71.14, 1.16), (29.38, 4.83), (25.30, 1.04), (3.42, 0.06)),
        "mcd": _row((70.37, 1.06), (27.92, 1.66), (25.07, 1.52), (3.50, 0.20)),
        "grat": _row((73.85, 1.05), (36.17, 1.62), (24.68, 0.86), (3.34, 0.08)),
        "genspp": _row((69.71, 0.40), (42.62, 0.73), (6.51, 0.58), (0.75, 0.05)),
    },
}

#: Table 2, the synthetic skew experiment. **Not reproducible here yet**: the
#: skew rows need a selector pre-trained to select the first token and then
#: injected into the initial population, and ``GenSPPTrainer`` builds every
#: founder at random with no way to seed one. ``genspp-g150`` is the exception
#: -- it is ``n_generations=150`` and nothing else, so it can be run today.
TABLE_2: Dict[str, Dict[str, Row]] = {
    "toy": {
        "fr": _row((99.85, 0.11), (58.91, 3.18), (14.57, 0.12), (2.91, 0.02)),
        "mgr": _row((97.75, 4.25), (37.45, 12.33), (14.99, 0.42), (3.00, 0.08)),
        "mcd": _row((99.93, 0.06), (62.94, 2.39), (15.10, 0.66), (3.02, 0.13)),
        "grat": _row((99.85, 0.14), (47.53, 12.77), (14.56, 0.36), (2.91, 0.07)),
        "genspp-g100": _row((98.93, 0.47), (70.52, 0.15), (13.04, 0.12), (2.60, 0.02)),
        "genspp-g150": _row((99.46, 0.36), (74.28, 0.61), (10.11, 0.42), (1.99, 0.04)),
        "genspp-sk-g100": _row(
            (98.74, 0.43), (63.45, 0.36), (8.03, 0.40), (1.58, 0.06)
        ),
    },
    "hatexplain": {
        "fr": _row((71.00, 0.76), (7.22, 2.29), (26.85, 0.92), (3.47, 0.08)),
        "mgr": _row((71.09, 1.00), (14.45, 4.84), (27.75, 1.36), (3.57, 0.11)),
        "mcd": _row((70.93, 0.95), (13.88, 10.13), (25.84, 1.87), (3.46, 0.16)),
        "grat": _row((73.15, 0.35), (34.33, 1.22), (25.43, 0.87), (3.40, 0.09)),
        "genspp-g100": _row((67.02, 0.57), (39.89, 0.73), (7.45, 0.48), (0.96, 0.05)),
        "genspp-g150": _row((69.89, 0.43), (42.81, 0.65), (6.74, 0.67), (0.87, 0.07)),
        "genspp-sk-g100": _row(
            (66.41, 0.35), (35.52, 0.46), (8.17, 0.62), (1.06, 0.07)
        ),
    },
}

#: A seed run's wall clock on the paper's hardware, an NVIDIA 3060Ti with 8 GB,
#: from its appendix. Worth reading before submitting: GenSPP is most of the
#: budget, because it trains one predictor per candidate per generation.
MINUTES_PER_SEED = {
    "toy": {"baseline": 8, "genspp": 36},
    "hatexplain": {"baseline": 4, "genspp": 78},
}

__all__ = ["COLUMNS", "MINUTES_PER_SEED", "TABLE_1", "TABLE_2"]
