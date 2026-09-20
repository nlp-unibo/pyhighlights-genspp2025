"""Ruggeri and Signorelli, 2025, *Interlocking-free Selective Rationalization
Through Genetic-based Learning*, ACL 2025.

Paper: <https://aclanthology.org/2025.acl-long.59/>.
Reference implementation: <https://github.com/nlp-unibo/gen-spp>.

Two corpora, a synthetic one and HateXplain, against FR, MGR, MCD, G-RAT
and GenSPP. Every value here is the released implementation's; where the paper
and the release disagree, or where ``pyhighlights`` cannot reproduce something
exactly, the configuration says so at the point it matters.

Nothing registers on import. Build the registry over this package, with the
library beside it::

    from pathlib import Path
    from cinnamon.registry import Registry
    import genspp2025, pyhighlights

    Registry.build(
        directory=Path(genspp2025.__file__).parent,
        external_directories=[Path(pyhighlights.__file__).parent],
    )
"""
