"""The miniature HateXplain these tests build.

A trimmed copy of the library's ``tests/corpora.py``, carrying the one fixture
this reproduction uses. Kept rather than imported: the library ships its tests
nowhere, and a fixture is cheaper to copy than to depend on.
"""

import json
from pathlib import Path

#: The fixtures below build stand-in corpora rather than the pinned upstream
#: releases, so a loader reading one has to opt out of the digest its defaults
#: now carry.
UNPINNED = {"sha256": None}


def annotator(label: str, index: int) -> dict:
    return {"label": label, "annotator_id": index, "target": ["None"]}


def hatexplain(directory: Path) -> dict:
    """Posts covering a majority label, a tie, a normal post and a duplicate."""
    directory.mkdir(parents=True, exist_ok=True)
    posts = {
        "p1": {
            "post_id": "p1",
            "annotators": [
                annotator("hatespeech", 1),
                annotator("hatespeech", 2),
                annotator("offensive", 3),
            ],
            "rationales": [[1, 1, 0], [1, 0, 0]],
            "post_tokens": ["burn", "them", "all"],
        },
        "p2": {
            "post_id": "p2",
            "annotators": [
                annotator("hatespeech", 1),
                annotator("offensive", 2),
                annotator("normal", 3),
            ],
            "rationales": [[0, 1]],
            "post_tokens": ["who", "cares"],
        },
        "p3": {
            "post_id": "p3",
            "annotators": [annotator("normal", i) for i in (1, 2, 3)],
            "rationales": [],
            "post_tokens": ["nice", "day"],
        },
        # Same text as p1: id-based splits do not stop text leaking.
        "p4": {
            "post_id": "p4",
            "annotators": [
                annotator("hatespeech", 1),
                annotator("hatespeech", 2),
                annotator("normal", 3),
            ],
            "rationales": [[1, 1, 0], [1, 1, 0], [0, 1, 0]],
            "post_tokens": ["burn", "them", "all"],
        },
    }
    divisions = {"train": ["p1", "p2"], "val": ["p3"], "test": ["p4"]}
    (directory / "posts.json").write_text(json.dumps(posts))
    (directory / "divisions.json").write_text(json.dumps(divisions))
    return {
        "url": (directory / "posts.json").as_uri(),
        "divisions_url": (directory / "divisions.json").as_uri(),
        "directory": directory / "cache",
        # Two downloads, two opt-outs: these are stand-ins rather than the
        # pinned upstream files.
        **UNPINNED,
        "divisions_sha256": None,
    }
