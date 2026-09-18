"""The GenSPP 2025 reproduction: its keys, its values, and its corpora."""

import zipfile
from pathlib import Path

import pandas as pd
import pyhighlights
import pytest
import torch as th
from cinnamon.registry import Registry
from pyhighlights.components.loaders import ToyLoader
from pyhighlights.components.models.spp.genspp import GenSPPTrainer
from pyhighlights.components.models.spp.implementations import GRUBackbone
from pyhighlights.components.preprocessors import Preprocessor
from pyhighlights.utility.embeddings import one_hot_table

import genspp2025
from genspp2025.components.corpora import ReleasedToyLoader
from genspp2025.configurations.hatexplain.keys import (
    HATEXPLAIN_FR_TASK,
    HATEXPLAIN_GENSPP,
    HATEXPLAIN_GENSPP_TASK,
    HATEXPLAIN_GENSPP_TRAINER,
    HATEXPLAIN_GRAT,
    HATEXPLAIN_PIPELINE,
    HATEXPLAIN_SPARSITY,
    HATEXPLAIN_SPARSITY_LOSS,
)
from genspp2025.configurations.keys import SEEDS
from genspp2025.configurations.toy import (
    EMBEDDING_DIM,
    VOCABULARY_SIZE,
)
from genspp2025.configurations.toy.datasets import ARCHIVE, RECORD, SHA256
from genspp2025.configurations.toy.keys import (
    TOY,
    TOY_BENCHMARK,
    TOY_FR_TASK,
    TOY_GENSPP,
    TOY_GENSPP_SMOKE_TRAINER,
    TOY_GENSPP_TASK,
    TOY_GENSPP_TRAINER,
    TOY_MGR,
)
from tests.corpora import hatexplain


def build_registry():
    """The reproduction is what runs; the library is what it references."""
    return Registry.build(
        directory=Path(genspp2025.__file__).parent,
        external_directories=[Path(pyhighlights.__file__).parent],
    )


def released_pickle(directory: Path, rows: int = 10) -> Path:
    """The release's schema: marked positions, and no `tokens` column."""
    path = directory / "toy_dataset.pkl"
    pd.DataFrame(
        {
            "text": ["abcdefghij"[:4] + f"{index:06d}" for index in range(rows)],
            "label": [index % 3 for index in range(rows)],
            "structure_indexes": [[0, 1, 2] for _ in range(rows)],
        }
    ).to_pickle(path)
    return path


def test_both_corpora_register():
    valid, invalid = build_registry()

    paper = [key for key in valid if key.namespace == "genspp2025"]
    # Naming the keys: an invalid one is an experiment missing from the
    # benchmark rather than an error, so the failure has to say which.
    assert not invalid, sorted(str(key) for key in invalid)
    # Both halves, not whichever the filesystem yielded first: registering the
    # second script needs the re-entrant registration context of cinnamon 2.0.2.
    assert len([key for key in paper if "toy" in key.tags]) >= 16
    assert len([key for key in paper if "hatexplain" in key.tags]) >= 22


def test_the_released_hyperparameters_are_what_is_registered():
    build_registry()

    task = Registry.from_key(TOY_FR_TASK)
    assert task.seeds == SEEDS == [2023, 15451, 1337, 2001, 2080]
    assert task.batch_size == 64
    # The paper's patience is the reproduction's own registration, not the
    # library's default of five.
    (stopping,) = [
        Registry.retrieve_configuration(registration_key=key)
        for key in task.callbacks
        if "early_stopping" in key.tags
    ]
    assert stopping.patience == 30
    assert task.trainer_args["max_epochs"] == 500

    # Three generators, and the baselines' hidden sizes: 8 for toy, 16 for
    # HateXplain.
    mgr = Registry.from_key(TOY_MGR)
    assert len(mgr.selectors) == 3
    assert mgr.selector_backbone.encoder.hidden_size == 8

    # On toy the embedding width is a one-hot width rather than a projection
    # size, and the release gives two: 25 on the baselines, 26 on the genetic
    # half, for the same twenty-four characters. Both carry dead columns, so
    # both halves here read the alphabet's own width. HateXplain's 25 is a
    # real GloVe dimension and is not touched.
    assert mgr.selector_backbone.embedding.embedding_dim == 24

    # GenSPP's own encoder is still the genetic half's in the way that counts:
    # one direction against the baselines' two.
    toy_genspp = Registry.from_key(TOY_GENSPP)
    assert toy_genspp.selector_backbone.embedding.embedding_dim == 24
    assert toy_genspp.selector_backbone.encoder.bidirectional is False
    hatexplain_genspp = Registry.from_key(HATEXPLAIN_GENSPP)
    assert hatexplain_genspp.selector_backbone.embedding.embedding_dim == 25
    assert hatexplain_genspp.selector_backbone.encoder.bidirectional is False

    grat = Registry.from_key(HATEXPLAIN_GRAT)
    assert grat.selector_backbone.encoder.hidden_size == 16
    assert grat.guide_decay == pytest.approx(1e-5)
    assert grat.pretrain_epochs == 10
    coefficients = {loss.name: loss.coefficient for loss in grat.losses}
    assert coefficients["guide"] == pytest.approx(2.5)
    assert coefficients["jsd"] == pytest.approx(1.5)
    assert "contiguity" not in coefficients

    # Sparsity is the one target that differs between the corpora.
    assert Registry.from_key(HATEXPLAIN_SPARSITY).threshold == pytest.approx(0.22)
    assert Registry.from_key(HATEXPLAIN_SPARSITY_LOSS).loss.threshold == pytest.approx(
        0.22
    )

    # So is the expected cross entropy the search calls a candidate good at.
    assert Registry.from_key(TOY_GENSPP_TRAINER).task_loss_limit == pytest.approx(0.1)
    assert Registry.from_key(
        HATEXPLAIN_GENSPP_TRAINER
    ).task_loss_limit == pytest.approx(0.6)

    benchmark = Registry.from_key(TOY_BENCHMARK)
    assert len(benchmark.tasks) == 5


def test_the_two_halves_of_hatexplain_build_their_vocabularies_differently(tmp_path):
    """The release does not embed HateXplain the same way twice.

    ``baselines/configurations/model.py`` sets ``use_pretrained_only=True``, and
    under that flag ``GloVeEmbedderCollator.fit`` discards the dataframe it is
    handed and takes all of ``twitter.27B`` as its vocabulary -- so no
    evaluation token the file covers is ever unknown.

    The genetic half does the opposite: ``Dataset.__build_tokenizer`` reads the
    training split only, and ``__embed_texts`` resolves an unknown id through a
    detokenizer built from it, so it embeds as zeros.

    Measured on these splits, the difference reaches 5.4% of validation tokens
    and 5.6% of test tokens.
    """
    build_registry()
    vectors = tmp_path / "vectors.txt"
    vectors.write_text(f"the {' '.join(['0.1'] * 25)}\n")

    baseline = Registry.from_key(
        HATEXPLAIN_FR_TASK, save_path=str(tmp_path), embeddings=str(vectors)
    )
    assert baseline.vocabulary_from == "vectors"

    genetic = Registry.from_key(
        HATEXPLAIN_GENSPP_TASK, save_path=str(tmp_path), embeddings=str(vectors)
    )
    assert genetic.vocabulary_from == "corpus"

    # Both refuse to be built without the file. The configuration used to
    # declare a vocabulary of two, so forgetting it trained on {'the': 1}.
    for key in (HATEXPLAIN_FR_TASK, HATEXPLAIN_GENSPP_TASK):
        with pytest.raises(ValueError, match="given none"):
            Registry.from_key(key, save_path=str(tmp_path))


def test_the_toy_corpus_reaches_a_model_as_one_hot(tmp_path):
    """The release does not embed this corpus, it one-hots it.

    Both halves build a one-hot matrix and hand it over -- the baselines
    through ``OneHotEmbedderCollator``, the genetic half through
    ``OneHotEmbedder``. This reproduction froze a *random* table instead,
    which is a different corpus to learn from: its rows had norm 5.16 and
    reached a cosine of 0.58 with one another, where one-hot rows are
    orthonormal and the padding row is zero.
    """
    build_registry()
    splits = ReleasedToyLoader(url=str(released_pickle(tmp_path))).load()

    for key in (TOY_FR_TASK, TOY_GENSPP_TASK):
        task = Registry.from_key(key, save_path=str(tmp_path))
        # One column per character. The release declares 25 on one half and 26
        # on the other, for the same twenty-four characters.
        assert task.one_hot_embeddings == 24
        task.tokenizer(splits)
        matrix = task._embedding_matrix

        assert matrix.shape == (25, 24)
        # Row zero is the unknown and padding id, and contributes nothing.
        assert not matrix[0].any()
        # Every other row is a distinct basis vector.
        assert th.equal(matrix[1:].sum(dim=1), th.ones(24))
        assert th.equal(matrix[1:] @ matrix[1:].T, th.eye(24))
        # Every column is reachable, where the release's 25 and 26 each
        # leave one and two that nothing can set.
        assert (matrix.sum(dim=0) == 0).sum() == 0

    # And the table the baselines' model is built with is that matrix, rather
    # than the random one its `vocab_size` sized.
    task = Registry.from_key(TOY_FR_TASK, save_path=str(tmp_path))
    task.tokenizer(splits)
    table = task.build_model().selector_backbone.embedding.weight
    assert th.equal(table, task._embedding_matrix)
    assert table.requires_grad is False


def test_a_dead_one_hot_column_changes_nothing_but_the_weight_count():
    """Why the release's 25 and 26 are both fine, and neither is kept.

    The corpus uses twenty-four characters, so a one-hot code needs twenty-four
    columns -- id zero is the zero row and carries none. The baselines declare
    25 and the genetic half 26, and the extra columns are never set, so their
    input weights never receive a gradient.

    Two backbones at the release's widest and at the alphabet's own, with the
    same weights on the live columns and the same ids in, have to encode
    identically. If they do not, the width is a hyperparameter after all and
    unifying the two halves was wrong.
    """
    th.manual_seed(0)
    wide = GRUBackbone(
        vocab_size=VOCABULARY_SIZE,
        embedding_dim=26,
        hidden_size=8,
        freeze_embeddings=True,
    )
    wide.load_embeddings(one_hot_table(VOCABULARY_SIZE, 26))
    narrow = GRUBackbone(
        vocab_size=VOCABULARY_SIZE,
        embedding_dim=EMBEDDING_DIM,
        hidden_size=8,
        freeze_embeddings=True,
    )
    narrow.load_embeddings(one_hot_table(VOCABULARY_SIZE, EMBEDDING_DIM))

    source = dict(wide.encoder.named_parameters())
    with th.no_grad():
        for name, parameter in narrow.encoder.named_parameters():
            # The input projection loses the dead columns; everything else is
            # the same shape.
            parameter.copy_(
                source[name][:, :EMBEDDING_DIM]
                if name.startswith("weight_ih")
                else source[name]
            )
        narrow.layer_norm.load_state_dict(wide.layer_norm.state_dict())

    features = th.randint(0, VOCABULARY_SIZE, (4, 20))
    mask = th.ones(4, 20)
    assert th.equal(wide.encode(features, mask), narrow.encode(features, mask))

    # What the release's widths cost: columns nothing can ever set.
    assert (one_hot_table(VOCABULARY_SIZE, 26).sum(dim=0) == 0).sum() == 2
    assert (one_hot_table(VOCABULARY_SIZE, 25).sum(dim=0) == 0).sum() == 1
    assert (one_hot_table(VOCABULARY_SIZE, EMBEDDING_DIM).sum(dim=0) == 0).sum() == 0


def test_the_toy_corpus_is_read_as_characters(tmp_path):
    loader = ReleasedToyLoader(url=str(released_pickle(tmp_path)))
    splits = loader.load()

    assert list(splits) == ["train", "val", "test"]
    row = splits["train"].iloc[0]
    # Tokens are characters, so the vocabulary is the alphabet.
    assert row["tokens"] == list(row["text"])
    assert sum(row["highlights"]) == 3
    assert row["highlights"][:3] == [1, 1, 1]

    # 80% train, a fifth of it held out, the rest test.
    assert len(splits["test"]) == 2
    assert len(splits["train"]) + len(splits["val"]) == 8

    # The validation draw is seeded, so two loads agree.
    again = ReleasedToyLoader(url=str(released_pickle(tmp_path))).load()
    assert splits["val"]["text"].tolist() == again["val"]["text"].tolist()


def test_the_proxy_reads_a_published_archive_of_the_old_schema(tmp_path):
    """Anyone still holding the original artifact can read it.

    The published record now carries the corpus already converted, so the
    registered key does not go through this. It is kept for a copy of the
    release -- from the reference implementation, or made before the record was
    converted -- which would otherwise have nothing to read it with.
    """
    archive = tmp_path / "pyhighlights-genspp-toy-v1.zip"
    with zipfile.ZipFile(archive, "w") as target:
        target.write(released_pickle(tmp_path), "toy_dataset.pkl")
        target.writestr("README.md", "# artifact")

    splits = ReleasedToyLoader(
        url=str(archive),
        sha256=None,
        member="toy_dataset.pkl",
        directory=tmp_path / "cache",
    ).load()

    assert list(splits) == ["train", "val", "test"]
    assert sum(len(frame) for frame in splits.values()) == 10


def test_the_registered_toy_key_names_the_published_artifact():
    """The documented key has to build without a private override.

    Registered with `url=None` it did not, and worse than not building: a
    `ToyLoader` without a `url` *generates*, so the reproduction would have
    trained on a corpus of the right shape and the wrong content.
    """
    build_registry()
    loader = Registry.from_key(TOY, expected_type=ToyLoader)

    assert loader.url.endswith(f"{ARCHIVE}/content")
    assert RECORD in loader.url
    assert loader.member == "corpus.pkl"
    # The digest of what the build produces, pinned before the deposit rather
    # than after it: two builds of the same released pickle are byte for byte
    # the same, so publishing cannot change this number -- only `RECORD` moves.
    assert loader.sha256 == SHA256
    # The released baselines' split scheme, which the artifact does not store.
    assert (loader.train_ratio, loader.val_ratio, loader.split_seed) == (
        0.8,
        0.2,
        15000,
    )


def test_the_converted_artifact_needs_no_proxy(tmp_path):
    """What the record holds now: the library's columns, read by ToyLoader.

    The proxy and a plain loader have to agree, or converting the artifact
    changed the corpus rather than its serialisation.
    """
    released = released_pickle(tmp_path)
    converted = tmp_path / "corpus.pkl"
    frame = pd.read_pickle(released)
    pd.DataFrame(
        {
            "sample_id": range(len(frame)),
            "text": frame["text"],
            "tokens": frame["text"].map(list),
            "label": frame["label"].astype(int),
            "highlights": [
                [1 if position in set(marked) else 0 for position in range(len(text))]
                for marked, text in zip(frame["structure_indexes"], frame["text"])
            ],
        }
    ).to_pickle(converted)

    through_proxy = ReleasedToyLoader(url=str(released)).load()
    direct = ToyLoader(url=str(converted)).load()

    for name, part in through_proxy.items():
        assert direct[name]["text"].tolist() == part["text"].tolist()
        assert direct[name]["highlights"].tolist() == part["highlights"].tolist()
        assert direct[name]["label"].tolist() == part["label"].tolist()


def test_the_hatexplain_pipeline_folds_classes_before_it_counts_votes(tmp_path):
    build_registry()
    from pyhighlights.components.loaders import HateXplainLoader

    splits = HateXplainLoader(**hatexplain(tmp_path / "corpus")).load()
    pipeline = Registry.from_key(HATEXPLAIN_PIPELINE, expected_type=Preprocessor)
    processed = pipeline.process(splits)

    labels = dict(zip(processed["train"]["text"], processed["train"]["label"]))
    # "who cares" is annotated hatespeech, offensive and normal. Over three
    # classes that is a tie and the row is dropped; folding offensive into
    # hatespeech first makes it a two-to-one majority, which is the paper's.
    assert labels["who cares"] == 0
    assert set(processed["train"]["label"]) <= {0, 1}
    assert processed["val"]["label"].tolist() == [1]


def test_a_smoke_search_is_small_where_trainer_arguments_cannot_reach():
    """`--smoke` bounds a baseline through Lightning. A search reads none of it.

    It builds a trainer per candidate, so the paper's fifty candidates over a
    hundred generations ran in full under a flag that promises minutes. The
    smoke key is the same search, small enough to finish.
    """
    build_registry()
    paper = Registry.from_key(TOY_GENSPP_TRAINER, expected_type=GenSPPTrainer)
    smoke = Registry.from_key(TOY_GENSPP_SMOKE_TRAINER, expected_type=GenSPPTrainer)

    assert (paper.population_size, paper.n_generations) == (50, 100)
    assert (smoke.population_size, smoke.n_generations) == (2, 1)
    assert smoke.predictor_epochs == 1
    # Everything the search is otherwise, it still is.
    assert smoke.model == paper.model
    assert smoke.task_loss_limit == paper.task_loss_limit
    assert smoke.mutation_std == paper.mutation_std


def test_the_search_scores_candidates_across_the_workers_the_job_allocates():
    """One worker leaves seven of the job's eight cores to one candidate.

    `cluster/run.sbatch` asks for eight, and the released implementation scores
    on a pool of that size.
    """
    build_registry()
    for key in (TOY_GENSPP_TRAINER, HATEXPLAIN_GENSPP_TRAINER):
        search = Registry.from_key(key, expected_type=GenSPPTrainer)
        assert [str(device) for device in search.devices] == ["cpu"] * 8
