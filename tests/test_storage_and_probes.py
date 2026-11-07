"""
Saving/loading results, re-analysis from saved covariances, and the probe-set builder.
"""

import sys
from pathlib import Path

import numpy as np

from nds import storage
from nds.moments import NeuronMoments
from nds.probe_text import build_probe_set
from nds.probe_text import load_probe_set
from nds.probe_text import save_probe_set

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_saved_moments_rebuild_the_same_covariance(tmp_path):
    from reanalyse import moments_from_saved

    activations = np.random.default_rng(0).standard_normal((300, 6))
    moments = NeuronMoments(6)
    moments.add_batch(activations)

    storage.save_layer_moments(tmp_path, 0, moments)
    saved = storage.load_layer_moments(tmp_path, 0)

    rebuilt = moments_from_saved(saved).merged()

    # float32 storage: agree to ~1e-6
    assert np.allclose(rebuilt.covariance(), moments.covariance(), atol=1e-5)
    assert rebuilt.count == 300


class FakeTokenizer:
    """
    Splits on spaces and maps each word to its length: enough to test the probe-set builder.
    """

    def __call__(self, text):
        return {"input_ids": [len(word) for word in text.split()]}


def test_probe_set_skips_short_documents_and_round_trips(tmp_path):
    documents = [
        ("too short", "A"),
        ("one two three four five six", "B"),
        ("a bb ccc dddd eeeee ffffff ggggggg", "C"),
    ]

    tokens, rows = build_probe_set(FakeTokenizer(), iter(documents), number_of_sequences=2, sequence_length=5)

    assert tokens.shape == (2, 5)
    assert [row["pile_set_name"] for row in rows] == ["B", "C"]
    assert list(tokens[1]) == [1, 2, 3, 4, 5]

    save_probe_set(tokens, rows, tmp_path / "t.npy", tmp_path / "t.json", "test")
    loaded, manifest = load_probe_set(tmp_path / "t.npy", tmp_path / "t.json")

    assert np.array_equal(loaded, tokens)
    assert manifest["shape"] == [2, 5]


def test_finished_steps_sorted(tmp_path):
    for step in [1000, 0, 512]:
        storage.save_summary(storage.step_folder(tmp_path, step), {"step": step})

    assert storage.finished_steps(tmp_path) == [0, 512, 1000]


def test_probe_rows_can_start_with_the_document_separator():
    documents = [
        ("one two three four", "B"),
        ("a bb ccc dddd eeeee", "C"),
    ]

    tokens, rows = build_probe_set(
        FakeTokenizer(),
        iter(documents),
        number_of_sequences=2,
        sequence_length=5,
        document_separator_id=0,
    )

    # 4 document tokens are enough when the separator takes the first place
    assert tokens.shape == (2, 5)
    assert list(tokens[0]) == [0, 3, 3, 5, 4]
    assert list(tokens[1]) == [0, 1, 2, 3, 4]


def test_runs_root_follows_the_environment(monkeypatch, tmp_path):
    monkeypatch.delenv("NDS_RUNS_DIR", raising=False)
    assert str(storage.runs_root()) == "runs"

    monkeypatch.setenv("NDS_RUNS_DIR", str(tmp_path / "shared"))
    assert storage.run_folder_for("week1") == tmp_path / "shared" / "week1"
