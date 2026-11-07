"""
Token sample (D31), token-graph DSE (D34), TwoNN (D35), the nulls (D36), partition stability (D33)
and event classification (D37), all on synthetic data with known answers.
"""

import numpy as np

from nds import storage
from nds.modules import adjusted_rand_index
from nds.moments import BlockedNeuronMoments
from nds.noise import effective_token_count
from nds.noise import lag_one_correlations
from nds.noise import marchenko_pastur_edges
from nds.noise import wishart_null_correlation
from nds.timeline import classify_events
from nds.timeline import partition_stability
from nds.token_sample import choose_token_positions
from nds.token_sample import gather_token_sample
from nds.token_sample import shuffle_neurons_independently
from nds.token_sample import token_graph_dse
from nds.token_sample import twonn_intrinsic_dimension


def test_token_positions_skip_position_zero_and_are_fixed():
    first = choose_token_positions(50, 32, sample_size=200, skip_first_positions=1, seed=3)
    second = choose_token_positions(50, 32, sample_size=200, skip_first_positions=1, seed=3)

    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert first[1].min() >= 1
    assert first[1].max() <= 31
    assert len(set(zip(first[0].tolist(), first[1].tolist()))) == 200


def test_gather_picks_the_right_tokens():
    values = np.arange(4 * 5 * 3).reshape(4, 5, 3)
    picked = gather_token_sample(values, np.array([0, 3]), np.array([1, 4]))
    assert np.array_equal(picked[0], values[0, 1])
    assert np.array_equal(picked[1], values[3, 4])


def test_twonn_finds_the_dimension_of_a_flat_set():
    generator = np.random.default_rng(0)
    # 1,500 points on a 3-dimensional plane inside 20 dimensions
    coefficients = generator.uniform(size=(1500, 3))
    basis = np.linalg.qr(generator.standard_normal((20, 3)))[0]
    points = coefficients @ basis.T
    dimension = twonn_intrinsic_dimension(points)
    assert 2.5 < dimension < 3.5


def test_token_graph_dse_is_unchanged_by_rotating_the_neurons():
    generator = np.random.default_rng(1)
    sample = generator.standard_normal((300, 40))
    rotation = np.linalg.qr(generator.standard_normal((40, 40)))[0]
    original = token_graph_dse(sample, diffusion_times=(1, 4), sigma="median")
    rotated = token_graph_dse(sample @ rotation, diffusion_times=(1, 4), sigma="median")
    assert abs(original[1] - rotated[1]) < 1e-8
    assert abs(original[4] - rotated[4]) < 1e-8


def test_shuffle_keeps_each_neuron_and_kills_correlation():
    generator = np.random.default_rng(2)
    shared = generator.standard_normal(4000)
    sample = np.stack([shared + 0.1 * generator.standard_normal(4000) for _ in range(5)], axis=1)
    shuffled = shuffle_neurons_independently(sample, generator)
    assert np.allclose(np.sort(shuffled[:, 2]), np.sort(sample[:, 2]))
    correlation = np.corrcoef(shuffled, rowvar=False)
    off_diagonal = correlation[~np.eye(5, dtype=bool)]
    assert np.max(np.abs(off_diagonal)) < 0.06


def test_effective_tokens_and_lag_one_correlation():
    assert effective_token_count(1000, 0.0) == 1000.0
    assert abs(effective_token_count(1000, 0.7) - 1000 * (1 - 0.49) / (1 + 0.49)) < 1e-9

    # AR(1) sequences with rho = 0.6
    generator = np.random.default_rng(3)
    sequences = np.zeros((200, 100, 4))
    for t in range(1, 100):
        sequences[:, t, :] = 0.6 * sequences[:, t - 1, :] + generator.standard_normal((200, 4))
    rho = lag_one_correlations(sequences)
    assert np.all(np.abs(rho - 0.6) < 0.05)


def test_wishart_null_fills_the_marchenko_pastur_interval():
    generator = np.random.default_rng(4)
    correlation = wishart_null_correlation(200, 2000, generator)
    eigenvalues = np.linalg.eigvalsh(correlation)
    lower, upper = marchenko_pastur_edges(200, 2000)
    assert np.allclose(np.diag(correlation), 1.0)
    assert eigenvalues.min() > lower - 0.1
    assert eigenvalues.max() < upper + 0.1


def test_adjusted_rand_index_extremes():
    labels = np.repeat([0, 1, 2], 20)
    renamed = np.repeat([2, 0, 1], 20)
    assert abs(adjusted_rand_index(labels, renamed) - 1.0) < 1e-12
    random_labels = np.random.default_rng(5).integers(0, 3, size=60)
    assert abs(adjusted_rand_index(labels, random_labels)) < 0.15


def test_partition_stability_uses_common_neurons_only():
    labels_a = np.array([0, 0, 1, 1, 2])
    ids_a = np.array([10, 11, 12, 13, 14])
    labels_b = np.array([5, 5, 7, 7])
    ids_b = np.array([11, 10, 13, 12])
    assert abs(partition_stability(labels_a, ids_a, labels_b, ids_b) - 1.0) < 1e-12


def test_events_are_sorted_into_hidden_and_explained():
    steps = np.array([0, 1, 2, 4, 8, 16, 32, 64])
    readout = np.array([1.0, 1.0, 1.0, 1.6, 1.6, 1.6, 2.4, 2.4])
    band_low = readout - 0.05
    band_high = readout + 0.05
    rotated = np.full(8, 1.1)
    events = classify_events(steps, readout, band_low, band_high, rotated, [3, 6], baseline_change_indices=[6])
    kinds = {event["step"]: event["kind"] for event in events}
    assert kinds == {4: "hidden", 32: "explained"}


def test_block_moments_round_trip(tmp_path):
    generator = np.random.default_rng(6)
    blocked = BlockedNeuronMoments(6, number_of_blocks=3)
    for block_index in range(3):
        blocked.add_batch(generator.standard_normal((100, 6)), block_index)
    storage.save_block_moments(tmp_path, 0, blocked)
    loaded = storage.load_block_moments(tmp_path, 0)
    assert np.allclose(loaded.merged().covariance(), blocked.merged().covariance(), atol=1e-5)
    assert loaded.number_of_blocks == 3


def test_collection_lag_sums_match_the_direct_formula():
    from nds.activations import median_lag_one_correlation

    generator = np.random.default_rng(7)
    sequences = np.zeros((100, 60, 3))
    for t in range(1, 60):
        sequences[:, t, :] = 0.5 * sequences[:, t - 1, :] + generator.standard_normal((100, 3))

    earlier = sequences[:, :-1, :].reshape(-1, 3)
    later = sequences[:, 1:, :].reshape(-1, 3)
    sums = {
        "pairs": earlier.shape[0],
        "product": (earlier * later).sum(axis=0),
        "first": earlier.sum(axis=0),
        "second": later.sum(axis=0),
        "first_squared": (earlier * earlier).sum(axis=0),
        "second_squared": (later * later).sum(axis=0),
    }
    median_value = median_lag_one_correlation(sums)
    assert abs(median_value - np.median(lag_one_correlations(sequences))) < 0.02


def test_mean_token_cosine_extremes_and_rotation():
    from nds.token_sample import mean_token_cosine

    generator = np.random.default_rng(8)

    same_direction = np.outer(np.abs(generator.standard_normal(50)) + 0.1, np.ones(20))
    assert abs(mean_token_cosine(same_direction, centre=False) - 1.0) < 1e-12

    spread = generator.standard_normal((3000, 40))
    assert abs(mean_token_cosine(spread, centre=True)) < 0.01

    rotation, _ = np.linalg.qr(generator.standard_normal((40, 40)))
    shifted = spread + 2.0
    assert abs(mean_token_cosine(shifted, centre=False) - mean_token_cosine(shifted @ rotation, centre=False)) < 1e-10
