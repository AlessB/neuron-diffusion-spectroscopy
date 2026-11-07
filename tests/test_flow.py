"""
Mode matching and the avoided-crossing fit.
"""

import numpy as np

from nds.flow import (
    fit_avoided_crossing,
    match_modes,
    restrict_to_common_neurons,
    two_level_eigenvalues,
)


def test_matching_recovers_a_permutation():
    random_generator = np.random.default_rng(0)
    q, _ = np.linalg.qr(random_generator.standard_normal((20, 5)))

    permutation = np.array([2, 0, 4, 1, 3])
    shuffled = q[:, permutation] * np.array([1, -1, 1, -1, 1])

    result = match_modes(q, shuffled)

    for mode in range(5):
        assert permutation[result["successor"][mode]] == mode
        assert np.isclose(result["overlap"][mode], 1.0)


def test_restrict_keeps_only_common_neurons():
    vectors_a = np.eye(4)[:, :2]
    vectors_b = np.eye(3)[:, :2]

    restricted_a, restricted_b, kept_a, kept_b = restrict_to_common_neurons(
        vectors_a,
        np.array([0, 1, 2, 3]),
        vectors_b,
        np.array([0, 1, 3]),
    )

    assert restricted_a.shape == (3, 2)
    assert restricted_b.shape == (3, 2)
    assert np.allclose(kept_a, [1.0, 1.0])


def test_avoided_crossing_fit_recovers_the_coupling():
    times = np.linspace(-1.0, 1.0, 41)

    lower, upper = two_level_eigenvalues(
        times,
        centre_intercept=0.7,
        centre_slope=0.05,
        sweep_rate=0.4,
        crossing_time=0.1,
        coupling=0.02,
    )

    noise = np.random.default_rng(0).normal(scale=1e-4, size=(2, len(times)))

    fit = fit_avoided_crossing(times, lower + noise[0], upper + noise[1])

    assert abs(fit["coupling"] - 0.02) < 1e-3
    assert abs(fit["crossing_time"] - 0.1) < 1e-2
    assert abs(fit["sweep_rate"] - 0.4) < 1e-2


def test_three_module_toy_shows_an_avoided_crossing():
    """
    The deep-dive probe T3, run through the library: the slowest non-trivial mode moves from
    module C to module A as `position` goes 0 -> 1, and the two eigenvalues come closest midway.
    """
    from nds.graphs import neuron_graph_from_covariance
    from nds.spectra import diffusion_spectrum
    from nds.synthetic import three_module_path

    positions = np.linspace(0.0, 1.0, 11)

    lower = []
    upper = []
    weight_on_a = []

    for position in positions:
        activations, labels = three_module_path(position, direct_coupling=0.1)
        covariance = np.cov(activations, rowvar=False)

        kernel = neuron_graph_from_covariance(covariance, neighbour_rank=10)["kernel"]
        spectrum = diffusion_spectrum(kernel)

        mu = spectrum["laplacian_eigenvalues"]
        lower.append(mu[1])
        upper.append(mu[2])

        slowest = spectrum["symmetric_vectors"][:, 1]
        weight_on_a.append(np.sum(slowest[labels == 0] ** 2))

    fit = fit_avoided_crossing(positions, np.array(lower), np.array(upper))

    assert weight_on_a[0] < 0.3
    assert weight_on_a[-1] > 0.5
    assert 0.3 < fit["crossing_time"] < 0.7
    assert fit["minimum_gap"] < 0.1


def test_group_matching_ignores_mixing_inside_a_degenerate_pair():
    from nds.flow import group_near_degenerate_modes
    from nds.flow import match_mode_groups
    from nds.flow import match_modes

    generator = np.random.default_rng(11)

    basis, _ = np.linalg.qr(generator.standard_normal((80, 4)))

    # step a: modes 1 and 2 are a degenerate pair; step b: the same pair, mixed by 45 degrees
    angle = np.pi / 4
    mixing = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    vectors_a = basis.copy()
    vectors_b = basis.copy()
    vectors_b[:, 1:3] = basis[:, 1:3] @ mixing

    single = match_modes(vectors_a, vectors_b)
    assert single["overlap"][1] < 0.8

    groups_a = group_near_degenerate_modes([0.1, 0.30, 0.301, 0.6], tolerance=0.01)
    groups_b = group_near_degenerate_modes([0.1, 0.30, 0.302, 0.6], tolerance=0.01)
    assert groups_a == [[0], [1, 2], [3]]

    grouped = match_mode_groups(vectors_a, groups_a, vectors_b, groups_b)
    assert list(grouped["successor"]) == [0, 1, 2]
    assert np.all(grouped["overlap"] > 1.0 - 1e-10)


def test_crossing_diagnostics_and_predicted_mixing():
    from nds.flow import crossing_diagnostics
    from nds.flow import predicted_mixing

    fit = {"coupling": 0.05, "sweep_rate": 0.5, "crossing_time": 3.0}

    wide = crossing_diagnostics(fit, checkpoint_spacing=0.05)
    assert abs(wide["width"] - 0.2) < 1e-12
    assert wide["resolved"]

    narrow = crossing_diagnostics(fit, checkpoint_spacing=0.3)
    assert not narrow["resolved"]

    times = np.linspace(1.0, 5.0, 81)
    overlaps = predicted_mixing(times, fit)

    assert abs(overlaps[0] - 1.0) < 1e-12
    assert overlaps[-1] < 0.01
    # half mixed exactly at the crossing time
    middle = int(np.argmin(np.abs(times - 3.0)))
    assert abs(overlaps[middle] - 0.5) < 0.05
