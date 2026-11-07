"""
On fake data with known modules, the neuron graph must find them.
"""

import numpy as np

from nds.graphs import neuron_graph_from_covariance, correlation_distance_squared
from nds.spectra import (
    diffusion_spectrum,
    diffusion_spectral_entropy,
    eigengap_module_count,
    inverse_participation_ratio,
    spectrum_summary,
)
from nds.synthetic import modular_activations


def test_distance_is_two_times_one_minus_correlation():
    correlation = np.array(
        [
            [1.0, 0.5, -1.0],
            [0.5, 1.0, 0.0],
            [-1.0, 0.0, 1.0],
        ]
    )

    signed = correlation_distance_squared(correlation, signed=True)
    unsigned = correlation_distance_squared(correlation, signed=False)

    assert np.isclose(signed[0, 1], 1.0)
    assert np.isclose(signed[0, 2], 4.0)
    assert np.isclose(unsigned[0, 2], 0.0)
    assert np.allclose(np.diag(signed), 0.0)


def test_four_modules_are_found():
    activations, labels = modular_activations(module_sizes=(30, 30, 30, 30), module_strength=0.7)

    covariance = np.cov(activations, rowvar=False)

    graph = neuron_graph_from_covariance(covariance, neighbour_rank=10)
    summary = spectrum_summary(graph["kernel"])

    assert summary["module_count"] == 4
    assert graph["dead_fraction"] == 0.0


def test_laplacian_eigenvalues_are_in_range_and_start_at_zero():
    activations, _ = modular_activations(module_sizes=(20, 20, 20))
    covariance = np.cov(activations, rowvar=False)
    graph = neuron_graph_from_covariance(covariance, neighbour_rank=8)

    spectrum = diffusion_spectrum(graph["kernel"])
    mu = spectrum["laplacian_eigenvalues"]

    assert abs(mu[0]) < 1e-10
    assert mu.min() > -1e-10
    assert mu.max() < 2.0 + 1e-10


def test_diffusion_coordinates_are_right_eigenvectors_of_P():
    activations, _ = modular_activations(module_sizes=(15, 15))
    covariance = np.cov(activations, rowvar=False)
    kernel = neuron_graph_from_covariance(covariance, neighbour_rank=5)["kernel"]

    spectrum = diffusion_spectrum(kernel)

    transition = kernel / kernel.sum(axis=1, keepdims=True)

    for mode in range(3):
        psi = spectrum["diffusion_coordinates"][:, mode]
        lam = spectrum["diffusion_eigenvalues"][mode]
        assert np.allclose(transition @ psi, lam * psi, atol=1e-10)


def test_dse_matches_liao_et_al_formula():
    eigenvalues = np.array([1.0, 0.9, -0.5, 0.1, 0.0])

    # the dse_dsmi reference steps, written independently
    powered = np.abs(eigenvalues) ** 2
    probabilities = powered / powered.sum() + np.finfo(float).eps
    expected = -np.sum(probabilities * np.log2(probabilities))

    assert np.isclose(diffusion_spectral_entropy(eigenvalues, diffusion_time=2), expected)


def test_dse_approaches_log2_k_for_k_separated_modules():
    # k exact eigenvalues at 1, the rest tiny, large t
    eigenvalues = np.concatenate([np.ones(4), 0.3 * np.ones(50)])

    entropy = diffusion_spectral_entropy(eigenvalues, diffusion_time=64)

    assert abs(entropy - 2.0) < 1e-6


def test_eigengap_picks_the_jump():
    mu = np.array([0.0, 0.01, 0.02, 0.5, 0.55, 0.6])

    k_star, gaps, gap_ratio = eigengap_module_count(mu, maximum_modules=5)

    assert k_star == 3
    assert gap_ratio > 5


def test_ipr_extremes():
    assert np.isclose(inverse_participation_ratio([1.0, 0.0, 0.0, 0.0]), 1.0)
    assert np.isclose(inverse_participation_ratio([0.5, 0.5, 0.5, 0.5]), 0.25)


def test_localization_ratio_is_about_one_for_random_vectors():
    from nds.spectra import localization_ratio

    random_generator = np.random.default_rng(0)
    vector = random_generator.standard_normal(5000)

    assert 0.9 < localization_ratio(vector) < 1.1

    localized = np.zeros(5000)
    localized[:50] = 1.0
    assert np.isclose(localization_ratio(localized), 5000 / (3 * 50))


def test_subspace_localization_ignores_mixing_inside_the_subspace():
    from nds.spectra import subspace_localization

    vectors = np.zeros((100, 2))
    vectors[:10, 0] = 1 / np.sqrt(10)
    vectors[10:20, 1] = 1 / np.sqrt(10)

    angle = 0.7
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    assert np.isclose(subspace_localization(vectors), 100 / 20)
    assert np.isclose(subspace_localization(vectors @ rotation), 100 / 20)


def test_alpha_decay_kernel_matches_the_graphtools_formula():
    """
    graphtools kNN kernel, written out from its source: bandwidth = distance to the k-th neighbour,
    affinity exp(-(d / bandwidth) ** decay), symmetrised (K + K^T) / 2, diagonal 1.
    (Code-study check S1 found our kernel identical to unmodified graphtools with thresh off.)
    """
    from nds.graphs import adaptive_gaussian_kernel

    random_generator = np.random.default_rng(0)
    points = random_generator.standard_normal((30, 4))
    squared_distances = np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=2)

    ours = adaptive_gaussian_kernel(
        squared_distances,
        neighbour_rank=5,
        kernel_type="alpha_decay",
        decay=40.0,
        keep_self_loops=True,
    )

    distances = np.sqrt(squared_distances)
    bandwidth = np.sort(distances, axis=1)[:, 5]
    one_sided = np.exp(-(distances / bandwidth[:, None]) ** 40.0)
    expected = (one_sided + one_sided.T) / 2
    np.fill_diagonal(expected, 1.0)

    assert np.allclose(ours, expected)


def test_anisotropy_one_removes_degree_scaling():
    from nds.graphs import apply_anisotropy

    kernel = np.array(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 3.0],
            [1.0, 3.0, 0.0],
        ]
    )

    unchanged = apply_anisotropy(kernel, 0.0)
    normalised = apply_anisotropy(kernel, 1.0)

    degrees = kernel.sum(axis=1)

    assert np.array_equal(unchanged, kernel)
    assert np.isclose(normalised[1, 2], 3.0 / (degrees[1] * degrees[2]))


def test_unknown_kernel_type_is_rejected():
    import pytest
    from nds.graphs import adaptive_gaussian_kernel

    with pytest.raises(ValueError):
        adaptive_gaussian_kernel(np.ones((12, 12)), neighbour_rank=3, kernel_type="gaussian")
