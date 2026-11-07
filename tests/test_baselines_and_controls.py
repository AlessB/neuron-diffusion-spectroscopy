"""
The heart of the project in two tests: rotating the neurons leaves every baseline unchanged
and destroys the neuron graph's modules.
"""

import numpy as np

from nds.baselines import (
    alpha_req,
    baseline_summary,
    covariance_eigenvalues,
    rankme_li,
)
from nds.controls import random_orthogonal_matrix, rotate_covariance
from nds.graphs import neuron_graph_from_covariance
from nds.spectra import spectrum_summary
from nds.synthetic import modular_activations


def test_random_orthogonal_matrix_is_orthogonal():
    q = random_orthogonal_matrix(6, np.random.default_rng(0))

    assert np.allclose(q.T @ q, np.eye(6))


def test_baselines_are_rotation_invariant_and_graph_is_not():
    activations, _ = modular_activations(module_sizes=(25, 25, 25, 25))
    covariance = np.cov(activations, rowvar=False)

    rotation = random_orthogonal_matrix(covariance.shape[0], np.random.default_rng(3))
    rotated = rotate_covariance(covariance, rotation)

    original_baselines = baseline_summary(covariance)
    rotated_baselines = baseline_summary(rotated)

    for name in original_baselines:
        assert np.isclose(original_baselines[name], rotated_baselines[name], rtol=1e-8), name

    original_graph = neuron_graph_from_covariance(covariance, neighbour_rank=10)
    rotated_graph = neuron_graph_from_covariance(rotated, neighbour_rank=10)

    original_modules = spectrum_summary(original_graph["kernel"])["module_count"]
    rotated_modules = spectrum_summary(rotated_graph["kernel"])["module_count"]

    assert original_modules == 4
    assert rotated_modules != 4


def test_rankme_li_matches_their_code():
    eigenvalues = np.array([5.0, 3.0, 1.0, 0.5, 0.0])

    # utils/powerlaw.py `rankme`, copied line by line
    l1 = np.sum(np.abs(eigenvalues))
    scores = eigenvalues / l1 + 1e-7
    expected = np.exp(-np.sum(scores * np.log(scores)))

    assert np.isclose(rankme_li(eigenvalues), expected)


def test_alpha_req_recovers_a_known_power_law():
    ranks = np.arange(1, 501)
    eigenvalues = 3.0 * ranks ** (-1.3)

    alpha, r_squared = alpha_req(eigenvalues)

    assert abs(alpha - 1.3) < 1e-8
    assert r_squared > 0.999999


def test_covariance_eigenvalues_sorted_and_nonnegative():
    covariance = np.array([[2.0, 0.0], [0.0, 5.0]])

    eigenvalues = covariance_eigenvalues(covariance)

    assert np.allclose(eigenvalues, [5.0, 2.0])


def test_rotation_destroys_localization_at_realistic_size():
    """
    At realistic size, rotation keeps slow modes but spreads them over all neurons.
    Localization (IPR x m / 3) is the readout that tells the two apart.
    """
    from nds.spectra import spectrum_summary

    random_generator = np.random.default_rng(0)
    number_of_tokens = 3000
    number_of_neurons = 600
    module_size = 40

    latents = random_generator.standard_normal((number_of_tokens, 6))
    activations = random_generator.standard_normal((number_of_tokens, number_of_neurons))
    for module in range(6):
        columns = slice(module * module_size, (module + 1) * module_size)
        noise = random_generator.standard_normal((number_of_tokens, module_size))
        activations[:, columns] = np.sqrt(0.3) * latents[:, [module]] + np.sqrt(0.7) * noise

    covariance = np.cov(activations, rowvar=False)
    rotated = rotate_covariance(covariance, random_orthogonal_matrix(number_of_neurons, np.random.default_rng(1)))

    original_summary = spectrum_summary(neuron_graph_from_covariance(covariance)["kernel"])
    rotated_summary = spectrum_summary(neuron_graph_from_covariance(rotated)["kernel"])

    # six modules of 40 among 600 neurons: the slow modes live on 240 neurons -> about 600/240 = 2.5
    assert original_summary["slow_subspace_localization"] > 2.0

    # rotated: spread over everything -> about 1 + 2/d
    assert rotated_summary["slow_subspace_localization"] < 1.6


def test_prompt_entropy_follows_skeans_recipe_and_ignores_rotation():
    from nds.baselines import normalised_token_rows
    from nds.baselines import prompt_entropy
    from nds.baselines import rankme_li
    from nds.baselines import covariance_eigenvalues
    from nds.baselines import matrix_entropy
    from nds.controls import random_orthogonal_matrix

    generator = np.random.default_rng(3)

    # one prompt: 64 tokens of 40 GELU-like neurons driven by 5 shared signals
    signals = generator.standard_normal((64, 5))
    mixing = generator.standard_normal((5, 40))
    activations = np.maximum(signals @ mixing + 0.3 * generator.standard_normal((64, 40)), 0.0)

    rows = normalised_token_rows(activations)

    # rows are centred tokens scaled to unit length
    assert np.allclose(np.linalg.norm(rows, axis=1), 1.0)

    # hand computation of their recipe: clamp the Gram at 0, divide by trace, von Neumann entropy
    gram = np.maximum(rows @ rows.T, 0.0)
    gram = gram / np.trace(gram)
    eigenvalues = np.linalg.eigvalsh(gram)
    eigenvalues = eigenvalues[eigenvalues > 0]
    probabilities = eigenvalues / eigenvalues.sum()
    expected = - np.sum(probabilities * np.log(probabilities))

    result = prompt_entropy(activations)

    assert abs(result["raw"] - expected) < 1e-10
    assert abs(result["normalised"] - expected / np.log(40)) < 1e-10

    # rotating the neurons changes nothing (it only uses token-token dot products)
    rotation = random_orthogonal_matrix(40, generator)
    rotated = prompt_entropy(activations @ rotation)
    assert abs(rotated["raw"] - result["raw"]) < 1e-9

    # and our covariance matrix entropy is log RankMe (why it is not counted as its own baseline)
    covariance = np.cov(activations, rowvar=False)
    covariance_spectrum = covariance_eigenvalues(covariance)
    difference = np.log(rankme_li(covariance_spectrum)) - matrix_entropy(covariance_spectrum)
    assert abs(difference) < 1e-2


def test_htsr_alpha_matches_one_plus_one_over_a_and_ignores_rotation():
    from nds.baselines import htsr_alpha

    generator = np.random.default_rng(9)

    # singular values chosen so the eigenvalues of W^T W decay like k^-1: alpha_HTSR = 2
    size = 400
    eigenvalues = 1.0 / np.arange(1, size + 1)
    left, _ = np.linalg.qr(generator.standard_normal((800, size)))
    right, _ = np.linalg.qr(generator.standard_normal((size, size)))
    weight = left @ np.diag(np.sqrt(eigenvalues)) @ right

    alpha, _, distance = htsr_alpha(weight)
    assert abs(alpha - 2.0) < 0.25
    assert distance < 0.1

    rotation, _ = np.linalg.qr(generator.standard_normal((800, 800)))
    rotated_alpha, _, _ = htsr_alpha(rotation.T @ weight)
    assert abs(rotated_alpha - alpha) < 1e-6
