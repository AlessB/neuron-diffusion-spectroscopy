"""
Part 2 helpers: the exact W_out bridge, the 80% rule, retention probability.
"""

import numpy as np

from nds.graphs import neuron_graph_from_covariance
from nds.modules import conductances
from nds.naming import feature_loadings_on_neurons
from nds.naming import mean_absolute_contributions
from nds.naming import retention_probability
from nds.naming import retention_z_score
from nds.naming import smallest_set_holding_share
from nds.synthetic import modular_activations


def test_bridge_reproduces_feature_pre_activations_exactly():
    random_generator = np.random.default_rng(0)

    neurons = random_generator.standard_normal((50, 64))         # tokens x hidden neurons
    mlp_output_weight = random_generator.standard_normal((16, 64))
    mlp_output_bias = random_generator.standard_normal(16)
    encoder_weight = random_generator.standard_normal((5, 16))
    encoder_bias = random_generator.standard_normal(5)
    dictionary_bias = random_generator.standard_normal(16)

    loadings, offsets = feature_loadings_on_neurons(
        encoder_weight,
        encoder_bias,
        dictionary_bias,
        mlp_output_weight,
        mlp_output_bias,
    )

    mlp_output = neurons @ mlp_output_weight.T + mlp_output_bias
    expected = (mlp_output - dictionary_bias) @ encoder_weight.T + encoder_bias

    assert np.allclose(neurons @ loadings.T + offsets, expected)

    scores = mean_absolute_contributions(neurons, loadings, expected > 0)
    assert scores.shape == (5, 64)


def test_smallest_set_holding_share():
    scores = np.array([0.0, 5.0, 1.0, 4.0])

    assert list(smallest_set_holding_share(scores, share=0.8)) == [1, 3]


def test_retention_is_one_minus_conductance_and_high_for_a_module():
    activations, labels = modular_activations(module_sizes=(20, 20, 20), module_strength=0.7)
    kernel = neuron_graph_from_covariance(np.cov(activations, rowvar=False), neighbour_rank=8)["kernel"]

    module_rows = np.flatnonzero(labels == 0)

    r1 = retention_probability(kernel, module_rows, diffusion_time=1)
    phi = conductances(kernel, (labels == 0).astype(int))[1]

    assert np.isclose(r1, 1.0 - phi)

    observed, z_score, _ = retention_z_score(kernel, module_rows, number_of_random_sets=100)
    assert z_score > 5
