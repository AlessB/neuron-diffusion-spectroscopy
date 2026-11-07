"""
Spectral partition recovers planted modules; conductance and Upsilon behave as expected.
"""

import numpy as np

from nds.graphs import neuron_graph_from_covariance
from nds.modules import conductances, spectral_partition, upsilon
from nds.spectra import diffusion_spectrum
from nds.synthetic import modular_activations


def adjusted_agreement(labels_a, labels_b):
    """
    Fraction of neuron pairs on which two labelings agree (same module / different module).
    """
    same_a = labels_a[:, None] == labels_a[None, :]
    same_b = labels_b[:, None] == labels_b[None, :]

    return float(np.mean(same_a == same_b))


def test_partition_recovers_planted_modules():
    activations, true_labels = modular_activations(module_sizes=(20, 20, 20, 20), module_strength=0.7)
    graph = neuron_graph_from_covariance(np.cov(activations, rowvar=False), neighbour_rank=8)

    spectrum = diffusion_spectrum(graph["kernel"])
    labels = spectral_partition(spectrum["symmetric_vectors"], 4)

    assert adjusted_agreement(labels, true_labels) > 0.99

    phi = conductances(graph["kernel"], labels)
    assert max(phi.values()) < 0.5

    assert upsilon(spectrum["laplacian_eigenvalues"], graph["kernel"], labels) > 1.0


def test_conductance_of_disconnected_module_is_zero():
    kernel = np.zeros((4, 4))
    kernel[0, 1] = kernel[1, 0] = 1.0
    kernel[2, 3] = kernel[3, 2] = 1.0

    phi = conductances(kernel, np.array([0, 0, 1, 1]))

    assert phi[0] == 0.0
    assert phi[1] == 0.0
