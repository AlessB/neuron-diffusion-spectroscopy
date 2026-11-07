"""
synthetic_activations.py
========================

Shared synthetic data for the numerical cross-checks in this folder. No torch, numpy/scipy only.

`gelu_neurons` imitates an MLP layer's post-GELU activations: a low-rank Gaussian "input" with a
power-law latent spectrum, a random read-in matrix, a negative bias (so most neurons are off most
of the time, like real GELU neurons), then the exact (erf) GELU that GPT-NeoX/Pythia uses.
"""

import numpy as np
from scipy.special import erf


def gelu(x):
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def gelu_neurons(
    number_of_tokens=6000,
    number_of_neurons=512,
    latent_dimension=256,
    latent_decay=1.0,
    bias=-1.0,
    seed=0,
):
    """
    (tokens, neurons) float64 array of GELU activations with a power-law-ish covariance spectrum.
    """
    rng = np.random.default_rng(seed)

    latent_scales = np.arange(1, latent_dimension + 1) ** (-latent_decay / 2.0)
    latent = rng.standard_normal((number_of_tokens, latent_dimension)) * latent_scales

    read_in = rng.standard_normal((latent_dimension, number_of_neurons))
    read_in /= np.linalg.norm(read_in, axis=0, keepdims=True)

    pre_activation = 3.0 * latent @ read_in + bias + 0.1 * rng.standard_normal(
        (number_of_tokens, number_of_neurons)
    )

    return gelu(pre_activation)


def power_law_gaussian(number_of_tokens=6000, number_of_neurons=512, alpha=1.0, seed=0):
    """
    Gaussian data whose covariance eigenvalues are exactly k^-alpha in a random basis.
    """
    rng = np.random.default_rng(seed)

    scales = np.arange(1, number_of_neurons + 1) ** (-alpha / 2.0)
    data = rng.standard_normal((number_of_tokens, number_of_neurons)) * scales

    q_matrix, r_matrix = np.linalg.qr(rng.standard_normal((number_of_neurons, number_of_neurons)))
    q_matrix = q_matrix * np.sign(np.diag(r_matrix))

    return data @ q_matrix.T
