"""
check_b_mphate_intraslice.py
============================

Cross-check (b): M-PHATE's intraslice kernel on z-scored unit activations versus our neuron kernel.

M-PHATE (scottgigante/m-phate, commit f912eab):
    m_phate/utils.py:4-8      z-score every unit over the probe inputs (population std, ddof=0)
    m_phate/kernel.py:93-105  optional joint PCA of all (slice, unit) vectors to n_pca=100 dims
    m_phate/kernel.py:70-76   per slice: graphtools.Graph(x, knn=intraslice_knn=2, decay=5).K
    m_phate/kernel.py:78-88   interslice: same unit across slices, fixed Gaussian bandwidth

Ours: nds covariance -> correlation -> d^2 = 2(1 - r) -> adaptive_gaussian_kernel(k=2, decay=5).

Why they should agree: for z-scored vectors of length p, ||z_i - z_j||^2 = 2 p (1 - r_ij), i.e.
M-PHATE's distance is sqrt(p) times ours, and the alpha-decay kernel exp(-(d/sigma_i)^a) is unchanged
when every distance is multiplied by the same constant (sigma_i scales with it).

Three versions are compared on each slice:
    ours        nds.graphs.adaptive_gaussian_kernel(..., keep_self_loops=True)
    reimpl      re-written from m_phate/utils.py + graphtools (kNN alpha decay, thresh 1e-4, "+")
    m_phate     unmodified m_phate/kernel.py + graphtools source (logging stubs in _stubs/)

Data: nds.synthetic.modular_activations, one slice per "epoch" with rising module strength.
Run:  python3 -I check_b_mphate_intraslice.py
"""

import importlib.util
import sys

HERE = "/home/claude/pg/research/checks"
sys.path[:0] = [
    HERE + "/_stubs",
    "/home/claude/refcode/KrishnaswamyLab__graphtools",
    "/home/claude/pg/repo/neuron-diffusion-spectroscopy/src",
]

import warnings

import numpy as np
from scipy.spatial.distance import pdist, squareform

from nds.graphs import adaptive_gaussian_kernel
from nds.graphs import correlation_distance_squared
from nds.graphs import correlation_from_covariance
from nds.moments import NeuronMoments
from nds.synthetic import modular_activations

warnings.filterwarnings("ignore")


def load_module(name, path):
    """Import one file of m_phate without running m_phate/__init__.py (which needs phate, keras)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MPHATE = "/home/claude/refcode/scottgigante__m-phate/m_phate/"
mphate_kernel = load_module("mphate_kernel", MPHATE + "kernel.py")
mphate_utils = load_module("mphate_utils", MPHATE + "utils.py")


# ----------------------------------------------------------------------------------------------
# M-PHATE intraslice kernel, re-implemented from reading the source
# ----------------------------------------------------------------------------------------------

def zscore_units(tensor):
    """m_phate.utils.normalize: per (slice, unit), subtract the mean and divide by the population std."""
    centred = tensor - tensor.mean(axis=2, keepdims=True)
    std = centred.std(axis=2, keepdims=True)
    return centred / np.where(std > 0, std, 1)


def intraslice_kernel_reimplemented(units_by_inputs, knn=2, decay=5.0, thresh=1e-4):
    """graphtools kNN alpha-decay kernel on one slice (Euclidean, k-th neighbour bandwidth, '+')."""
    distances = squareform(pdist(units_by_inputs))
    bandwidths = np.sort(distances, axis=1)[:, knn]          # column 0 is the unit itself
    one_sided = np.exp(-(distances / bandwidths[:, None]) ** decay)
    one_sided[one_sided < thresh] = 0
    return (one_sided + one_sided.T) / 2


def our_kernel(units_by_inputs, knn=2, decay=5.0):
    """Our pipeline: streaming moments -> covariance -> correlation -> d^2 -> kernel."""
    moments = NeuronMoments(number_of_neurons=units_by_inputs.shape[0])
    moments.add_batch(units_by_inputs.T)                      # (tokens, neurons)
    correlation = correlation_from_covariance(moments.covariance())
    squared_distances = correlation_distance_squared(correlation, signed=True)
    return adaptive_gaussian_kernel(squared_distances, neighbour_rank=knn, decay=decay, keep_self_loops=True)


def build_tensor(number_of_slices, number_of_inputs):
    """(slices, units, inputs): one modular_activations draw per slice, module strength 0.2 -> 0.8."""
    slices = []
    for index, strength in enumerate(np.linspace(0.2, 0.8, number_of_slices)):
        activations, _ = modular_activations(
            number_of_tokens=number_of_inputs,
            module_sizes=(16, 16, 16, 16),
            module_strength=strength,
            random_generator=np.random.default_rng(index),
        )
        slices.append(activations.T)
    return np.stack(slices)


def compare(number_of_inputs, number_of_slices=30, n_pca=100):
    tensor = build_tensor(number_of_slices, number_of_inputs)
    n, m, p = tensor.shape

    normalised = mphate_utils.normalize(tensor)
    assert np.allclose(normalised, zscore_units(tensor))

    full = mphate_kernel.multislice_kernel(
        normalised, intraslice_knn=2, interslice_knn=25, decay=5, n_pca=n_pca, n_jobs=1
    ).toarray()

    worst_reimpl = 0.0
    worst_mphate = 0.0
    worst_mphate_vs_reimpl = 0.0
    for tau in range(n):
        block = slice(tau * m, (tau + 1) * m)
        ours = our_kernel(tensor[tau])
        reimpl = intraslice_kernel_reimplemented(normalised[tau])
        from_mphate = full[block, block]
        worst_reimpl = max(worst_reimpl, np.abs(ours - reimpl).max())
        worst_mphate = max(worst_mphate, np.abs(ours - from_mphate).max())
        worst_mphate_vs_reimpl = max(worst_mphate_vs_reimpl, np.abs(reimpl - from_mphate).max())

    # interslice part: unit v at slice a <-> unit v at slice b, fixed bandwidth
    per_unit = [squareform(pdist(normalised[:, v, :])) for v in range(m)]
    bandwidth = np.mean([max(np.median(np.sort(D, axis=1)[:, 25]), np.finfo(float).eps) for D in per_unit])
    worst_inter = 0.0
    for v in range(m):
        expected = np.exp(-(per_unit[v] / bandwidth) ** 2)
        # the interslice diagonal is zeroed in m_phate (kernel.py:39); the 1 there comes from the
        # intraslice block, so the full K has 1 on these diagonal positions
        np.fill_diagonal(expected, 1)
        nodes = np.arange(n) * m + v
        worst_inter = max(worst_inter, np.abs(full[np.ix_(nodes, nodes)] - expected).max())

    pca_used = n_pca is not None and n_pca < p
    print("p = {:>5} probe inputs, {} slices x {} units, PCA to {} dims applied: {}".format(
        p, n, m, n_pca, pca_used))
    print("   max |K_ours - K_reimpl|           = {:.2e}".format(worst_reimpl))
    print("   max |K_ours - K_mphate block|     = {:.2e}".format(worst_mphate))
    print("   max |K_reimpl - K_mphate block|   = {:.2e}".format(worst_mphate_vs_reimpl))
    print("   interslice: bandwidth eps = {:.3f} (= {:.3f} sqrt(p));  max |reimpl - m_phate| = {:.2e}".format(
        bandwidth, bandwidth / np.sqrt(p), worst_inter))
    print("   diagonal of the full multislice K: min {:.3f}, max {:.3f}".format(
        full.diagonal().min(), full.diagonal().max()))


def main():
    compare(number_of_inputs=100)     # p = n_pca: no PCA, should match up to the 1e-4 threshold
    compare(number_of_inputs=2000)    # p > n_pca: M-PHATE projects to 100 PCs first
    compare(number_of_inputs=2000, n_pca=None)


if __name__ == "__main__":
    main()
