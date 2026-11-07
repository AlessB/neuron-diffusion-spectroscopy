"""
check_a_graphtools_kernel.py
============================

Cross-check (a): graphtools' kNN alpha-decay kernel versus nds.graphs.adaptive_gaussian_kernel.

Three versions of "the same" kernel on the same synthetic neurons:

    ours        nds.graphs.adaptive_gaussian_kernel(d^2, neighbour_rank=k, decay=a, keep_self_loops=...)
    reimpl      a few lines re-written from graphtools/graphs.py (TraditionalGraph.build_kernel,
                lines 1574-1609) and graphtools/base.py (symmetrize_kernel, line 557)
    graphtools  the unmodified graphtools source (HEAD 785c0c1), imported through sys.path with the
                logging/pygsp stubs in _stubs/ (no numba, so the pure numpy code path runs)

Data: nds.synthetic.modular_activations (4 modules x 32 neurons, 2000 tokens).
Run:  python3 -I check_a_graphtools_kernel.py
"""

import sys

HERE = "/home/claude/pg/research/checks"
sys.path[:0] = [
    HERE + "/_stubs",
    "/home/claude/refcode/KrishnaswamyLab__graphtools",
    "/home/claude/pg/repo/neuron-diffusion-spectroscopy/src",
]

import warnings

import numpy as np
from scipy.linalg import eigh

import graphtools

from nds.graphs import adaptive_gaussian_kernel
from nds.graphs import correlation_distance_squared
from nds.graphs import correlation_from_covariance
from nds.spectra import diffusion_spectrum
from nds.spectra import eigengap_module_count
from nds.synthetic import modular_activations

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------------------------------
# graphtools kernel, re-implemented from reading the source
# ----------------------------------------------------------------------------------------------

def graphtools_alpha_decay_reimplemented(distances, knn=5, decay=40.0, thresh=1e-4):
    """
    graphtools TraditionalGraph.build_kernel on a precomputed distance matrix, then "+" symmetrisation.

        sigma_i = max of the knn+1 smallest distances in row i (self included) = k-th neighbour
        A_ij    = exp(-(d_ij / sigma_i) ** decay)
        A_ij    = 0 where A_ij < thresh
        K       = (A + A^T) / 2                     diagonal stays 1 (self loops kept)
    """
    smallest = np.partition(distances, knn + 1, axis=1)[:, : knn + 1]
    bandwidths = np.max(smallest, axis=1)

    scaled = (distances.T / bandwidths).T
    one_sided = np.exp(-1 * np.power(scaled, decay))
    one_sided[one_sided < thresh] = 0

    kernel = (one_sided + one_sided.T) / 2
    return kernel


def low_spectrum(kernel, how_many=8):
    """First Laplacian eigenvalues and the eigengap module count of a kernel."""
    spectrum = diffusion_spectrum(kernel)
    k_star, _ = eigengap_module_count(spectrum["laplacian_eigenvalues"], maximum_modules=20)
    return spectrum["laplacian_eigenvalues"][:how_many], k_star


def main():
    activations, labels = modular_activations(
        number_of_tokens=2000,
        module_sizes=(32, 32, 32, 32),
        module_strength=0.7,
        random_generator=np.random.default_rng(0),
    )

    covariance = np.cov(activations, rowvar=False)
    correlation = correlation_from_covariance(covariance)
    squared_distances = correlation_distance_squared(correlation, signed=True)
    distances = np.sqrt(squared_distances)

    print("neurons:", distances.shape[0], " tokens:", activations.shape[0])
    print()
    print("A. same formula, decay != 2, self loops kept (ours keep_self_loops=True)")
    print("   max |K_ours - K_x| over all entries")
    print("   {:>6} {:>5} | {:>12} {:>12} {:>12} | {:>14}".format(
        "decay", "k", "reimpl t=1e-4", "reimpl t=0", "graphtools", "gt kNN (data)"))

    # z-scored neuron profiles, rows = neurons, population std as in M-PHATE / sklearn
    centred = activations - activations.mean(axis=0)
    zscored = (centred / centred.std(axis=0)).T

    for decay in (40.0, 5.0, 2.5):
        for knn in (5, 10):
            ours = adaptive_gaussian_kernel(
                squared_distances, neighbour_rank=knn, decay=decay, keep_self_loops=True
            )

            reimpl = graphtools_alpha_decay_reimplemented(distances, knn=knn, decay=decay, thresh=1e-4)
            reimpl_exact = graphtools_alpha_decay_reimplemented(distances, knn=knn, decay=decay, thresh=0.0)

            exact_graph = graphtools.Graph(
                distances, precomputed="distance", knn=knn, decay=decay, thresh=1e-4,
                n_pca=None, verbose=False,
            )
            graphtools_kernel = np.asarray(exact_graph.K)

            # the default graphtools path on data: kNNGraph, euclidean on z-scored profiles
            knn_graph = graphtools.Graph(
                zscored, knn=knn, decay=decay, thresh=1e-4, n_pca=None, verbose=False,
            )
            knn_kernel = knn_graph.K.toarray()

            print("   {:>6} {:>5} | {:12.2e} {:12.2e} {:12.2e} | {:14.2e}".format(
                decay, knn,
                np.abs(ours - reimpl).max(),
                np.abs(ours - reimpl_exact).max(),
                np.abs(ours - graphtools_kernel).max(),
                np.abs(ours - knn_kernel).max(),
            ))

    print()
    print("B. decay = 2: ours switches to the Zelnik-Manor form exp(-d^2/(s_i s_j)),")
    print("   graphtools keeps (exp(-d^2/s_i^2) + exp(-d^2/s_j^2)) / 2")
    for knn in (5, 10):
        ours = adaptive_gaussian_kernel(squared_distances, neighbour_rank=knn, decay=2.0, keep_self_loops=True)
        ours_near2 = adaptive_gaussian_kernel(squared_distances, neighbour_rank=knn, decay=2.0 + 1e-9, keep_self_loops=True)
        gt = graphtools_alpha_decay_reimplemented(distances, knn=knn, decay=2.0, thresh=0.0)

        eig_ours, k_ours = low_spectrum(ours)
        eig_gt, k_gt = low_spectrum(gt)

        print("   k={:>2}: max |K_ours(2) - K_gt(2)| = {:.3f};  max |K_ours(2) - K_ours(2+1e-9)| = {:.3f}".format(
            knn, np.abs(ours - gt).max(), np.abs(ours - ours_near2).max()))
        print("          mu_0..7 ours: {}   k*={}".format(np.array2string(eig_ours, precision=4), k_ours))
        print("          mu_0..7 gt  : {}   k*={}".format(np.array2string(eig_gt, precision=4), k_gt))

    print()
    print("C. self loops: graphtools K has K_ii = 1; our default keep_self_loops=False sets K_ii = 0")
    for decay, knn in ((40.0, 5), (2.0, 10)):
        with_loops = adaptive_gaussian_kernel(squared_distances, neighbour_rank=knn, decay=decay, keep_self_loops=True)
        without = adaptive_gaussian_kernel(squared_distances, neighbour_rank=knn, decay=decay, keep_self_loops=False)

        degrees = without.sum(axis=1)
        eig_with, k_with = low_spectrum(with_loops)
        eig_without, k_without = low_spectrum(without)

        print("   decay={:>4}, k={:>2}: mean degree without loops = {:.2f} (a self loop adds 1)".format(
            decay, knn, degrees.mean()))
        print("          mu_0..7 with loops   : {}   k*={}".format(np.array2string(eig_with, precision=4), k_with))
        print("          mu_0..7 without loops: {}   k*={}".format(np.array2string(eig_without, precision=4), k_without))

        # agreement of the slow 4-mode subspace (modes 0..3). Single-vector overlaps are meaningless
        # when the graph splits into components (degenerate zero eigenvalues), so compare subspaces:
        # cosines of the principal angles = singular values of V_with^T V_without.
        vectors_with = diffusion_spectrum(with_loops)["symmetric_vectors"][:, :4]
        vectors_without = diffusion_spectrum(without)["symmetric_vectors"][:, :4]
        cosines = np.linalg.svd(vectors_with.T @ vectors_without, compute_uv=False)
        print("          principal-angle cosines, slow 4-mode subspace: {}".format(
            np.array2string(cosines, precision=4)))

    print()
    print("D. connectivity of the PHATE-default kernel (k=5, decay=40) on this neuron graph")
    kernel = graphtools_alpha_decay_reimplemented(distances, knn=5, decay=40.0, thresh=1e-4)
    from scipy.sparse.csgraph import connected_components
    count, component = connected_components(kernel > 0, directed=False)
    print("   connected components: {}  (true modules: {})".format(count, len(np.unique(labels))))
    for decay in (40.0, 10.0, 5.0):
        kernel = graphtools_alpha_decay_reimplemented(distances, knn=5, decay=decay, thresh=1e-4)
        count, _ = connected_components(kernel > 0, directed=False)
        print("   k=5, decay={:>4}: {} components".format(decay, count))


if __name__ == "__main__":
    main()
