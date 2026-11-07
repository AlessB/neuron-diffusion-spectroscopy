"""
modules.py
==========

Turning the slow modes into an actual list of modules (which neuron belongs where), and scoring
how well separated those modules are.

Partition
---------
The standard recipe (Ng, Jordan & Weiss 2002; von Luxburg 2007, Sec. 4): take the first k
eigenvectors of the normalised Laplacian as k coordinates per neuron, normalise each neuron's row
to unit length, and run k-means on those rows. We use scipy's k-means (several restarts, fixed
seed) so the result is reproducible.

Conductance and Upsilon
-----------------------
For a set S of neurons, its conductance is the weight of edges leaving S divided by the total
weight attached to S:
    phi(S) = w(S, rest) / vol(S),        vol(S) = sum of degrees in S
A good module has low conductance. Peng, Sun & Zanetti (2017) show spectral clustering provably
recovers the modules when

    Upsilon(k) = lambda_(k+1) / rho(k)

is large, where lambda_(k+1) is the (k+1)-th smallest Laplacian eigenvalue (1-indexed, so with our
0-indexed array mu[k]) and rho(k) is the worst conductance among the k modules. We report Upsilon
computed with the partition we found (so it is an estimate: the optimal partition can only have
lower rho, i.e. higher Upsilon).

Credit
------
- A. Y. Ng, M. I. Jordan, Y. Weiss (2002). "On spectral clustering: analysis and an algorithm." NIPS 14.
- U. von Luxburg (2007). "A tutorial on spectral clustering." Statistics and Computing 17(4).
- R. Peng, H. Sun, L. Zanetti (2017). "Partitioning well-clustered graphs: spectral clustering
  works!" SIAM Journal on Computing 46(2):710-743. (Upsilon, their eq. 3)
"""

import numpy as np
from scipy.cluster.vq import kmeans2


def spectral_partition(symmetric_vectors, number_of_modules, random_seed=0, restarts=10):
    """
    Assign every neuron to one of `number_of_modules` modules.

    symmetric_vectors: (neurons, modes) eigenvectors of L_sym, sorted from the smallest eigenvalue.
    Returns an integer label per neuron.
    """
    coordinates = symmetric_vectors[:, :number_of_modules].astype(np.float64)

    row_norms = np.linalg.norm(coordinates, axis=1, keepdims=True)
    row_norms = np.maximum(row_norms, 1e-300)

    coordinates = coordinates / row_norms

    best_labels = None
    best_inertia = np.inf

    random_generator = np.random.default_rng(random_seed)

    for _ in range(restarts):
        seed = int(random_generator.integers(0, 2**31 - 1))

        centroids, labels = kmeans2(coordinates, number_of_modules, minit="++", seed=seed)

        inertia = np.sum((coordinates - centroids[labels]) ** 2)

        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels

    return best_labels


def conductances(kernel, labels):
    """
    phi(S) = w(S, rest) / vol(S) for every module label. Returns a dict label -> conductance.
    """
    degrees = kernel.sum(axis=1)

    result = {}

    for label in np.unique(labels):
        inside = labels == label

        volume = degrees[inside].sum()

        weight_leaving = kernel[np.ix_(inside, ~inside)].sum()

        if volume > 0:
            result[int(label)] = float(weight_leaving / volume)
        else:
            result[int(label)] = float("nan")

    return result


def upsilon(laplacian_eigenvalues, kernel, labels):
    """
    Peng-Sun-Zanetti's Upsilon(k) = lambda_(k+1) / rho(k), estimated with our partition.

    laplacian_eigenvalues: sorted ascending, 0-indexed (mu[0] = 0), so lambda_(k+1) = mu[k].
    """
    number_of_modules = len(np.unique(labels))

    worst_conductance = max(conductances(kernel, labels).values())

    if worst_conductance <= 0:
        return float("inf")

    return float(laplacian_eigenvalues[number_of_modules] / worst_conductance)


def adjusted_rand_index(labels_a, labels_b):
    """
    How alike two partitions of the same neurons are: 1 for identical (up to renaming the
    modules), about 0 for unrelated, can be slightly negative. Hubert and Arabie (1985):

        ARI = (index - expected) / (maximum - expected)

    with index = sum over module pairs of C(n_ij, 2) from the contingency table n_ij,
    expected = [sum_i C(a_i, 2)] [sum_j C(b_j, 2)] / C(n, 2), maximum = mean of the two row/column
    sums. Used for partition stability between neighbouring checkpoints (DECISIONS D33).

    Credit: L. Hubert, P. Arabie (1985). "Comparing partitions." Journal of Classification 2(1).
    """
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)

    if len(labels_a) != len(labels_b):
        raise ValueError("both partitions must cover the same neurons")

    _, codes_a = np.unique(labels_a, return_inverse=True)
    _, codes_b = np.unique(labels_b, return_inverse=True)

    table = np.zeros((codes_a.max() + 1, codes_b.max() + 1))
    np.add.at(table, (codes_a, codes_b), 1.0)

    def pairs(counts):
        return counts * (counts - 1.0) / 2.0

    index = pairs(table).sum()

    row_pairs = pairs(table.sum(axis=1)).sum()
    column_pairs = pairs(table.sum(axis=0)).sum()

    total_pairs = pairs(float(len(labels_a)))

    expected = row_pairs * column_pairs / total_pairs

    maximum = (row_pairs + column_pairs) / 2.0

    if maximum == expected:
        return 1.0

    return float((index - expected) / (maximum - expected))
