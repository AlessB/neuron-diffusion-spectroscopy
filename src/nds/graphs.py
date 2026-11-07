"""
graphs.py
=========

From a neuron covariance matrix to the neuron graph (a kernel matrix K over neurons).

The idea in one paragraph
-------------------------
Each neuron is a point. Its coordinates are its activations on every probe token (one axis per
token). Two neurons are close when they rise and fall together on the same text. After
standardising each neuron (mean 0, unit length), the squared Euclidean distance between two
neurons is exactly

    d^2(i, j) = 2 (1 - r_ij)

where r_ij is their Pearson correlation. So the whole graph can be built from the correlation
matrix, which we get from the covariance without ever storing activations (see moments.py).

The kernel turns distances into edge weights:

    K_ij = exp( - d^2(i, j) / (sigma_i * sigma_j) )

with a separate scale sigma_i for every neuron: the distance from neuron i to its k-th nearest
neighbour. A neuron in a dense region gets a small scale, one in a sparse region a large scale,
so both end up with a handful of strong neighbours. This "self-tuning" choice is from
Zelnik-Manor & Perona (2004). PHATE / graphtools use the same k-th-neighbour bandwidth with an
extra decay exponent (the "alpha-decay" kernel, Moon et al. 2019); we expose that as `decay`.

Credit
------
- L. Zelnik-Manor, P. Perona (2004). "Self-tuning spectral clustering." NeurIPS 17.
- K. R. Moon, D. van Dijk, Z. Wang, ..., G. Wolf, S. Krishnaswamy (2019). "Visualizing structure
  and transitions in high-dimensional biological data" (PHATE). Nature Biotechnology 37.
  Code: https://github.com/KrishnaswamyLab/PHATE and the kernel builder
  https://github.com/KrishnaswamyLab/graphtools (kNNGraph: bandwidth = distance to the knn-th
  neighbour, affinity = exp(-(d / bandwidth) ** decay), then symmetrised).
- S. Gigante, A. S. Charles, S. Krishnaswamy, G. Mishne (2019). "Visualizing the PHATE of neural
  networks" (M-PHATE). NeurIPS 32. They also build a kernel over hidden units from their
  activation profiles; we follow that idea and add the spectral analysis.
"""

import numpy as np


# ----------------------------------------------------------------------------------------------
# step 1: covariance -> correlation, with dead neurons removed
# ----------------------------------------------------------------------------------------------

def find_dead_neurons(
    covariance,
    firing_rate=None,
    minimum_variance=1e-10,
    minimum_firing_rate=1e-4,
):
    """
    Return a boolean mask, True for neurons we drop.

    A neuron is dead if it barely varies over the probe text (variance below `minimum_variance`)
    or almost never activates (firing rate below `minimum_firing_rate`, if a firing rate is given).

    Why drop them: a neuron that never changes has no correlation with anything. In the graph it
    becomes an isolated point, and every isolated point adds a zero Laplacian eigenvalue, which
    would look like an extra "module" that isn't there.
    """
    variances = np.diag(covariance)

    is_dead = variances < minimum_variance

    if firing_rate is not None:
        is_dead = is_dead | (np.asarray(firing_rate) < minimum_firing_rate)

    return is_dead


def correlation_from_covariance(covariance):
    """
    Pearson correlation r_ij = cov_ij / (std_i * std_j).

    Call this only after dead neurons are removed (a zero standard deviation would divide by zero).
    """
    standard_deviations = np.sqrt(np.diag(covariance))

    correlation = covariance / np.outer(standard_deviations, standard_deviations)

    # Rounding can push values a hair outside [-1, 1]; clip so 1 - r never goes negative.
    correlation = np.clip(correlation, -1.0, 1.0)

    np.fill_diagonal(correlation, 1.0)

    return correlation


# ----------------------------------------------------------------------------------------------
# step 2: correlation -> distance
# ----------------------------------------------------------------------------------------------

def correlation_distance_squared(correlation, signed=True):
    """
    Squared distance between standardised neuron profiles.

    signed=True   d^2 = 2 (1 - r)       anti-correlated neurons are FAR apart (d^2 = 4 at r = -1)
    signed=False  d^2 = 2 (1 - |r|)     anti-correlated neurons count as close (they carry the
                                        same information with opposite sign)

    The default is signed: two neurons that switch on for opposite inputs are doing different
    jobs. The unsigned version is kept as a robustness check (see DECISIONS.md).
    """
    if signed:
        similarity = correlation
    else:
        similarity = np.abs(correlation)

    squared_distances = 2.0 * (1.0 - similarity)

    squared_distances = np.maximum(squared_distances, 0.0)

    np.fill_diagonal(squared_distances, 0.0)

    return squared_distances


# ----------------------------------------------------------------------------------------------
# step 3: distance -> kernel (edge weights)
# ----------------------------------------------------------------------------------------------

def neighbour_bandwidths(squared_distances, neighbour_rank=10):
    """
    sigma_i = distance from neuron i to its k-th nearest neighbour (k = neighbour_rank).

    The neuron itself (distance 0) is not counted as a neighbour.
    """
    number_of_neurons = squared_distances.shape[0]

    if neighbour_rank >= number_of_neurons:
        raise ValueError(
            f"neighbour_rank={neighbour_rank} needs more than {number_of_neurons} neurons"
        )

    # sort each row; column 0 is the neuron itself (distance 0)
    sorted_squared_distances = np.sort(squared_distances, axis=1)

    kth_squared_distance = sorted_squared_distances[:, neighbour_rank]

    bandwidths = np.sqrt(kth_squared_distance)

    # A neuron with k exact duplicates would get bandwidth 0. Give it the smallest positive one.
    is_positive = bandwidths > 0

    if np.any(is_positive):
        smallest_positive = bandwidths[is_positive].min()
    else:
        smallest_positive = 1.0

    bandwidths = np.where(is_positive, bandwidths, smallest_positive)

    return bandwidths


def adaptive_gaussian_kernel(
    squared_distances,
    neighbour_rank=10,
    kernel_type="self_tuning",
    decay=40.0,
    keep_self_loops=False,
    threshold=0.0,
):
    """
    Edge weights between neurons. Two kernel families, chosen by `kernel_type`:

    "self_tuning" (default), Zelnik-Manor & Perona (2004):
        K_ij = exp( - d_ij^2 / (sigma_i sigma_j) )

    "alpha_decay", graphtools / PHATE (Moon et al. 2019):
        A_ij = exp( - (d_ij / sigma_i) ** decay )                  one-sided, not symmetric
        K    = (A + A^T) / 2                                       symmetrised as in graphtools
        `decay` is used only here. graphtools' default is 40 (the PHATE paper text says about 10).
        With decay = 2 this is NOT the self-tuning kernel (different denominator); the two
        families are kept as separate switches on purpose (code-study note S1, 17 Oct 2026).

    sigma_i is the distance to the neuron's `neighbour_rank`-th nearest neighbour in both cases.

    keep_self_loops:
        False (default) sets K_ii = 0, the usual choice for graph Laplacians (von Luxburg 2007).
        True keeps K_ii = 1, as graphtools and the diffusion-maps codes do. It makes the random
        walk "lazy" (it can stay put), which shifts eigenvalues of P towards 1. To reproduce a
        graphtools kernel exactly, use kernel_type="alpha_decay" and keep_self_loops=True.

    threshold:
        entries below it are set to 0 (graphtools drops entries below thresh = 1e-4). A threshold
        can split the graph into disconnected pieces; then the eigengap just counts pieces.
        Default 0 (keep every weak link).
    """
    bandwidths = neighbour_bandwidths(squared_distances, neighbour_rank)

    if kernel_type == "self_tuning":
        bandwidth_products = np.outer(bandwidths, bandwidths)

        kernel = np.exp(
            - squared_distances / bandwidth_products
        )

    elif kernel_type == "alpha_decay":
        distances = np.sqrt(squared_distances)

        scaled_distances = distances / bandwidths[:, np.newaxis]

        one_sided = np.exp(
            - scaled_distances ** decay
        )

        if threshold > 0:
            one_sided = np.where(one_sided >= threshold, one_sided, 0.0)

        kernel = (one_sided + one_sided.T) / 2.0

    else:
        raise ValueError(f"unknown kernel_type {kernel_type!r}: use 'self_tuning' or 'alpha_decay'")

    if kernel_type == "self_tuning" and threshold > 0:
        kernel = np.where(kernel >= threshold, kernel, 0.0)

    if keep_self_loops:
        np.fill_diagonal(kernel, 1.0)
    else:
        np.fill_diagonal(kernel, 0.0)

    return kernel


def apply_anisotropy(kernel, anisotropy=0.0):
    """
    Coifman & Lafon (2006) density normalisation:  K_ij / (d_i d_j)^anisotropy,  d = row sums of K.

    anisotropy 0 (default): no change; the diffusion then depends on how densely neurons are packed.
    anisotropy 1: removes the density effect (the limit operator is the Laplace-Beltrami operator).
    anisotropy 0.5: Fokker-Planck. graphtools calls this parameter `anisotropy` too.
    """
    if anisotropy == 0.0:
        return kernel

    degrees = kernel.sum(axis=1)

    scaling = degrees ** anisotropy

    return kernel / np.outer(scaling, scaling)


def fixed_bandwidth_gaussian_kernel(squared_distances, bandwidth=None):
    """
    The kernel used by the published DSE code (Liao et al.; dse_dsmi/diffusion.py), for comparison:

        G_ij = exp( - d_ij^2 / (2 bandwidth^2) )          one bandwidth for every point, self loops kept

    Their code fixes bandwidth = 10 on raw representation vectors. Our distances live on a
    different scale (d^2 = 2(1 - r) is between 0 and 4), so a fixed 10 would make every neuron
    look the same. When `bandwidth` is None we use the median distance between neurons, a common
    default for Gaussian kernels. (The constant factor 1/(sigma sqrt(2 pi)) in their code cancels
    in the normalisation and is left out.)

    This is a robustness check only: our main DSE uses the adaptive kernel above. See DECISIONS.md.
    """
    if bandwidth is None:
        off_diagonal = squared_distances[~np.eye(squared_distances.shape[0], dtype=bool)]
        bandwidth = float(np.sqrt(np.median(off_diagonal)))

    kernel = np.exp(
        - squared_distances / (2.0 * bandwidth ** 2)
    )

    np.fill_diagonal(kernel, 1.0)

    return kernel


# ----------------------------------------------------------------------------------------------
# the whole step, covariance in, kernel out
# ----------------------------------------------------------------------------------------------

def neuron_graph_from_covariance(
    covariance,
    firing_rate=None,
    neighbour_rank=10,
    kernel_type="self_tuning",
    decay=40.0,
    signed=True,
    keep_self_loops=False,
    threshold=0.0,
    anisotropy=0.0,
    minimum_variance=1e-10,
    minimum_firing_rate=1e-4,
):
    """
    Covariance (m x m) -> kernel over the live neurons.

    Returns a dict with:
        kernel          (m_live x m_live) edge weights
        live_indices    which original neuron indices the rows/columns of `kernel` refer to
        dead_fraction   fraction of neurons dropped as dead
        correlation     the correlation matrix of the live neurons
    """
    is_dead = find_dead_neurons(
        covariance,
        firing_rate=firing_rate,
        minimum_variance=minimum_variance,
        minimum_firing_rate=minimum_firing_rate,
    )

    live_indices = np.flatnonzero(~is_dead)

    live_covariance = covariance[np.ix_(live_indices, live_indices)]

    correlation = correlation_from_covariance(live_covariance)

    squared_distances = correlation_distance_squared(correlation, signed=signed)

    kernel = adaptive_gaussian_kernel(
        squared_distances,
        neighbour_rank=neighbour_rank,
        kernel_type=kernel_type,
        decay=decay,
        keep_self_loops=keep_self_loops,
        threshold=threshold,
    )

    kernel = apply_anisotropy(kernel, anisotropy)

    result = {
        "kernel": kernel,
        "live_indices": live_indices,
        "dead_fraction": float(is_dead.mean()),
        "correlation": correlation,
    }

    return result
