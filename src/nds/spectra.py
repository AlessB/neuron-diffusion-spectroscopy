"""
spectra.py
==========

The spectrum of the neuron graph: eigenvalues, eigenvectors, and the numbers we read off them.

Three matrices, one spectrum
----------------------------
From the kernel K (graphs.py) and the degrees d_i = sum_j K_ij, with D = diag(d):

    P     = D^-1 K                      random-walk (diffusion) operator, rows sum to 1
    A     = D^-1/2 K D^-1/2             its symmetric twin: same eigenvalues as P
    L_sym = I - A                       normalised graph Laplacian

If A v = lambda v then
    P psi = lambda psi    with  psi = D^-1/2 v        (right eigenvectors of P, "diffusion coordinates")
    L_sym v = mu v        with  mu = 1 - lambda       (Laplacian eigenvalues)

so one symmetric eigendecomposition (numpy/scipy eigh, exact and stable) gives everything.
Laplacian eigenvalues mu lie in [0, 2]; diffusion eigenvalues lambda = 1 - mu lie in [-1, 1].

What we read off
----------------
- module count k*:   the position of the largest gap among the smallest Laplacian eigenvalues
                     (eigengap heuristic; von Luxburg 2007, Sec. 8.3; made rigorous by
                     Peng, Sun & Zanetti 2017 and Lee, Oveis Gharan & Trevisan 2014)
- DSE:               diffusion spectral entropy (Liao et al.), the Shannon entropy of the
                     normalised |lambda|^t. About log2(k) for k well separated modules.
- IPR:               inverse participation ratio of an eigenvector, sum v_i^4 for unit v.
                     1/m for a vector spread evenly over m neurons, 1 for a single neuron.
                     Large IPR = the mode "localises" on a few neurons.
- localization:      IPR x m / 3, about 1 for a random-looking mode; subspace_localization for
                     a group of slow modes (robust to near-degenerate eigenvalues, but blind to a
                     partition that covers every neuron evenly); and aligned_localization, the
                     per-mode IPR after a varimax rotation of the group (sees both). THE readouts
                     that tell neuron-aligned structure from a rotated control.
- heat capacity:     C(tau) = tau^2 Var(mu) under p ~ exp(-tau mu); its peak gives a diffusion
                     scale of the structure (heat_capacity_curve). An eigenvalue readout.
- degree floor:      localisation of the trivial mode phi_0 = sqrt(d / sum d): 1/3 for equal
                     degrees, larger when a few hub neurons dominate. Read slow-mode
                     localisation next to it: hubs, not modules, if the two move together.
- degree check:      the degree of the neurons the slow modes sit on over the median degree
                     (Hata & Nakao 2017: eigenvectors can localise on unusual degrees).
- spacing ratio <r>: level statistics, mean of min/max of neighbouring spacings (Oganesyan & Huse
                     2007). 0.386 for independent levels, 0.531 for repelling (GOE) levels.
                     A rotation-blind number (eigenvalues only), reported as context.

Credit
------
- R. R. Coifman, S. Lafon (2006). "Diffusion maps." Applied and Computational Harmonic Analysis
  21(1):5-30. (P = D^-1 K, the symmetric conjugate, diffusion coordinates psi.)
- U. von Luxburg (2007). "A tutorial on spectral clustering." Statistics and Computing 17(4).
- D. Liao, C. Liu, B. W. Christensen, A. Tong, G. Huguet, G. Wolf, M. Nickel, I. Adelstein,
  S. Krishnaswamy. "Assessing neural network representations during training using
  noise-resilient diffusion spectral entropy." arXiv:2312.04823.
  Code: https://github.com/ChenLiu-1996/DiffusionSpectralEntropy (dse_dsmi/dse.py).
  Our `diffusion_spectral_entropy` follows their definition: absolute eigenvalues, raised to the
  power t, normalised to sum to 1, a machine-epsilon added, entropy in bits (log base 2).
  The difference: they build the kernel over SAMPLES (data points); we build it over NEURONS.
- H. F. Kaiser (1958). "The varimax criterion for analytic rotation in factor analysis."
  Psychometrika 23(3). (varimax_rotation)
- K. Rohe, M. Zeng (2023). "Vintage factor analysis with varimax performs statistical
  inference." Journal of the Royal Statistical Society B 85(4). (why varimax finds sparse
  structure in spectral embeddings)
- L. Zelnik-Manor, P. Perona (2004). "Self-tuning spectral clustering." NeurIPS 17. (aligning
  eigenvectors to count clusters; also our kernel, graphs.py)
- C. Davis, W. M. Kahan (1970). "The rotation of eigenvectors by a perturbation. III." SIAM
  Journal on Numerical Analysis 7(1). Y. Yu, T. Wang, R. J. Samworth (2015). "A useful variant of
  the Davis-Kahan theorem for statisticians." Biometrika 102(2), 315-323. (davis_kahan_ratio)
- P. O. Hoyer (2004). "Non-negative matrix factorization with sparseness constraints." JMLR 5.
  (the L1/L2 ratio behind support_ratio)
- V. Ozolins, R. Lai, R. Caflisch, S. Osher (2013). "Compressed modes for variational problems in
  mathematics and physics." UCLA CAM report 13-45. (L1-localised modes of Laplacians)
- N. Marzari, D. Vanderbilt (1997). "Maximally localized generalized Wannier functions for
  composite energy bands." Physical Review B 56, 12847. (choosing the most compact basis of a
  group of modes, the physics version of varimax_rotation)
- S. Hata, H. Nakao (2017). "Localization of Laplacian eigenvectors on random networks."
  Scientific Reports 7. doi:10.1038/s41598-017-01010-0 (degree localisation, the confound the
  degree check reads)
- V. Oganesyan, D. A. Huse (2007). "Localization of interacting fermions at high temperature."
  Physical Review B 75, 155111. (the spacing ratio)
- Y. Y. Atas, E. Bogomolny, O. Giraud, G. Roux (2013). "Distribution of the ratio of consecutive
  level spacings in random matrix ensembles." Physical Review Letters 110, 084101.
"""

import numpy as np
from scipy.linalg import eigh


def symmetric_diffusion_operator(kernel):
    """
    A = D^-1/2 K D^-1/2, plus the degree vector d.

    Raises if any neuron has zero degree (an isolated neuron). That should not happen once dead
    neurons are removed and every live neuron has neighbours in the kernel.
    """
    degrees = kernel.sum(axis=1)

    if np.any(degrees <= 0):
        number_isolated = int(np.sum(degrees <= 0))
        raise ValueError(f"{number_isolated} neurons have zero degree (isolated) in the kernel")

    inverse_sqrt_degrees = 1.0 / np.sqrt(degrees)

    symmetric_operator = kernel * np.outer(inverse_sqrt_degrees, inverse_sqrt_degrees)

    # exact symmetry, so eigh is happy even after rounding
    symmetric_operator = (symmetric_operator + symmetric_operator.T) / 2.0

    return symmetric_operator, degrees


def diffusion_spectrum(kernel, number_of_modes=None):
    """
    Eigen-decomposition of the neuron graph.

    Returns a dict, everything sorted from the slowest mode (lambda = 1, mu = 0) downwards:

        diffusion_eigenvalues   lambda_0 >= lambda_1 >= ...       eigenvalues of P
        laplacian_eigenvalues   mu_0 <= mu_1 <= ...               mu = 1 - lambda
        symmetric_vectors       columns v_k, unit length          eigenvectors of A and L_sym
        diffusion_coordinates   columns psi_k = D^-1/2 v_k        right eigenvectors of P
        degrees                 d

    number_of_modes: keep only the first this-many modes (None keeps all). For m = 2048 the full
    dense decomposition takes about a second, so we compute it all and slice.
    """
    symmetric_operator, degrees = symmetric_diffusion_operator(kernel)

    eigenvalues, eigenvectors = eigh(symmetric_operator)

    # eigh returns ascending lambda; we want descending (slowest mode first)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    if number_of_modes is not None:
        eigenvalues = eigenvalues[:number_of_modes]
        eigenvectors = eigenvectors[:, :number_of_modes]

    # Fix each eigenvector's sign so that its largest entry is positive. Eigenvectors are only
    # defined up to sign; fixing it makes plots and checkpoint-to-checkpoint comparisons stable.
    for mode in range(eigenvectors.shape[1]):
        largest = np.argmax(np.abs(eigenvectors[:, mode]))

        if eigenvectors[largest, mode] < 0:
            eigenvectors[:, mode] = -eigenvectors[:, mode]

    inverse_sqrt_degrees = 1.0 / np.sqrt(degrees)
    diffusion_coordinates = eigenvectors * inverse_sqrt_degrees[:, np.newaxis]

    result = {
        "diffusion_eigenvalues": eigenvalues,
        "laplacian_eigenvalues": 1.0 - eigenvalues,
        "symmetric_vectors": eigenvectors,
        "diffusion_coordinates": diffusion_coordinates,
        "degrees": degrees,
    }

    return result


# ----------------------------------------------------------------------------------------------
# numbers read off the spectrum
# ----------------------------------------------------------------------------------------------

def eigengap_module_count(laplacian_eigenvalues, maximum_modules=20, skip_trivial=True):
    """
    The eigengap heuristic: k* = the k that maximises  mu_k - mu_(k-1).

    Laplacian eigenvalues are sorted ascending, mu_0 = 0. If the graph has k well separated
    modules, mu_0 ... mu_(k-1) are all near 0 and mu_k is clearly larger, so the biggest jump
    sits just after the k-th eigenvalue.

    skip_trivial=True (default) searches k = 2..maximum_modules only. Why: on a dense neuron graph
    the slow modes sit just below the bulk (e.g. mu_1..mu_7 = 0.986, bulk = 0.995) and the first
    gap mu_1 - mu_0 is always the largest, which would always answer "1". This was found by the
    spectral-foundations research pass on a 2,048-node synthetic graph (17 Oct 2026) and is
    recorded in DECISIONS.md. The price: with skip_trivial the answer is never 1, so also look at
    `gap_ratio` (how much the winning gap stands out).

    Returns (k_star, gaps, gap_ratio):
        gaps[k-1] = mu_k - mu_(k-1)
        gap_ratio = winning gap / median of the searched gaps (about 1 = no real gap)

    Caveat: the heuristic always returns SOME number. Whether it means anything depends on
    gap_ratio and on the bootstrap band.
    """
    eigenvalues = np.sort(np.asarray(laplacian_eigenvalues))

    largest_k = min(maximum_modules, len(eigenvalues) - 1)

    gaps = np.diff(eigenvalues[: largest_k + 1])

    if skip_trivial and len(gaps) >= 2:
        searched = gaps[1:]
        k_star = int(np.argmax(searched)) + 2
    else:
        searched = gaps
        k_star = int(np.argmax(searched)) + 1

    median_gap = np.median(searched)

    if median_gap > 0:
        gap_ratio = float(np.max(searched) / median_gap)
    else:
        gap_ratio = float("inf")

    return k_star, gaps, gap_ratio


def localization_ratio(vector):
    """
    IPR x m / 3: how localised a unit vector over m neurons is, compared with a random vector.

    A random unit vector in m dimensions has expected IPR of about 3 / m (Gaussian entries:
    E[x^4] = 3 E[x^2]^2). So this ratio is about 1 for a "spread over everything" mode and grows
    as the mode concentrates: a mode spread evenly over s neurons gives m / (3 s).

    Why it matters (found 17 Oct 2026 on 2,048-neuron synthetic data, see docs/method-notes.md):
    rotating the neurons keeps the slow modes and the module count (shared signals become a smooth
    manifold over all neurons) but destroys localisation (ratio drops to about 1). So localisation
    is the readout that says "these specific neurons", which is the project's claim.
    """
    vector = np.asarray(vector, dtype=np.float64)

    return inverse_participation_ratio(vector) * len(vector) / 3.0


def subspace_localization(vectors):
    """
    How concentrated a SET of modes is on a subset of neurons, ignoring how the set is rotated
    internally.

    Why a set and not single modes: slow modes are often nearly degenerate (eigenvalues 0.970,
    0.971, 0.972 ...). Inside a degenerate group the individual eigenvectors are arbitrary mixtures,
    so the IPR of one of them can look spread out even when the group as a whole sits on a few
    modules. The group's "leverage" per neuron,

        l_i = sum over modes k of v_k(i)^2          (diagonal of the projector V V^T)

    does not depend on that arbitrary mixing. With p_i = l_i / sum(l):

        subspace localization = m * sum_i p_i^2

    It is 1 when the modes are spread evenly over all m neurons, m / s when they live on s neurons.
    For d random modes its expected value is about 1 + 2/d (a chi-squared fluctuation), which is
    the baseline the rotation control should reproduce.

    vectors: (neurons, modes) unit columns, e.g. symmetric_vectors[:, 1:d+1].
    """
    vectors = np.asarray(vectors, dtype=np.float64)

    number_of_neurons = vectors.shape[0]

    leverage = np.sum(vectors ** 2, axis=1)

    probabilities = leverage / np.sum(leverage)

    return float(number_of_neurons * np.sum(probabilities ** 2))


def varimax_rotation(vectors, maximum_iterations=500, tolerance=1e-10):
    """
    Rotate the columns of `vectors` (an orthonormal basis of a subspace) among themselves so that
    each column is as concentrated on few rows as possible: Kaiser's varimax (1958), the standard
    iteration through the SVD of V^T (V^3 - V diag(column sums of V^2) / m).

    Why: inside a group of nearly equal eigenvalues the eigenvectors are arbitrary mixtures. The
    subspace is well defined, its basis is not, and every basis-dependent number (IPR, "which
    neurons") is meaningless until a basis is chosen. Varimax chooses the sparsest one, so if the
    subspace is spanned by module indicators (each neuron in one module), the rotated columns are
    those indicators again. Rohe & Zeng (2023) show this recovers sparse structure from spectral
    embeddings in general; Zelnik-Manor & Perona (2004) use the same alignment idea to count
    clusters.

    Returns the rotated (neurons, modes) matrix, still orthonormal.
    """
    basis = np.asarray(vectors, dtype=np.float64)

    number_of_rows, number_of_columns = basis.shape

    if number_of_columns < 2:
        return basis.copy()

    rotation = np.eye(number_of_columns)

    previous_objective = 0.0

    for _ in range(maximum_iterations):
        rotated = basis @ rotation

        column_sums = np.sum(rotated ** 2, axis=0)

        gradient = basis.T @ (rotated ** 3 - rotated * column_sums[np.newaxis, :] / number_of_rows)

        left, singular_values, right = np.linalg.svd(gradient)

        rotation = left @ right

        objective = np.sum(singular_values)

        if objective - previous_objective < tolerance * max(objective, 1e-300):
            break

        previous_objective = objective

    return basis @ rotation


def support_ratio(vector):
    """
    How few neurons a unit vector really uses, measured with the L1 norm instead of the fourth
    power, relative to a random Gaussian vector:

        support_ratio = (2 / pi) * m * ||v||_2^2 / ||v||_1^2

    An indicator of s of m neurons gives 2 m / (pi s); a random Gaussian vector gives about 1
    (E|x| = sqrt(2 / pi) for unit-variance entries). Why not IPR: IPR x m / 3 of an indicator is
    m / (3 s), which is 1 (the random value) for a module holding a third of the neurons. Large
    modules that cover the layer are invisible to IPR and visible here (found 22 Oct 2026 on the
    three-module toy: IPR-aligned 1.00 original vs 0.91 rotated; support 1.87 vs 1.04).
    """
    vector = np.asarray(vector, dtype=np.float64)

    l1 = np.sum(np.abs(vector))

    if l1 <= 0:
        return float("nan")

    return float((2.0 / np.pi) * len(vector) * np.sum(vector ** 2) / l1 ** 2)


def aligned_support(vectors):
    """
    Mean support_ratio of the columns after varimax_rotation (the sparsest basis of the subspace).
    """
    rotated = varimax_rotation(vectors)

    ratios = []
    for mode in range(rotated.shape[1]):
        ratios.append(support_ratio(rotated[:, mode]))

    return float(np.mean(ratios))


def aligned_localization(vectors):
    """
    Localisation of a set of slow modes after choosing their sparsest basis (varimax_rotation):
    the mean over modes of localization_ratio (IPR x m / 3).

    This sees what subspace_localization cannot. If every neuron belongs to one of k equal modules,
    the leverage of the indicator subspace is the same for every neuron, so subspace_localization
    is exactly 1, the same as for random modes (found 22 Oct 2026 on a synthetic layer where 8
    modules of 32 neurons switch on: the count, RankMe and DSE all jumped, subspace localisation
    did not). After varimax the modes are the indicators again, with IPR 1/32 each, so this
    number is m / (3 s) = 2.7 for s = 32 of m = 256, and about 1 for random modes. Rotating the
    neurons (controls.py) destroys it: there is no sparse basis in a rotated subspace.

    vectors: (neurons, modes) unit columns, e.g. symmetric_vectors[:, 1:d+1].
    """
    rotated = varimax_rotation(vectors)

    ratios = []
    for mode in range(rotated.shape[1]):
        ratios.append(localization_ratio(rotated[:, mode]))

    return float(np.mean(ratios))


def diffusion_spectral_entropy(diffusion_eigenvalues, diffusion_time=1):
    """
    DSE (Liao et al.):  S(t) = - sum_i p_i log2 p_i,   p_i = |lambda_i|^t / sum_j |lambda_j|^t

    Follows dse_dsmi/dse.py: absolute values, power t, normalise, add machine epsilon, log base 2.

    Larger t "low-pass filters" the spectrum: only slow modes (|lambda| near 1) survive, so S(t)
    counts the number of well separated modules. For k perfectly separated modules and large t,
    S -> log2(k). Liao et al. suggest choosing t so that ~1% of |lambda|^t stay above 0.01;
    we scan several t instead (see configs/).
    """
    magnitudes = np.abs(np.asarray(diffusion_eigenvalues, dtype=np.float64))

    powered = magnitudes ** diffusion_time

    total = powered.sum()

    if total <= 0:
        return 0.0

    probabilities = powered / total

    probabilities = probabilities + np.finfo(float).eps

    entropy = - np.sum(probabilities * np.log2(probabilities))

    return float(entropy)


def inverse_participation_ratio(vector):
    """
    IPR = sum v_i^4 / (sum v_i^2)^2.

    1/m when the vector is spread evenly over m neurons; 1 when it sits on a single neuron.
    The effective number of neurons a mode lives on is 1 / IPR.
    """
    vector = np.asarray(vector, dtype=np.float64)

    squared = vector ** 2

    numerator = np.sum(squared ** 2)
    denominator = np.sum(squared) ** 2

    return float(numerator / denominator)


def spacing_ratio(eigenvalues, first_index=0, last_index=None):
    """
    Mean level-spacing ratio <r> of a stretch of sorted eigenvalues (Oganesyan & Huse 2007).

    With spacings s_n = lambda_{n+1} - lambda_n, each ratio is
        r_n = min(s_n, s_{n+1}) / max(s_n, s_{n+1}),
    a number in [0, 1] that needs no unfolding (no estimate of the local density of levels).
    Reference values (Atas, Bogomolny, Giraud, Roux 2013):
        independent levels (Poisson)        <r> = 2 ln 2 - 1 = 0.386
        repelling levels (GOE, real symmetric random matrix)  <r> = 0.531
    Levels that come in nearly degenerate pairs (many decoupled copies of the same module) push
    <r> below the Poisson value; strongly coupled, mixed levels push it towards the GOE value.

    first_index, last_index   the stretch of eigenvalues to use (python slice bounds)
    Returns nan when fewer than three levels are given.
    """
    values = np.sort(np.asarray(eigenvalues, dtype=np.float64))

    values = values[first_index:last_index]

    if len(values) < 3:
        return float("nan")

    spacings = np.diff(values)

    earlier = spacings[:-1]
    later = spacings[1:]

    smaller = np.minimum(earlier, later)
    larger = np.maximum(earlier, later)

    # two exactly degenerate spacings in a row give 0/0; leave those out
    usable = larger > 0

    if not np.any(usable):
        return float("nan")

    ratios = smaller[usable] / larger[usable]

    return float(np.mean(ratios))


def heat_capacity_curve(laplacian_eigenvalues, diffusion_scales=None):
    """
    Spectral entropy S(tau) and heat capacity C(tau) of the graph, treating the Laplacian
    eigenvalues as energy levels at inverse temperature tau.

        p_k(tau) = exp(-tau mu_k) / Z         over the NON-trivial eigenvalues (mu_0 = 0 dropped)
        S(tau)   = - sum p_k log p_k
        C(tau)   = tau^2 Var_p(mu) = - dS / d ln tau

    In words: at diffusion time tau only modes with mu below about 1/tau still carry weight. C(tau)
    is large where many modes drop out at once, so a peak at tau* says "there is structure at
    diffusion scale tau*", and the peak height says how sharp it is. Unlike DSE at t >= 2, which
    collapses onto the slowest few modes (method notes 4b), C(tau) has a scale.

    Why mu_0 is dropped: it is 0 for every connected graph, and with it included the curves of a
    graph with modules, without, and rotated were identical in our synthetic check (research2/R4,
    pitfall P15). Like every eigenvalue readout, the rotated control keeps much of the peak, so it is
    shown next to the rotated value (D33).

    diffusion_scales   the tau values; default 60 log-spaced values from 1/mu_max to 1000/mu_1
    Returns a dict with the arrays tau, entropy, heat_capacity and the peak (tau*, C*).

    Credit: the statistical-mechanics reading of a graph spectrum; used for networks by
    M. De Domenico, J. Biamonte (2016) "Spectral entropies as information-theoretic tools for
    complex network comparison", Phys. Rev. X 6, 041062, and by P. Villegas, A. Gabrielli,
    F. Santucci, G. Caldarelli, T. Gili (2022) "Laplacian paths in complex networks: information
    core emerges from entropic transitions", Phys. Rev. Research 4, 033196 (the heat-capacity
    peak as a scale). Hennick & Corlouer (2026) and
    Deng & Jin (2026) use it for networks inside trained models (research/B).
    """
    levels = np.sort(np.asarray(laplacian_eigenvalues, dtype=np.float64))[1:]

    levels = levels[levels > 1e-12]

    if len(levels) < 2:
        empty = {
            "tau": np.array([]),
            "entropy": np.array([]),
            "heat_capacity": np.array([]),
            "peak_scale": float("nan"),
            "peak_heat_capacity": float("nan"),
        }
        return empty

    if diffusion_scales is None:
        smallest = 1.0 / levels.max()
        largest = 1000.0 / levels.min()
        diffusion_scales = np.logspace(np.log10(smallest), np.log10(largest), 60)

    entropies = []
    capacities = []

    for tau in diffusion_scales:
        # shift by the smallest level before exponentiating: p does not change, nothing underflows
        weights = np.exp(- tau * (levels - levels[0]))

        probabilities = weights / weights.sum()

        positive = probabilities[probabilities > 0]
        entropy = - np.sum(positive * np.log(positive))

        mean_level = np.sum(probabilities * levels)
        variance = np.sum(probabilities * (levels - mean_level) ** 2)

        entropies.append(entropy)
        capacities.append(tau ** 2 * variance)

    capacities = np.asarray(capacities)

    peak = int(np.argmax(capacities))

    result = {
        "tau": np.asarray(diffusion_scales, dtype=np.float64),
        "entropy": np.asarray(entropies),
        "heat_capacity": capacities,
        "peak_scale": float(diffusion_scales[peak]),
        "peak_heat_capacity": float(capacities[peak]),
    }

    return result


def operator_norm(matrix):
    """
    Largest absolute eigenvalue of a symmetric matrix (its spectral norm), by Lanczos iteration
    (scipy eigsh, one eigenvalue): a few dozen matrix-vector products instead of a full
    eigendecomposition.
    """
    from scipy.sparse.linalg import eigsh

    symmetric = np.asarray(matrix, dtype=np.float64)

    if symmetric.shape[0] < 3:
        return float(np.max(np.abs(np.linalg.eigvalsh(symmetric))))

    value = eigsh(symmetric, k=1, which="LM", return_eigenvectors=False, tol=1e-6)

    return float(np.abs(value[0]))


def leading_eigenvectors(operator, count):
    """
    The eigenvectors of the `count` largest eigenvalues of a symmetric operator, as columns
    (Lanczos for large matrices, a full eigendecomposition for small ones). For the symmetric
    diffusion operator these span the module subspace, modes 0 .. count-1.
    """
    from scipy.sparse.linalg import eigsh

    symmetric = np.asarray(operator, dtype=np.float64)
    size = symmetric.shape[0]
    count = int(count)

    if size <= 4 * count + 10:
        _, vectors = np.linalg.eigh(symmetric)
        return vectors[:, ::-1][:, :count]

    _, vectors = eigsh(symmetric, k=count, which="LA", tol=1e-8)

    return vectors


def chordal_subspace_distance(vectors_a, vectors_b):
    """
    1 - ||A^T B||_F^2 / k for two (neurons, k) blocks on the same neurons: the mean sin^2 of the
    principal angles (the same number as timeline.subspace_distance, kept here so analysis.py does
    not import timeline.py).
    """
    a, _ = np.linalg.qr(np.asarray(vectors_a, dtype=np.float64))
    b, _ = np.linalg.qr(np.asarray(vectors_b, dtype=np.float64))

    overlap = a.T @ b

    number_of_modes = min(a.shape[1], b.shape[1])

    return float(1.0 - np.sum(overlap ** 2) / number_of_modes)


def crossvalidated_projector_distance(from_one, to_one, from_two, to_two):
    """
    The cross-validated ("crossnobis") distance between two subspaces, from two independent halves
    of the data, and its per-neuron parts:

        d_cv = trace[(P_from^(1) - P_to^(1)) (P_from^(2) - P_to^(2))] / (2k)
        c_i  = [(P_from^(1) - P_to^(1)) (P_from^(2) - P_to^(2))]_ii / (2k)        (sum_i c_i = d_cv)

    P = U U^T for each (neurons, k) basis; all four share the same rows (neurons) and k. The plain
    distance ||P_from - P_to||^2 / (2k) is a sum of squares, biased upwards by noise; here the noise
    of half (1) and half (2) is independent, so it averages out of the product and d_cv is close to
    unbiased (it can be negative when nothing changed). Borrowed from neuroimaging (cross-validated
    Mahalanobis distance: Walther et al. 2016; Diedrichsen et al.); the same inner-product trick is
    used for change points in network sequences (Wang, Yu & Rinaldo 2021) and in debiased CKA.
    Checked in research/round5 (S9 and the D50 review): 0 within noise when nothing changed, the
    true distance when one neuron of 192 changed module, independent of the number of tokens.

    Computed without forming neurons x neurons matrices:
        <P_U - P_V, P_X - P_Y> = ||U^T X||^2 - ||U^T Y||^2 - ||V^T X||^2 + ||V^T Y||^2
    Returns (d_cv, c) with c of length neurons.
    """
    bases = []
    for basis in [from_one, to_one, from_two, to_two]:
        orthonormal, _ = np.linalg.qr(np.asarray(basis, dtype=np.float64))
        bases.append(orthonormal)
    u, v, x, y = bases

    count = u.shape[1]

    def diagonal_of_product(left, right):
        # diag(left left^T right right^T), row by row
        return np.sum((left @ (left.T @ right)) * right, axis=1)

    per_neuron = diagonal_of_product(u, x) - diagonal_of_product(u, y) - diagonal_of_product(v, x) + diagonal_of_product(v, y)
    per_neuron = per_neuron / (2.0 * count)

    return float(np.sum(per_neuron)), per_neuron


def davis_kahan_ratio(perturbation_norms, laplacian_eigenvalues, module_count):
    """
    How well the data pin down the module subspace (modes 0 .. k*-1):

        r = mean ||A_replicate - A||_2 / (mu_k* - mu_k*-1)

    the typical size of the bootstrap perturbation of the symmetric diffusion operator over the
    eigengap that ends the module subspace. The subspace moves by an angle theta with
    sin(theta) <= 2 ||E|| / gap (Davis & Kahan 1970; Yu, Wang & Samworth 2015), so r well below
    0.5 means the count, the varimax basis, the localisation and the lineage are all pinned down by
    the probe text; r near 1 means they are not, whatever the readouts say. On synthetic layers
    (research/round4/sim3) r = 0.15 gave a localisation spread of 0.02 across replicates, r = 0.9 a
    spread of 0.36. The bound itself is loose (13 to 28 times the actual distance on those layers); r is a ranking, not
    a guarantee.
    """
    norms = np.asarray(perturbation_norms, dtype=np.float64)
    norms = norms[np.isfinite(norms)]

    values = np.asarray(laplacian_eigenvalues, dtype=np.float64)
    count = max(int(module_count), 2)

    if len(norms) == 0 or count >= len(values):
        return float("nan")

    gap = values[count] - values[count - 1]

    if gap <= 0:
        return float("inf")

    return float(np.mean(norms) / gap)


def spectrum_summary(
    kernel,
    number_of_modes=50,
    maximum_modules=20,
    diffusion_times=(1, 2, 4, 8, 16, 32, 64),
    ipr_modes=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
):
    """
    Everything we record for one (checkpoint, layer) neuron graph, in one dict.

    DSE is computed from the FULL spectrum (all modes), as in Liao et al.; the eigenvectors and
    IPR only from the first `number_of_modes`.
    """
    full = diffusion_spectrum(kernel, number_of_modes=None)

    k_star, gaps, gap_ratio = eigengap_module_count(
        full["laplacian_eigenvalues"],
        maximum_modules=maximum_modules,
    )

    entropies = {}
    for diffusion_time in diffusion_times:
        entropies[int(diffusion_time)] = diffusion_spectral_entropy(
            full["diffusion_eigenvalues"],
            diffusion_time=diffusion_time,
        )

    iprs = {}
    localizations = {}
    for mode in ipr_modes:
        if mode < full["symmetric_vectors"].shape[1]:
            vector = full["symmetric_vectors"][:, mode]
            iprs[int(mode)] = inverse_participation_ratio(vector)
            localizations[int(mode)] = localization_ratio(vector)

    if len(localizations) > 0:
        median_localization = float(np.median(list(localizations.values())))
    else:
        median_localization = float("nan")

    # the slow modes as a group: the k*-1 non-trivial modes below the eigengap, and a fixed 10
    slow_mode_count = max(k_star - 1, 1)
    slow_vectors = full["symmetric_vectors"][:, 1:slow_mode_count + 1]
    first_ten_vectors = full["symmetric_vectors"][:, 1:11]

    slow_subspace_localization = subspace_localization(slow_vectors)
    first_ten_subspace_localization = subspace_localization(first_ten_vectors)

    # the same two sets after choosing their sparsest basis (sees complete partitions too)
    slow_aligned_localization = aligned_localization(slow_vectors)
    first_ten_aligned_localization = aligned_localization(first_ten_vectors)

    # the module subspace itself: modes 0 .. k*-1, INCLUDING the trivial mode. k modules span the
    # same space as their k indicator vectors, and that span contains the near-constant mode 0, so
    # leaving mode 0 out (as the two sets above do) means varimax cannot rebuild the indicators.
    # On a planted run (tests/test_planted_events.py) this was the only localisation that was both
    # exact (m / (3 s)) and steady from checkpoint to checkpoint; the "first ten" sets wander
    # because bulk noise modes enter them whenever k* < 10.
    module_vectors = full["symmetric_vectors"][:, 0:max(k_star, 2)]
    # one varimax, two read-outs (IPR and L1 support) of the same rotated basis
    module_basis = varimax_rotation(module_vectors)
    module_ipr_ratios = []
    module_support_ratios = []
    for mode in range(module_basis.shape[1]):
        module_ipr_ratios.append(localization_ratio(module_basis[:, mode]))
        module_support_ratios.append(support_ratio(module_basis[:, mode]))
    module_aligned_localization = float(np.mean(module_ipr_ratios))
    module_aligned_support = float(np.mean(module_support_ratios))

    # level statistics: the slow end (modes 1 to 50) and the bulk (drop 10% at each edge)
    number_of_levels = len(full["laplacian_eigenvalues"])
    edge = number_of_levels // 10

    slow_spacing_ratio = spacing_ratio(full["laplacian_eigenvalues"], 1, 51)
    bulk_spacing_ratio = spacing_ratio(full["laplacian_eigenvalues"], edge, number_of_levels - edge)

    heat = heat_capacity_curve(full["laplacian_eigenvalues"])

    # degree floor (research2/R4 pitfall P5): the trivial mode is sqrt(d / sum d), so its
    # localisation measures only how uneven the degrees are. Equal degrees give exactly 1/3 (a
    # perfectly even vector); a few hub neurons push it up. If it climbs towards the slow-mode
    # localisation, the slow modes may be showing hubs, not modules. Reported in the table.
    degree_floor = localization_ratio(full["symmetric_vectors"][:, 0])

    # degree check (Hata & Nakao 2017): Laplacian eigenvectors can localise on neurons of unusual
    # degree, which is not a module. For each of modes 1-10, the degree of the neurons it sits on,
    # sum_i v_i^2 d_i, divided by the median degree; about 1 means "ordinary neurons"
    degrees = full["degrees"]
    median_degree = float(np.median(degrees))
    slow_degree_ratios = []
    for mode in range(1, min(11, full["symmetric_vectors"].shape[1])):
        weights = full["symmetric_vectors"][:, mode] ** 2
        slow_degree_ratios.append(float(np.sum(weights * degrees) / median_degree))
    slow_mode_degree_ratio = float(np.median(slow_degree_ratios))

    # the localisation of EVERY mode, slowest first (research2/R2: the IPR profile along the
    # spectrum; modules localise modes all along it, a rotation flattens the whole profile)
    vectors = full["symmetric_vectors"]
    all_mode_localization = (vectors ** 4).sum(axis=0) / (vectors ** 2).sum(axis=0) ** 2 * vectors.shape[0] / 3.0

    summary = {
        "laplacian_eigenvalues": full["laplacian_eigenvalues"][:number_of_modes],
        "symmetric_vectors": full["symmetric_vectors"][:, :number_of_modes],
        "module_count": k_star,
        "gap_ratio": gap_ratio,
        "eigengaps": gaps,
        "dse": entropies,
        "ipr": iprs,
        "localization": localizations,
        "median_localization": median_localization,
        "slow_subspace_localization": slow_subspace_localization,
        "random_subspace_baseline": 1.0 + 2.0 / slow_mode_count,
        "first_ten_subspace_localization": first_ten_subspace_localization,
        "slow_aligned_localization": slow_aligned_localization,
        "first_ten_aligned_localization": first_ten_aligned_localization,
        "module_aligned_localization": module_aligned_localization,
        "module_aligned_support": module_aligned_support,
        "slow_spacing_ratio": slow_spacing_ratio,
        "bulk_spacing_ratio": bulk_spacing_ratio,
        "heat_capacity_peak": heat["peak_heat_capacity"],
        "heat_capacity_peak_scale": heat["peak_scale"],
        "degree_floor_localization": degree_floor,
        "all_laplacian_eigenvalues": full["laplacian_eigenvalues"],
        "all_mode_localization": all_mode_localization,
        "slow_mode_degree_ratio": slow_mode_degree_ratio,
        "median_localization_whole_spectrum": float(np.median(all_mode_localization[1:])),
        "number_of_neurons": int(kernel.shape[0]),
    }

    return summary
