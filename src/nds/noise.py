"""
noise.py
========

How much of what we see could be noise from a finite probe text. Pure numpy.

Three tools:

1. Effective number of tokens. Neighbouring tokens in a document give similar activations, so
   T tokens carry less information than T independent samples. If a neuron's activations follow
   an AR(1) picture with lag-one correlation rho, the variance of a sample correlation between two
   such neurons is about (1 / T) (1 + rho^2) / (1 - rho^2) (Bartlett 1946), so

       T_eff = T (1 - rho^2) / (1 + rho^2)

   We measure rho per neuron during collection (activations.collect_neuron_moments) and use the
   median.

2. Marchenko-Pastur edges. For m neurons and T_eff independent tokens with no structure at all,
   the eigenvalues of the sample correlation matrix fill [(1 - sqrt q)^2, (1 + sqrt q)^2] with
   q = m / T_eff (Marchenko and Pastur 1967). A module only shows up as a separate eigenvalue if its
   population eigenvalue is above 1 + sqrt q (Baik, Ben Arous and Peche 2005).

3. A null covariance (DECISIONS D36). The "shuffle null" asks what our readouts look like when
   every neuron is independent of every other, at the same noise level as the real data. Shuffling
   262,144 tokens x 2,048 neurons is expensive; drawing the matching Wishart matrix directly is the
   same thing in distribution and costs one m x m matrix product. We use the Bartlett decomposition
   W = A A^T, A lower triangular with A_ii ~ sqrt(chi^2_{dof - i}) and A_ij ~ N(0, 1) below the
   diagonal (Bartlett 1933; e.g. Anderson, "An Introduction to Multivariate Statistical Analysis").
   Dividing W by its diagonal gives the null correlation matrix. The token-sample shuffle in
   token_sample.py is the literal version of the same null on fewer tokens.

Credit
------
- M. S. Bartlett (1946). "On the theoretical specification and sampling properties of
  autocorrelated time-series." Supplement to the Journal of the Royal Statistical Society 8(1).
- V. A. Marchenko, L. A. Pastur (1967). "Distribution of eigenvalues for some sets of random
  matrices." Mathematics of the USSR-Sbornik 1(4).
- J. Baik, G. Ben Arous, S. Peche (2005). "Phase transition of the largest eigenvalue for nonnull
  complex sample covariance matrices." Annals of Probability 33(5).
- M. S. Bartlett (1933). "On the theory of statistical regression." Proc. Royal Society of
  Edinburgh 53. (the triangular decomposition of a Wishart matrix)
"""

import numpy as np


def effective_token_count(number_of_tokens, lag_one_correlation):
    """
    T_eff = T (1 - rho^2) / (1 + rho^2).  rho = 0 gives T; rho near 1 gives almost nothing.
    """
    rho_squared = float(lag_one_correlation) ** 2

    factor = (1.0 - rho_squared) / (1.0 + rho_squared)

    return float(number_of_tokens) * max(factor, 0.0)


def lag_one_correlations(token_activations_by_sequence):
    """
    Lag-one autocorrelation of every neuron, from activations shaped (sequences, positions, neurons).
    Used by the tests and by anyone holding raw activations; collection accumulates the same sums
    batch by batch on the GPU.
    """
    values = np.asarray(token_activations_by_sequence, dtype=np.float64)

    flat = values.reshape(-1, values.shape[-1])

    mean = flat.mean(axis=0)

    centred = values - mean

    variance = (centred ** 2).mean(axis=(0, 1))

    neighbour_products = centred[:, 1:, :] * centred[:, :-1, :]

    lag_covariance = neighbour_products.mean(axis=(0, 1))

    safe_variance = np.maximum(variance, 1e-30)

    return lag_covariance / safe_variance


def marchenko_pastur_edges(number_of_neurons, effective_tokens):
    """
    The interval the eigenvalues of a pure-noise correlation matrix fill.
    """
    q = float(number_of_neurons) / float(effective_tokens)

    root_q = np.sqrt(q)

    lower = (1.0 - root_q) ** 2
    upper = (1.0 + root_q) ** 2

    return lower, upper


def wishart_null_correlation(number_of_neurons, degrees_of_freedom, random_generator):
    """
    A correlation matrix of `number_of_neurons` independent neurons estimated from
    `degrees_of_freedom` independent tokens (use the effective token count), drawn directly.

    Returns an (m, m) correlation matrix with unit diagonal.
    """
    m = int(number_of_neurons)
    dof = float(degrees_of_freedom)

    if dof <= m:
        raise ValueError("the null needs more effective tokens than neurons")

    triangle = np.zeros((m, m))

    for row in range(m):
        triangle[row, row] = np.sqrt(random_generator.chisquare(dof - row))

        triangle[row, :row] = random_generator.standard_normal(row)

    scatter = triangle @ triangle.T

    scale = np.sqrt(np.diag(scatter))

    correlation = scatter / np.outer(scale, scale)

    return correlation
