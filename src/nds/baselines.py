"""
baselines.py
============

The covariance-based measures our neuron-graph spectrum is compared against, on every figure.

All of them are functions of the eigenvalues of the covariance matrix of the representations.
That is exactly why they share a blind spot: rotating the neurons (mixing them with an orthogonal
matrix Q) changes the covariance to Q^T C Q, which has the SAME eigenvalues. So RankMe, alpha-ReQ
and matrix entropy cannot say which neurons reorganise. The neuron graph can (see controls.py).

We compute them from the same neuron covariance we build the graph from, so both sides of every
comparison see identical data.

RankMe: two published versions
-------------------------------
- Garrido et al. (2023) define RankMe on the SINGULAR VALUES sigma_k of the representation matrix:
      p_k = sigma_k / sum(sigma) + eps,   RankMe = exp( - sum p_k log p_k )
- Li et al. (2025) code (utils/powerlaw.py, function `rankme`) applies the same formula to the
  covariance EIGENVALUES (explained-variance ratios), i.e. to sigma_k^2, with eps = 1e-7.
These give different numbers. We compute both and name them clearly; the Li et al. version is the
one to use when comparing with their published curves.

alpha-ReQ
---------
Agrawal et al. (2022): the eigenvalues of the covariance decay like a power law lambda_k ~ k^-alpha;
alpha is the decay exponent. Li et al. (2025) fit it with the function they credit to
Stringer & Pachitariu (2019): a least-squares line in log-log space over a fixed range of ranks,
weighted by 1/rank. Their call uses eigenvalue indices 11..99 (ranks 12..100). We reproduce that
fit exactly so the numbers are comparable.

Matrix entropy and Skean's prompt entropy (corrected 17 Oct 2026, see research/S2 and D18)
------------------------------------------------------------------------------------------
`matrix_entropy` is the Shannon entropy of the covariance eigenvalues / trace. At order 1 that is
exactly log(RankMe-Li) up to RankMe's epsilon, so it duplicates RankMe; it stays as a sanity number.

Skean et al. (2025) measure something different ("prompt entropy"): for EACH prompt, centre its
tokens, scale each token row to unit length, build the token Gram matrix, set its negative entries
to 0, divide by the trace, take the von Neumann entropy, divide by min(log N, log D), and average
over prompts. It needs the activations of each prompt, so it is computed during collection
(`activations.collect_neuron_moments`), with `prompt_entropy` below doing the maths. Skean et al.
apply it to the residual stream; we apply the same recipe to the MLP neurons.

Knowledge entropy: a neuron-basis baseline that is not a graph (added 22 Oct 2026)
-----------------------------------------------------------------------------------
Kim et al. (2025): average |activation| of every neuron over all tokens, normalise over the m
neurons to a distribution p_i, report H = - sum p_i log p_i (we divide by log m so layers
compare). Unlike RankMe it DOES change when the neurons are rotated, because it looks at each
neuron's mass. It answers a reviewer's obvious question: is high localisation of the slow modes
just "a few neurons carry most of the activation"? If localisation moves while knowledge entropy
does not (or the other way round), they measure different things.
The full-probe value uses sums of |x| accumulated during collection; the rotated control needs
real activations, so it is computed on the token sample (knowledge_entropy_of_sample).

Credit
------
- J. Kim, H. Lee, H. Cho, J. Jang, H. Hwang, S. Won, Y. Ahn, D. Lee, M. Seo (2025). "Knowledge
  entropy decay during language model pretraining hinders new knowledge acquisition." ICLR 2025.
  Code: https://github.com/kaistAI/Knowledge-Entropy (analysis/entropy.py; formula re-implemented).
- C. H. Martin, M. W. Mahoney (2021). "Implicit self-regularization in deep neural networks:
  evidence from random matrix theory and implications for learning." JMLR 22(165). Code
  (Apache-2.0): https://github.com/CalculatedContent/WeightWatcher (WW_powerlaw.py; the x_min
  scan re-implemented, without its small-sample correction).
- A. Clauset, C. R. Shalizi, M. E. J. Newman (2009). "Power-law distributions in empirical data."
  SIAM Review 51(4).
- Q. Garrido, R. Balestriero, L. Najman, Y. LeCun (2023). "RankMe: Assessing the downstream
  performance of pretrained self-supervised representations by their rank." ICML 2023.
- K. K. Agrawal, A. K. Mondal, A. Ghosh, B. A. Richards (2022). "alpha-ReQ: Assessing representation
  quality in self-supervised learning by measuring eigenspectrum decay." NeurIPS 2022.
- M. Z. Li, K. K. Agrawal, A. Ghosh, K. K. Teru, A. Santoro, G. Lajoie, B. A. Richards (2025).
  "Tracing the representation geometry of language models from pretraining to post-training."
  NeurIPS 2025. Code: https://github.com/melodylizx/Tracing-the-representation-geometry-of-language-models
  (utils/powerlaw.py: `rankme`, `stringer_get_powerlaw`).
- C. Stringer, M. Pachitariu, N. Steinmetz, M. Carandini, K. D. Harris (2019). "High-dimensional
  geometry of population responses in visual cortex." Nature 571. (the weighted power-law fit)
- O. Skean, M. R. Arefin, D. Zhao, N. Patel, J. Naghiyev, Y. LeCun, R. Shwartz-Ziv (2025).
  "Layer by layer: Uncovering hidden representations in language models." ICML 2025.
  Code: https://github.com/OFSkean/information_flow (no license file; recipe re-implemented from
  experiments/utils/metrics/metric_functions.py, whose `normalize` they credit to
  https://github.com/waltonfuture/Matrix-Entropy), entropy from
  https://github.com/uk-cliplab/representation-itl (repitl, `vonNeumannEntropy`).
"""

import numpy as np
from scipy.linalg import eigvalsh


def covariance_eigenvalues(covariance):
    """
    Eigenvalues of a covariance matrix, sorted from largest to smallest, negatives (rounding) set to 0.
    """
    eigenvalues = eigvalsh(covariance)

    eigenvalues = np.sort(eigenvalues)[::-1]

    eigenvalues = np.maximum(eigenvalues, 0.0)

    return eigenvalues


def rankme_from_singular_values(singular_values, epsilon=1e-12):
    """
    Garrido et al. (2023) RankMe, on singular values of the representation matrix.
    """
    values = np.abs(np.asarray(singular_values, dtype=np.float64))

    probabilities = values / values.sum() + epsilon

    entropy = - np.sum(probabilities * np.log(probabilities))

    return float(np.exp(entropy))


def rankme_garrido(covariance_eigenvalues_sorted):
    """
    Garrido et al. (2023) RankMe, from covariance eigenvalues.

    The singular values of the centred representation matrix are sqrt((n - 1) * lambda_k).
    The factor (n - 1) cancels in the normalisation, so sqrt(lambda_k) is enough.
    """
    singular_values = np.sqrt(np.maximum(covariance_eigenvalues_sorted, 0.0))

    return rankme_from_singular_values(singular_values)


def rankme_li(covariance_eigenvalues_sorted, epsilon=1e-7):
    """
    RankMe exactly as in Li et al. (2025) code (utils/powerlaw.py, `rankme`):
    the RankMe formula applied to covariance eigenvalues, with eps = 1e-7.
    """
    values = np.asarray(covariance_eigenvalues_sorted, dtype=np.float64)

    l1_norm = np.sum(np.abs(values))

    scores = values / l1_norm + epsilon

    entropy = - np.sum(scores * np.log(scores))

    return float(np.exp(entropy))


def alpha_req(
    covariance_eigenvalues_sorted,
    first_index=11,
    last_index=99,
):
    """
    Power-law decay exponent alpha of the eigenspectrum, fitted as in Li et al. (2025),
    who use Stringer & Pachitariu's weighted log-log least squares.

    Indices are 0-based positions in the sorted eigenvalue list, so the default 11..99 means
    ranks 12..100, matching their call `stringer_get_powerlaw(eigen, np.arange(11, 100))`.

    Their eigenvalues are explained-variance ratios (eigenvalue / sum). Dividing by the sum only
    shifts log(lambda) by a constant, which does not change the slope, so alpha is the same.

    Returns (alpha, r_squared) where r_squared is computed on the fitted range.
    """
    eigenvalues = np.asarray(covariance_eigenvalues_sorted, dtype=np.float64)

    if last_index >= len(eigenvalues):
        last_index = len(eigenvalues) - 1

    indices = np.arange(first_index, last_index + 1)

    ranks = indices + 1

    log_eigenvalues = np.log(np.abs(eigenvalues[indices]) + 1e-300)

    # design matrix: columns (-log rank, 1), so the first coefficient is alpha
    design = np.column_stack(
        [
            - np.log(ranks),
            np.ones(len(ranks)),
        ]
    )

    weights = 1.0 / ranks

    weighted_design = design * weights[:, np.newaxis]

    left_side = design.T @ weighted_design
    right_side = weighted_design.T @ log_eigenvalues

    coefficients = np.linalg.solve(left_side, right_side)

    alpha = float(coefficients[0])

    predicted = design @ coefficients
    residual_sum = np.sum((log_eigenvalues - predicted) ** 2)
    total_sum = np.sum((log_eigenvalues - log_eigenvalues.mean()) ** 2)
    r_squared = float(1.0 - residual_sum / total_sum)

    return alpha, r_squared


def matrix_entropy(covariance_eigenvalues_sorted, order=1.0):
    """
    Renyi entropy of the normalised covariance spectrum, natural log.

        p_k = lambda_k / sum(lambda)
        order 1:   - sum p_k log p_k
        order a:   log( sum p_k^a ) / (1 - a)

    Careful: at order 1 this is log(RankMe) up to RankMe's small epsilon (checked: they differ by
    3e-4 to 2.5e-3 on synthetic neurons, research/checks/S2_results.txt), so it is NOT an
    independent baseline and it is NOT Skean et al.'s "prompt entropy". For theirs use
    `prompt_entropy` below. Kept because it is the natural "spectral entropy" of the covariance.
    """
    values = np.asarray(covariance_eigenvalues_sorted, dtype=np.float64)

    values = values[values > 0]

    probabilities = values / values.sum()

    if order == 1.0:
        entropy = - np.sum(probabilities * np.log(probabilities))
    else:
        entropy = np.log(np.sum(probabilities ** order)) / (1.0 - order)

    return float(entropy)


def prompt_entropy_from_gram(gram, order=1.0, clamp_negative=True):
    """
    Skean et al. (2025) prompt entropy from one prompt's token Gram matrix.

    gram             (tokens, tokens) Gram matrix of the centred, row-normalised token activations
                     (`normalised_token_rows` below, then rows @ rows.T)
    clamp_negative   True = their code: negative entries of the Gram matrix are set to 0
                     (element by element, not on eigenvalues) before the trace normalisation

    Returns the raw entropy in nats. (The division by min(log N, log D) needs the number of
    neurons, so `prompt_entropy` does it.)

    Their code (OFSkean/information_flow, experiments/utils/metrics/metric_functions.py l.191-207)
    builds the token Gram when tokens < neurons and the neuron Gram otherwise, and clamps whichever
    it built. We always use the token Gram: identical to theirs in our setting (255 or 256 tokens
    per prompt, 2,048 or more neurons). Re-implemented, not copied: that repo has no license.
    """
    gram = np.array(gram, dtype=np.float64)

    if clamp_negative:
        gram = np.maximum(gram, 0.0)

    trace = np.trace(gram)

    if trace <= 0:
        return float("nan")

    gram = gram / trace

    eigenvalues = np.linalg.eigvalsh(gram)

    eigenvalues = eigenvalues[eigenvalues > 0]

    probabilities = eigenvalues / eigenvalues.sum()

    if order == 1.0:
        entropy = - np.sum(probabilities * np.log(probabilities))
    else:
        entropy = np.log(np.sum(probabilities ** order)) / (1.0 - order)

    return float(entropy)


def normalised_token_rows(token_activations):
    """
    Skean et al.'s `normalize`: subtract the prompt's mean token, then scale every token row to
    unit length. token_activations: (tokens, neurons) for ONE prompt.
    """
    rows = np.array(token_activations, dtype=np.float64)

    rows = rows - rows.mean(axis=0)

    lengths = np.linalg.norm(rows, axis=1, keepdims=True)

    lengths = np.maximum(lengths, 1e-12)

    return rows / lengths


def prompt_entropy(token_activations, order=1.0, clamp_negative=True):
    """
    Skean et al. (2025) "prompt entropy" for one prompt, raw and normalised.

    Like RankMe it does not change when the neurons are rotated (it only uses token-token dot
    products), so it belongs with the covariance baselines. Unlike our `matrix_entropy` it is
    computed per prompt (tokens within one document) and then averaged over prompts by the caller
    (activations.collect_neuron_moments does that on the GPU side).

    Returns {"raw": nats, "normalised": raw / min(log tokens, log neurons)}.
    """
    rows = normalised_token_rows(token_activations)

    gram = rows @ rows.T

    raw = prompt_entropy_from_gram(gram, order=order, clamp_negative=clamp_negative)

    number_of_tokens = rows.shape[0]
    number_of_neurons = rows.shape[1]

    largest_possible = min(np.log(number_of_tokens), np.log(number_of_neurons))

    result = {
        "raw": raw,
        "normalised": raw / largest_possible,
    }

    return result


def knowledge_entropy(mean_absolute_activation):
    """
    Kim et al.'s knowledge entropy of one layer, divided by log m (1 = every neuron carries the
    same mass, 0 = one neuron carries everything).

    mean_absolute_activation   (m,) the mean of |x| per neuron over the probe tokens
    Neurons with zero mass are kept in m (they are part of the layer) but add nothing to the sum.
    """
    mass = np.asarray(mean_absolute_activation, dtype=np.float64)

    total = mass.sum()

    if total <= 0:
        return float("nan")

    probabilities = mass / total

    positive = probabilities[probabilities > 0]

    entropy = - np.sum(positive * np.log(positive))

    return float(entropy / np.log(len(mass)))


def knowledge_entropy_of_sample(token_sample, number_of_rotations=0, random_generator=None):
    """
    Knowledge entropy from a (tokens, neurons) sample of activations, and the same after mixing
    the neurons with random rotations (the control: rotation changes this number, unlike RankMe).

    Returns {"original": value, "rotated": [values]}.
    """
    sample = np.asarray(token_sample, dtype=np.float64)

    original = knowledge_entropy(np.abs(sample).mean(axis=0))

    rotated_values = []

    if number_of_rotations > 0:
        from nds.controls import random_orthogonal_matrix

        for _ in range(number_of_rotations):
            orthogonal = random_orthogonal_matrix(sample.shape[1], random_generator)

            rotated_sample = sample @ orthogonal

            rotated_values.append(knowledge_entropy(np.abs(rotated_sample).mean(axis=0)))

    result = {
        "original": original,
        "rotated": rotated_values,
    }

    return result


def htsr_alpha(weight_matrix, minimum_tail=20):
    """
    Heavy-tailed self-regularisation exponent of a weight matrix (Martin & Mahoney 2021), the
    number WeightWatcher reports: fit a power law rho(lambda) ~ lambda^-alpha to the tail of the
    eigenvalues of W^T W, choosing the start of the tail (x_min) that minimises the
    Kolmogorov-Smirnov distance, with the maximum-likelihood alpha for each candidate
    (Clauset, Shalizi & Newman 2009).

    A weight-side baseline: rotating the neurons (W -> Q^T W) keeps the singular values, so alpha
    cannot see which neurons reorganise, like RankMe. If ranked eigenvalues decay like k^-a
    (alpha-ReQ = a), the density decays like lambda^-(1 + 1/a), so alpha_HTSR = 1 + 1/a
    (research2/R2, checked numerically there).

    weight_matrix   e.g. mlp.dense_h_to_4h.weight (2048 x 512) or mlp.dense_4h_to_h.weight
    Returns (alpha, x_min, ks_distance). Only 512 eigenvalues for pythia-70m, so the fit is noisy:
    report the KS distance with it.
    """
    singular_values = np.linalg.svd(np.asarray(weight_matrix, dtype=np.float64), compute_uv=False)

    eigenvalues = np.sort(singular_values ** 2)
    eigenvalues = eigenvalues[eigenvalues > 0]

    best_distance = np.inf
    best_alpha = float("nan")
    best_minimum = float("nan")

    for start in range(len(eigenvalues) - minimum_tail):
        tail = eigenvalues[start:]
        minimum = tail[0]

        log_ratio_sum = np.sum(np.log(tail / minimum))

        if log_ratio_sum <= 0:
            continue

        alpha = 1.0 + len(tail) / log_ratio_sum

        fitted_cdf = 1.0 - (tail / minimum) ** (1.0 - alpha)
        empirical_cdf = np.arange(len(tail)) / len(tail)

        distance = float(np.max(np.abs(fitted_cdf - empirical_cdf)))

        if distance < best_distance:
            best_distance = distance
            best_alpha = float(alpha)
            best_minimum = float(minimum)

    return best_alpha, best_minimum, float(best_distance)


def baseline_summary(covariance):
    """
    All baselines for one covariance matrix, in one dict.
    """
    eigenvalues = covariance_eigenvalues(covariance)

    alpha, r_squared = alpha_req(eigenvalues)

    summary = {
        "rankme_li": rankme_li(eigenvalues),
        "rankme_garrido": rankme_garrido(eigenvalues),
        "alpha_req": alpha,
        "alpha_req_r_squared": r_squared,
        "matrix_entropy": matrix_entropy(eigenvalues),
    }

    return summary
