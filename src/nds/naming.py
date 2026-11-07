"""
naming.py
=========

Part 2 of the project, "name the events": connect named circuits to our neuron graph. Pure numpy.

Two questions:

1. Which MLP neurons sit behind a named feature?
   Marks et al. (2025) dictionaries read the 512-dimensional MLP OUTPUT, not our 2,048 hidden
   neurons. In a Pythia layer
        a = GELU(h)                     the 2,048 neurons (our graph nodes)
        y = W_out a + b_out             the 512-dimensional MLP output
        pre_i = e_i . (y - b) + c_i     feature i's pre-activation (dictionary encoder row e_i)
   There is no nonlinearity between a and y, so exactly
        pre_i = sum_j a_j L_ij + kappa_i,   L_ij = e_i . W_out[:, j],   kappa_i = e_i . (b_out - b) + c_i
   L is a "virtual weight" (Dunefsky et al. 2024 use the same idea for transcoders).

2. Did a given set of neurons S form a module at a given checkpoint?
   The diffusion retention probability
        r_t(S) = (1 / vol S) sum_{i in S} d_i sum_{j in S} (P^t)_ij
   is the chance that a random walk started inside S (from the degree distribution on S) is still
   inside S after t steps. r_1(S) = 1 - conductance(S). About 1 for an isolated module, about
   vol(S) / vol(all) for a random set. Report it as a z-score against random sets of the same size.

Credit
------
- S. Marks, C. Rager, E. J. Michaud, Y. Belinkov, D. Bau, A. Mueller (2025). "Sparse feature
  circuits." ICLR 2025. Code (MIT): https://github.com/saprmarks/feature-circuits ,
  https://github.com/saprmarks/dictionary_learning
- J. Dunefsky, P. Chlenski, N. Nanda (2024). "Transcoders find interpretable LLM feature circuits."
  NeurIPS 2024. (input-invariant "virtual weight" times activation)
- The 80% cumulative rule follows circuit-tracer's node_threshold (Hanna et al. 2025) and ADAG's
  node attribution threshold (Arora et al. 2026).
- The retention probability and the W_out bridge were worked out in this project's code study
  (research/S3_code_circuits.md, 17 Oct 2026).
"""

import numpy as np


def feature_loadings_on_neurons(
    encoder_weight,
    encoder_bias,
    dictionary_bias,
    mlp_output_weight,
    mlp_output_bias,
):
    """
    How strongly each MLP neuron feeds each MLP-output dictionary feature.

    encoder_weight       (features, 512)    dictionary encoder rows e_i     (ae.encoder.weight)
    encoder_bias         (features,)        c_i                             (ae.encoder.bias)
    dictionary_bias      (512,)             b, subtracted before encoding   (ae.bias)
    mlp_output_weight    (512, 2048)        W_out                           (mlp.dense_4h_to_h.weight)
    mlp_output_bias      (512,)             b_out                           (mlp.dense_4h_to_h.bias)

    Returns (loadings, offsets): loadings (features, 2048), offsets (features,), such that exactly
        pre_activation_i = sum_j a_j * loadings[i, j] + offsets[i]
    """
    loadings = encoder_weight @ mlp_output_weight

    bias_difference = mlp_output_bias - dictionary_bias

    offsets = encoder_weight @ bias_difference
    offsets = offsets + encoder_bias

    return loadings, offsets


def mean_absolute_contributions(
    neuron_activations,
    loadings,
    feature_is_active,
):
    """
    S_ij = mean over tokens where feature i fires of | a_j L_ij |.

    neuron_activations   (tokens, 2048)       post-GELU activations (hook on mlp.act)
    loadings             (features, 2048)     from feature_loadings_on_neurons
    feature_is_active    (tokens, features)   True where the feature's activation is > 0
    """
    number_of_features = loadings.shape[0]
    number_of_neurons = loadings.shape[1]

    scores = np.zeros((number_of_features, number_of_neurons))

    for feature in range(number_of_features):
        active_tokens = feature_is_active[:, feature]

        if not np.any(active_tokens):
            continue

        contributions = neuron_activations[active_tokens] * loadings[feature]

        scores[feature] = np.abs(contributions).mean(axis=0)

    return scores


def smallest_set_holding_share(scores, share=0.8):
    """
    Indices, largest score first, until they hold `share` of the total score.
    """
    scores = np.asarray(scores, dtype=np.float64)

    order = np.argsort(scores)[::-1]

    cumulative_share = np.cumsum(scores[order]) / scores.sum()

    number_to_keep = int(np.searchsorted(cumulative_share, share)) + 1

    return order[:number_to_keep]


def retention_probability(kernel, member_rows, diffusion_time=1):
    """
    r_t(S): probability that a walk started on S (degree-weighted) is still in S after t steps.

    kernel        (n, n) symmetric edge weights of the neuron graph
    member_rows   row indices of S in that kernel (map model neuron ids with live_neuron_ids first)
    """
    degrees = kernel.sum(axis=1)

    transition = kernel / degrees[:, np.newaxis]

    member_rows = np.asarray(member_rows)

    # start: degree distribution restricted to S, normalised
    start = np.zeros(kernel.shape[0])
    start[member_rows] = degrees[member_rows]
    start = start / start.sum()

    distribution = start
    for _ in range(int(diffusion_time)):
        distribution = distribution @ transition

    return float(distribution[member_rows].sum())


def retention_z_score(
    kernel,
    member_rows,
    diffusion_time=1,
    number_of_random_sets=200,
    random_generator=None,
):
    """
    r_t(S) compared with random sets of the same size: (r - mean_random) / std_random.

    Returns (r, z, random_values).
    """
    if random_generator is None:
        random_generator = np.random.default_rng(0)

    observed = retention_probability(kernel, member_rows, diffusion_time)

    set_size = len(member_rows)
    number_of_rows = kernel.shape[0]

    random_values = np.zeros(number_of_random_sets)
    for index in range(number_of_random_sets):
        random_rows = random_generator.choice(number_of_rows, size=set_size, replace=False)
        random_values[index] = retention_probability(kernel, random_rows, diffusion_time)

    spread = random_values.std()

    if spread > 0:
        z_score = (observed - random_values.mean()) / spread
    else:
        z_score = float("inf")

    return observed, float(z_score), random_values
