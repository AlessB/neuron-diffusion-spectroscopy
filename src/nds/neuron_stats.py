"""
neuron_stats.py
===============

Per-neuron statistics that need no graph: the "neuron family" numbers of Gurnee et al. (2024),
and the position-0 spike check. Pure numpy. Computed at every checkpoint from sums kept during
collection (activations.collect_neuron_moments) and from the checkpoint's weights.

Why they are here: when the neuron graph shows an event, these say what KIND of neurons carry it
(the "explained" events of DECISIONS D37 need a name). Gurnee et al. found that neurons fall into
families with telltale statistics:
- activation skew and kurtosis: heavy-tailed neurons fire rarely and strongly;
- sparsity: the fraction of tokens on which a neuron is positive (they call < 0.02 sparse);
- vocabulary kurtosis: how peaked cos(W_U, w_out) is over the vocabulary. A neuron whose output
  direction lines up with a few tokens' unembeddings is a "prediction" neuron (large positive
  skew) or a "suppression" neuron (large negative skew); Gurnee et al. use kurtosis > 10.

All of these change when the neurons are rotated (they are per-neuron), so like localisation
they are neuron-basis numbers.

Maths from raw moments (one pass, float64 sums):
    variance  = E[x^2] - E[x]^2
    skew      = (E[x^3] - 3 E[x] E[x^2] + 2 E[x]^3) / variance^(3/2)
    kurtosis  = (E[x^4] - 4 E[x] E[x^3] + 6 E[x]^2 E[x^2] - 3 E[x]^4) / variance^2
(kurtosis here is the plain fourth standardised moment: 3 for a Gaussian.)

Credit
------
- W. Gurnee, T. Horsley, Z. C. Guo, T. R. Kheirkhah, Q. Sun, W. Hathaway, N. Nanda, D. Bertsimas
  (2024). "Universal neurons in GPT2 language models." TMLR 2024.
  Code (MIT): https://github.com/wesg52/universal-neurons (analysis/weights.py,
  neuron_vocab_cosine_moments; paper_notebooks/family_count.ipynb thresholds). Re-implemented.
- M. Sun, X. Chen, J. Z. Kolter, Z. Liu (2024). "Massive activations in large language models."
  COLM 2024. (the top-activation vs median ratio used for position 0)
"""

import numpy as np


def skew_and_kurtosis(mean, mean_square, mean_cube, mean_fourth_power):
    """
    Per-neuron skew and kurtosis from raw moments E[x], E[x^2], E[x^3], E[x^4].
    Neurons with (numerically) zero variance get NaN.
    """
    m1 = np.asarray(mean, dtype=np.float64)
    m2 = np.asarray(mean_square, dtype=np.float64)
    m3 = np.asarray(mean_cube, dtype=np.float64)
    m4 = np.asarray(mean_fourth_power, dtype=np.float64)

    variance = m2 - m1 ** 2

    third_central = m3 - 3.0 * m1 * m2 + 2.0 * m1 ** 3

    fourth_central = m4 - 4.0 * m1 * m3 + 6.0 * m1 ** 2 * m2 - 3.0 * m1 ** 4

    skew = np.full(len(m1), np.nan)
    kurtosis = np.full(len(m1), np.nan)

    alive = variance > 1e-12

    skew[alive] = third_central[alive] / variance[alive] ** 1.5
    kurtosis[alive] = fourth_central[alive] / variance[alive] ** 2

    return skew, kurtosis


def vocabulary_alignment_moments(unembedding, output_weights, chunk_size=256):
    """
    Gurnee et al.'s vocabulary statistics: for each neuron, the cosines between its output
    direction w_out and every token's unembedding row, summarised by skew and kurtosis over the
    vocabulary.

    unembedding     (vocabulary, d_model)   W_U (Pythia: embed_out.weight)
    output_weights  (d_model, neurons)      W_out of one MLP (Pythia: mlp.dense_4h_to_h.weight)
    Returns (skew, kurtosis), each of shape (neurons,).
    """
    vocabulary_rows = np.asarray(unembedding, dtype=np.float64)
    neuron_columns = np.asarray(output_weights, dtype=np.float64)

    row_norms = np.linalg.norm(vocabulary_rows, axis=1, keepdims=True)
    row_norms = np.maximum(row_norms, 1e-12)
    unit_rows = vocabulary_rows / row_norms

    column_norms = np.linalg.norm(neuron_columns, axis=0, keepdims=True)
    column_norms = np.maximum(column_norms, 1e-12)
    unit_columns = neuron_columns / column_norms

    number_of_neurons = unit_columns.shape[1]

    skew = np.zeros(number_of_neurons)
    kurtosis = np.zeros(number_of_neurons)

    for start in range(0, number_of_neurons, chunk_size):
        stop = min(start + chunk_size, number_of_neurons)

        # (vocabulary, chunk) cosines
        cosines = unit_rows @ unit_columns[:, start:stop]

        centred = cosines - cosines.mean(axis=0)

        variance = (centred ** 2).mean(axis=0)
        variance = np.maximum(variance, 1e-30)

        skew[start:stop] = (centred ** 3).mean(axis=0) / variance ** 1.5
        kurtosis[start:stop] = (centred ** 4).mean(axis=0) / variance ** 2

    return skew, kurtosis


def position_zero_spike_ratio(position_zero_max_absolute, max_absolute_elsewhere):
    """
    How much bigger the largest activation at position 0 is than a typical neuron's largest
    activation elsewhere: max over neurons of |x| at position 0, divided by the median over
    neurons of max |x| at the other positions. Sun et al. call activations about 1,000x the median
    "massive"; above 100 is worth flagging. D30 leaves position 0 out of every statistic, and this
    number shows whether that mattered.
    """
    at_zero = np.asarray(position_zero_max_absolute, dtype=np.float64)
    elsewhere = np.asarray(max_absolute_elsewhere, dtype=np.float64)

    median_elsewhere = float(np.median(elsewhere))

    if median_elsewhere <= 0:
        return float("nan")

    return float(at_zero.max() / median_elsewhere)


def layer_neuron_statistics(
    mean,
    variance,
    raw_moments,
    position_zero=None,
    max_absolute_elsewhere=None,
    vocabulary_skew=None,
    vocabulary_kurtosis=None,
):
    """
    Everything for one layer: per-neuron arrays (for the .npz file) and layer scalars (for
    summary.json).

    mean, variance   per neuron, over the same tokens as raw_moments (from the merged moments)
    raw_moments      {"mean_cube", "mean_fourth_power", "positive_fraction"} from collection
    """
    mean = np.asarray(mean, dtype=np.float64)
    variance = np.asarray(variance, dtype=np.float64)

    mean_square = variance + mean ** 2

    skew, kurtosis = skew_and_kurtosis(
        mean,
        mean_square,
        raw_moments["mean_cube"],
        raw_moments["mean_fourth_power"],
    )

    positive_fraction = np.asarray(raw_moments["positive_fraction"], dtype=np.float64)

    arrays = {
        "activation_skew": skew,
        "activation_kurtosis": kurtosis,
        "positive_fraction": positive_fraction,
    }

    scalars = {
        "median_activation_kurtosis": float(np.nanmedian(kurtosis)),
        "sparse_neuron_fraction": float(np.mean(positive_fraction < 0.02)),
    }

    if vocabulary_skew is not None:
        arrays["vocabulary_skew"] = np.asarray(vocabulary_skew, dtype=np.float64)
        arrays["vocabulary_kurtosis"] = np.asarray(vocabulary_kurtosis, dtype=np.float64)

        scalars["prediction_neuron_fraction"] = float(
            np.mean((arrays["vocabulary_kurtosis"] > 10.0) & (arrays["vocabulary_skew"] > 0.0))
        )
        scalars["suppression_neuron_fraction"] = float(
            np.mean((arrays["vocabulary_kurtosis"] > 10.0) & (arrays["vocabulary_skew"] < 0.0))
        )

    if position_zero is not None:
        arrays["position_zero_mean_absolute"] = np.asarray(position_zero["mean_absolute"], dtype=np.float64)
        arrays["position_zero_max_absolute"] = np.asarray(position_zero["max_absolute"], dtype=np.float64)

        if max_absolute_elsewhere is not None:
            scalars["position_zero_spike_ratio"] = position_zero_spike_ratio(
                position_zero["max_absolute"],
                max_absolute_elsewhere,
            )

    return arrays, scalars
