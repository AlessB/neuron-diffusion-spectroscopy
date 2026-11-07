"""
token_sample.py
===============

A small fixed sample of token activations, kept at every checkpoint (DECISIONS D31). Pure numpy.

The covariance (moments.py) is all the neuron graph needs, but some measures need individual
tokens: a graph over tokens instead of neurons, an intrinsic dimension, a shuffle null. We keep the
same 2,048 token positions at every checkpoint, all neurons, in float16 (about 25 MB per layer of
pythia-70m), on the shared drive.

Measures computed from the sample:

- Token-graph DSE (DECISIONS D34): Liao et al.'s diffusion spectral entropy on a graph whose nodes
  are TOKENS. Their recipe: a single-bandwidth Gaussian on raw Euclidean distances
  G = exp(-D^2 / (2 sigma^2)) with sigma = 10 and self loops kept, the symmetric conjugate
  D^-1/2 G D^-1/2, eigenvalues lambda, p_i = |lambda_i|^t / sum |lambda|^t, entropy in bits.
  Distances between tokens do not change when the neurons are rotated, so this is a baseline, not
  a neuron-level readout. Because sigma = 10 only means something on their scale, we also report
  the same entropy with sigma set to the median distance.
- TwoNN intrinsic dimension (DECISIONS D35): for each token, r1 and r2 are the distances to its
  nearest and second-nearest neighbours and mu = r2 / r1. If the data locally look like a
  d-dimensional set, the cumulative distribution of mu is F(mu) = 1 - mu^-d, so
  -log(1 - F(mu)) = d log(mu): a line through the origin whose slope is d. Facco et al. fit that line
  after dropping the largest 10% of mu values. Lee et al. (2025) use this on Pythia checkpoints.
- Knowledge entropy on the sample, original and after 5 random rotations (baselines.py; Kim et al.
  2025): the neuron-basis, non-graph baseline and its control.
- Embedding condensation (Liu et al. 2026): mean cosine between token vectors.
- Shuffle null (DECISIONS D36): every neuron's column permuted independently, which keeps each
  neuron's own distribution and destroys every relationship between neurons.

Credit
------
- D. Liao, C. Liu, B. W. Christensen, A. Tong, G. Huguet, G. Wolf, M. Nickel, I. Adelstein,
  S. Krishnaswamy (2023). "Assessing neural network representations during training using
  noise-resilient diffusion spectral entropy." ICML 2023 workshop TAG-ML. Recipe re-implemented
  from https://github.com/ChenLiu-1996/DiffusionSpectralEntropy (Yale non-commercial licence; not
  copied: dse_dsmi/diffusion.py l.24-36, dse.py l.153-180, read in research/S1).
- E. Facco, M. d'Errico, A. Rodriguez, A. Laio (2017). "Estimating the intrinsic dimension of
  datasets by a minimal neighborhood information." Scientific Reports 7, 12140. Fit as in DADApy
  (https://github.com/sissa-data-science/DADApy, compute_id_2NN), re-implemented.
- C. Liu et al., D. Liao, ..., S. Krishnaswamy (2026). "Dispersion loss counteracts embedding
  condensation and improves generalization in small language models." ICML 2026. Code:
  https://github.com/ChenLiu-1996/LM-Dispersion (Yale non-commercial licence; read for the
  definition, re-implemented, nothing copied).
- J. H. Lee, T. Jiralerspong, L. Yu, Y. Bengio, E. Cheng (2025). "Geometric signatures of
  compositionality across a language model's lifetime." ACL 2025. (TwoNN on Pythia checkpoints)
"""

import numpy as np
from scipy.linalg import eigvalsh
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

from nds.baselines import knowledge_entropy_of_sample
from nds.spectra import diffusion_spectral_entropy


def choose_token_positions(
    number_of_sequences,
    sequence_length,
    sample_size=2048,
    skip_first_positions=1,
    seed=0,
):
    """
    The fixed token positions kept at every checkpoint, as (sequence index, position) pairs,
    sorted so collection can pick them batch by batch. Positions before `skip_first_positions`
    are never chosen (DECISIONS D30).
    """
    usable_positions = sequence_length - skip_first_positions

    total = number_of_sequences * usable_positions

    if sample_size > total:
        raise ValueError("asked for more sampled tokens than the probe text has")

    random_generator = np.random.default_rng(seed)

    flat_choice = random_generator.choice(total, size=sample_size, replace=False)

    flat_choice = np.sort(flat_choice)

    sequence_index = flat_choice // usable_positions

    position = flat_choice % usable_positions + skip_first_positions

    return sequence_index.astype(np.int64), position.astype(np.int64)


def gather_token_sample(activations_by_sequence, sequence_index, position):
    """
    Pick the sampled tokens out of activations shaped (sequences, positions, neurons).
    (Collection does the same batch by batch on the GPU.)
    """
    values = np.asarray(activations_by_sequence)

    return values[sequence_index, position, :]


def shuffle_neurons_independently(token_sample, random_generator):
    """
    The shuffle null: permute each neuron's column on its own.
    """
    sample = np.array(token_sample, dtype=np.float64)

    shuffled = np.empty_like(sample)

    for neuron in range(sample.shape[1]):
        order = random_generator.permutation(sample.shape[0])
        shuffled[:, neuron] = sample[order, neuron]

    return shuffled


def token_graph_dse(token_sample, diffusion_times=(1, 2, 4, 8), sigma=10.0):
    """
    Liao et al.'s DSE on the graph over sampled tokens.

    sigma   the Gaussian bandwidth on raw Euclidean distances; 10 is their default, "median" uses
            the median pairwise distance instead (scale-free).
    Returns {t: entropy in bits}.
    """
    points = np.asarray(token_sample, dtype=np.float64)

    distances = squareform(pdist(points))

    if sigma == "median":
        upper = distances[np.triu_indices_from(distances, k=1)]
        bandwidth = float(np.median(upper))
    else:
        bandwidth = float(sigma)

    bandwidth = max(bandwidth, 1e-12)

    # G in Liao et al.; the constant in front cancels when the eigenvalues are normalised
    gaussian = np.exp(
        - distances ** 2 / (2.0 * bandwidth ** 2)
    )

    degrees = gaussian.sum(axis=1)

    inverse_root_degrees = 1.0 / np.sqrt(degrees)

    symmetric = gaussian * inverse_root_degrees[:, np.newaxis] * inverse_root_degrees[np.newaxis, :]

    eigenvalues = eigvalsh(symmetric)

    entropies = {}
    for diffusion_time in diffusion_times:
        entropies[int(diffusion_time)] = diffusion_spectral_entropy(eigenvalues, diffusion_time=diffusion_time)

    return entropies


def twonn_intrinsic_dimension(token_sample, discard_fraction=0.1):
    """
    Facco et al.'s TwoNN estimate of the intrinsic dimension of the sampled tokens.
    """
    points = np.asarray(token_sample, dtype=np.float64)

    distances = squareform(pdist(points))

    np.fill_diagonal(distances, np.inf)

    sorted_distances = np.sort(distances, axis=1)

    nearest = sorted_distances[:, 0]
    second = sorted_distances[:, 1]

    # identical tokens (distance 0) carry no information about dimension
    keep = nearest > 0

    ratios = second[keep] / nearest[keep]

    ratios = np.sort(ratios)

    count = len(ratios)

    kept_count = int(np.floor(count * (1.0 - discard_fraction)))

    log_ratio = np.log(ratios[:kept_count])

    empirical_cdf = np.arange(1, count + 1)[:kept_count] / count

    target = - np.log(1.0 - empirical_cdf)

    # least squares line through the origin: slope = sum(x y) / sum(x x)
    dimension = float(np.sum(log_ratio * target) / np.sum(log_ratio * log_ratio))

    return dimension


def mean_token_cosine(token_sample, centre=True, maximum_tokens=2048):
    """
    "Embedding condensation" (Liu et al. 2026): the mean cosine similarity between pairs of
    different tokens' activation vectors. Near 1 = all tokens point the same way (condensed); near
    0 = spread out. Centring first removes the shared mean direction (post-GELU activations are
    mostly positive, so uncentred cosines are high for trivial reasons); we report both.
    Difference from theirs: they average cosines among the tokens of ONE text (uncentred), over
    many texts (pypi/embedding_condensation, measure_embedding_condensation); ours pools the fixed
    sample across documents, so it also counts between-document spread.
    Rotation-invariant: rotating the neurons rotates every token vector by the same Q.
    """
    sample = np.asarray(token_sample, dtype=np.float64)[:maximum_tokens]

    if centre:
        sample = sample - sample.mean(axis=0)

    lengths = np.linalg.norm(sample, axis=1)

    keep = lengths > 1e-12

    unit_rows = sample[keep] / lengths[keep, np.newaxis]

    count = unit_rows.shape[0]

    if count < 2:
        return float("nan")

    # sum of all pairwise cosines = |sum of unit vectors|^2 - count (the diagonal)
    total = np.sum(unit_rows.sum(axis=0) ** 2) - count

    return float(total / (count * (count - 1)))


def token_sample_baselines(token_sample, diffusion_times=(1, 2, 4, 8), number_of_rotations=5, seed=0):
    """
    Every token-sample measure in one dict (written into summary.json).
    """
    sample = np.asarray(token_sample, dtype=np.float64)

    knowledge = knowledge_entropy_of_sample(
        sample,
        number_of_rotations=number_of_rotations,
        random_generator=np.random.default_rng(seed),
    )

    summary = {
        "token_graph_dse": token_graph_dse(sample, diffusion_times, sigma=10.0),
        "token_graph_dse_median_bandwidth": token_graph_dse(sample, diffusion_times, sigma="median"),
        "twonn_dimension": twonn_intrinsic_dimension(sample),
        "mean_token_cosine_centred": mean_token_cosine(sample, centre=True),
        "mean_token_cosine_uncentred": mean_token_cosine(sample, centre=False),
        "knowledge_entropy_sample": knowledge["original"],
        "knowledge_entropy_sample_rotated": knowledge["rotated"],
        "number_of_sampled_tokens": int(sample.shape[0]),
    }

    return summary
