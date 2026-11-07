"""
S9 (D50 follow-up). A cross-validated, nearly unbiased subspace distance.

Idea borrowed from neuroimaging: the squared distance between two noisy estimates is biased
upwards by the noise (it is a sum of squares, never below 0). The "crossnobis" estimator (Walther
et al. 2016; Diedrichsen and colleagues) computes the difference on two independent halves of the
data and multiplies them, so the noise, independent between halves, averages out of the product:

    plain:            d     = ||P_a - P_b||_F^2 / (2k)
    cross-validated:  d_cv  = trace[(P_a^(1) - P_b^(1)) (P_a^(2) - P_b^(2))] / (2k)

P = U U^T is the projector onto a module subspace of k modes; (1) and (2) are two disjoint halves
of the probe text (blocks 0-3 and 4-7); a and b are neighbouring checkpoints. With a shared probe
text each half is still paired between the checkpoints (S7), so common-random-number cancellation
is kept inside each half, and the cross-product removes what is left. d_cv can be negative when
nothing changed (that is what unbiased means), and its expectation does not depend on the number
of tokens, while the plain d shrinks like 1 / tokens. This matters because the effective number
of tokens changes during training (lag-one correlation, S4): a plain d would drift for that reason
alone.

Not exactly unbiased: an estimated projector is itself a biased estimate of the true one (second
order in the noise), and the bias is the same at both checkpoints only if the noise level is.

Tests, 256 neurons, 8 modules of 24 plus 64 free, strength 0.15:
  1. no change, fresh tokens per checkpoint, tokens 2,000 to 32,000: d and d_cv against tokens
  2. one neuron of 192 changes module: d and d_cv, fresh and shared tokens, 10 repeats each

    python research/round5/sim9_crossvalidated_subspace_distance.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "round4"))

from sims_common import MODULE_NEURONS
from sims_common import MODULES
from sims_common import NEURONS

from nds.graphs import neuron_graph_from_covariance
from nds.moments import BlockedNeuronMoments
from nds.spectra import diffusion_spectrum
from nds.spectra import eigengap_module_count
from nds.spectra import leading_eigenvectors
from nds.spectra import symmetric_diffusion_operator

STRENGTH = 0.15
BLOCKS = 8


def projector(covariance, count):
    graph = neuron_graph_from_covariance(covariance, minimum_firing_rate=0.0)
    operator, _ = symmetric_diffusion_operator(graph["kernel"])
    vectors = leading_eigenvectors(operator, count)
    return vectors @ vectors.T


def module_count(covariance):
    graph = neuron_graph_from_covariance(covariance, minimum_firing_rate=0.0)
    spectrum = diffusion_spectrum(graph["kernel"])
    found, _, _ = eigengap_module_count(spectrum["laplacian_eigenvalues"])
    return max(int(found), 2)


def activations(labels, tokens, generator, shared=None):
    if shared is None:
        latents = generator.standard_normal((tokens, MODULES))
        private = generator.standard_normal((tokens, NEURONS))
        free = generator.standard_normal((tokens, NEURONS))
    else:
        latents, private, free = shared
    values = np.sqrt(1 - STRENGTH) * private
    inside = labels >= 0
    values[:, inside] += np.sqrt(STRENGTH) * latents[:, labels[inside]]
    values[:, ~inside] += np.sqrt(STRENGTH) * free[:, ~inside]
    return values


def blocked(values):
    moments = BlockedNeuronMoments(NEURONS, number_of_blocks=BLOCKS)
    size = values.shape[0] // BLOCKS
    for block in range(BLOCKS):
        moments.add_batch(values[block * size:(block + 1) * size], block)
    return moments


def distances(moments_a, moments_b):
    count = module_count(moments_a.merged().covariance())
    full_a = projector(moments_a.merged().covariance(), count)
    full_b = projector(moments_b.merged().covariance(), count)
    plain = np.sum((full_a - full_b) ** 2) / (2 * count)
    first = list(range(BLOCKS // 2))
    second = list(range(BLOCKS // 2, BLOCKS))
    difference_one = projector(moments_a.merged(first).covariance(), count) - projector(moments_b.merged(first).covariance(), count)
    difference_two = projector(moments_a.merged(second).covariance(), count) - projector(moments_b.merged(second).covariance(), count)
    crossed = np.sum(difference_one * difference_two) / (2 * count)
    return plain, crossed


def base_labels():
    labels = np.full(NEURONS, -1)
    labels[:MODULE_NEURONS] = np.repeat(np.arange(MODULES), MODULE_NEURONS // MODULES)
    return labels


def main():
    print("1. No change, fresh tokens at each checkpoint: mean (sd) over 6 repeats")
    print(f"   {'tokens':>7s} {'plain d':>18s} {'cross-validated d':>22s}")
    for tokens in [2000, 4000, 8000, 16000, 32000]:
        plain_values = []
        crossed_values = []
        for repeat in range(6):
            generator = np.random.default_rng(900 + repeat)
            labels = base_labels()
            plain, crossed = distances(blocked(activations(labels, tokens, generator)), blocked(activations(labels, tokens, generator)))
            plain_values.append(plain)
            crossed_values.append(crossed)
        print(f"   {tokens:7d} {np.mean(plain_values):10.5f} ({np.std(plain_values):.5f}) {np.mean(crossed_values):13.5f} ({np.std(crossed_values):.5f})")

    print("\n2. One neuron of 192 changes module, 4,000 tokens: mean (sd) over 10 repeats; and with no change")
    for sharing in ["fresh", "shared"]:
        for moved in [0, 1]:
            plain_values = []
            crossed_values = []
            for repeat in range(10):
                generator = np.random.default_rng(950 + repeat)
                labels = base_labels()
                after = labels.copy()
                if moved:
                    after[0] = 1
                shared = None
                if sharing == "shared":
                    shared = (generator.standard_normal((4000, MODULES)), generator.standard_normal((4000, NEURONS)), generator.standard_normal((4000, NEURONS)))
                plain, crossed = distances(blocked(activations(labels, 4000, generator, shared)), blocked(activations(after, 4000, generator, shared)))
                plain_values.append(plain)
                crossed_values.append(crossed)
            print(f"   {sharing:6s} tokens, moved {moved}: plain {np.mean(plain_values):.5f} ({np.std(plain_values):.5f}), "
                  f"cross-validated {np.mean(crossed_values):.5f} ({np.std(crossed_values):.5f})")


if __name__ == "__main__":
    main()
