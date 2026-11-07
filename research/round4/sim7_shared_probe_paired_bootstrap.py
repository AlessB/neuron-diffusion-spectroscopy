"""
S7. The probe text is the same at every checkpoint: what is the right noise yardstick then?

Found while checking S6 on the dry run: there the signal-to-noise ratio of the subspace change was
about 0.0002 at every transition, because the dry run (like the real pipeline) feeds the same
probe sequences to every checkpoint. S6 drew fresh tokens per checkpoint. With shared tokens most
of the token noise is common to both checkpoints and cancels in the distance between them.

Numerical analysis: this is the method of common random numbers, a standard variance reduction for
comparing two simulated systems: drive both with the same random input and the noise of the
difference shrinks, as long as the two outputs are positively correlated (also called correlated
sampling or matched pairs; Kahn & Marshall 1953; Botev & Ridder 2017, "Variance Reduction", Wiley
StatsRef). The matching bootstrap resamples the same token blocks for both checkpoints, so each
replicate distance carries the shared-token correlation (the paired bootstrap; Efron & Tibshirani
1993, "An Introduction to the Bootstrap").

Compared yardsticks for d (distance between neighbouring module subspaces):
  unpaired floor   n_from + n_to, each n from that checkpoint's own bootstrap (S6, what the code
                   computes now)
  paired           spread of d over replicates that resample the same blocks at both checkpoints;
                   z = (d - median of the other transitions' d) / paired standard deviation
  contrast         d / median of the other transitions' d (no noise model at all)

Test: 16 evenly spaced checkpoints, 256 neurons, 8 modules of 24 plus 64 free, strength 0.3,
4,000 tokens in 8 blocks of consecutive tokens (like probe sequences), the same tokens at every
checkpoint. The network drifts slowly (each checkpoint mixes 2% fresh variation into each token's
latent and private parts). At transition 7 -> 8 a fraction of the module neurons change module:
0 (no event), 5%, 10%, 25%. 3 seeds, 12 paired replicates.

    python research/round4/sim7_shared_probe_paired_bootstrap.py
"""

import numpy as np

from sims_common import MODULE_NEURONS
from sims_common import MODULES
from sims_common import NEURONS

from nds.graphs import neuron_graph_from_covariance
from nds.moments import BlockedNeuronMoments
from nds.spectra import chordal_subspace_distance
from nds.spectra import diffusion_spectrum
from nds.spectra import eigengap_module_count
from nds.spectra import leading_eigenvectors
from nds.spectra import symmetric_diffusion_operator

STRENGTH = 0.3
TOKENS = 4000
BLOCKS = 8
CHECKPOINTS = 16
EVENT = 8
DRIFT = 0.02
REPLICATES = 12


def module_subspace(covariance, count=None):
    graph = neuron_graph_from_covariance(covariance, minimum_firing_rate=0.0)
    if count is None:
        spectrum = diffusion_spectrum(graph["kernel"])
        found, _, _ = eigengap_module_count(spectrum["laplacian_eigenvalues"])
        count = max(int(found), 2)
        return spectrum["symmetric_vectors"][:, :count], count
    operator, _ = symmetric_diffusion_operator(graph["kernel"])
    return leading_eigenvectors(operator, count), count


def blocked(values):
    moments = BlockedNeuronMoments(NEURONS, number_of_blocks=BLOCKS)
    size = TOKENS // BLOCKS
    for block in range(BLOCKS):
        moments.add_batch(values[block * size:(block + 1) * size], block)
    return moments


def run(fraction, seed):
    generator = np.random.default_rng(700 + seed)
    labels = np.full(NEURONS, -1)
    labels[:MODULE_NEURONS] = np.repeat(np.arange(MODULES), MODULE_NEURONS // MODULES)
    labels_after = labels.copy()
    moved = generator.permutation(MODULE_NEURONS)[: int(round(fraction * MODULE_NEURONS))]
    if len(moved) > 0:
        labels_after[moved] = (labels_after[moved] + 1 + generator.integers(0, MODULES - 1, len(moved))) % MODULES

    latents = generator.standard_normal((TOKENS, MODULES))
    private = generator.standard_normal((TOKENS, NEURONS))
    free = generator.standard_normal((TOKENS, NEURONS))

    moments = []
    for index in range(CHECKPOINTS):
        if index > 0:
            latents = np.sqrt(1 - DRIFT) * latents + np.sqrt(DRIFT) * generator.standard_normal(latents.shape)
            private = np.sqrt(1 - DRIFT) * private + np.sqrt(DRIFT) * generator.standard_normal(private.shape)
            free = np.sqrt(1 - DRIFT) * free + np.sqrt(DRIFT) * generator.standard_normal(free.shape)
        current = labels_after if index >= EVENT else labels
        values = np.sqrt(1 - STRENGTH) * private
        inside = current >= 0
        values[:, inside] += np.sqrt(STRENGTH) * latents[:, current[inside]]
        values[:, ~inside] += np.sqrt(STRENGTH) * free[:, ~inside]
        moments.append(blocked(values))

    subspaces = []
    counts = []
    noises = []
    for moment in moments:
        vectors, count = module_subspace(moment.merged().covariance())
        subspaces.append(vectors)
        counts.append(count)
        own = []
        for _ in range(4):
            replicate_vectors, _ = module_subspace(moment.bootstrap_replicate(generator).covariance(), count)
            own.append(chordal_subspace_distance(vectors, replicate_vectors))
        noises.append(np.mean(own))

    distances = []
    paired_spread = []
    for index in range(CHECKPOINTS - 1):
        distances.append(chordal_subspace_distance(subspaces[index], subspaces[index + 1]))
        replicate_distances = []
        for _ in range(REPLICATES):
            chosen = generator.integers(0, BLOCKS, BLOCKS)
            vectors_from, _ = module_subspace(moments[index].merged(chosen).covariance(), counts[index])
            vectors_to, _ = module_subspace(moments[index + 1].merged(chosen).covariance(), counts[index + 1])
            replicate_distances.append(chordal_subspace_distance(vectors_from, vectors_to))
        paired_spread.append(np.std(replicate_distances, ddof=1))
    return np.asarray(distances), np.asarray(noises), np.asarray(paired_spread)


def main():
    event = EVENT - 1
    print(f"{'moved':>6s} {'seed':>4s} | {'d event':>9s} {'d others':>9s} | {'unpaired floor':>14s} {'d/floor':>8s} | "
          f"{'paired sd':>9s} {'z event':>8s} {'max z other':>11s} | {'contrast':>8s} {'max other':>9s}")
    for fraction in [0.0, 0.05, 0.10, 0.25]:
        for seed in range(3):
            distances, noises, spread = run(fraction, seed)
            floor = noises[:-1] + noises[1:]
            others = np.delete(distances, event)
            baseline = np.median(others)
            z = (distances - baseline) / spread
            contrast = distances / baseline
            print(f"{fraction:6.2f} {seed:4d} | {distances[event]:9.5f} {baseline:9.5f} | {floor[event]:14.5f} "
                  f"{distances[event] / floor[event]:8.2f} | {spread[event]:9.5f} {z[event]:8.1f} "
                  f"{np.max(np.delete(z, event)):11.1f} | {contrast[event]:8.1f} {np.max(np.delete(contrast, event)):9.2f}")


if __name__ == "__main__":
    main()
