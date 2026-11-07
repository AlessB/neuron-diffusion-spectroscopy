"""
S6. Does the fidelity susceptibility survive Pythia's uneven checkpoint spacing?

Found while writing up S2: chi = d / dt^2 assumes the distance d between neighbouring
checkpoints shrinks like dt^2. That holds for smooth change of the structure, but token noise adds
a roughly constant amount n to every distance, whatever the spacing. In t = log10(step + 1),
Pythia's checkpoints are log-spaced up to step 512 (dt about 0.3) and then every 1,000 steps
(dt about 0.003 at the end). So noise alone gives a raw chi about 10,000 times larger at the end
than at the start. Numerical analysis: this is the usual problem of differentiating noisy data,
where the error of a difference quotient grows like noise / h^2 as the step h shrinks.

Fixes tested: subtract the noise floor measured by the bootstrap at each checkpoint
(module_subspace_noise, the distance between the full module subspace and a replicate's):
    chi_excess = max(d - (n_from + n_to), 0) / dt^2,     signal_to_noise = d / (n_from + n_to)

Test: 37 checkpoints on Pythia's schedule (all of 0..512, every 1,000 to 10,000, every 10,000 to
140,000, then 141,000 to 143,000), 256 neurons, 8 modules of 24 plus 64 free, module strength
0.15 (a weakly pinned subspace, Davis-Kahan r about 0.3), 4,000 fresh tokens per checkpoint in 8
blocks, 8 bootstrap replicates. Runs: "stable" (nothing changes) and "reshuffle" (half the module
neurons change module between steps 2,000 and 3,000). 3 seeds each.

    python research/round4/sim6_susceptibility_noise_floor.py
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
from nds.timeline import fidelity_susceptibility
from nds.timeline import subspace_change_signal_to_noise
from nds.timeline import training_time

STRENGTH = 0.15
TOKENS = 4000
BLOCKS = 8
REPLICATES = 8
EVENT_STEP = 3000


def pythia_like_steps():
    steps = [0] + [2 ** power for power in range(10)]
    steps += list(range(1000, 10001, 1000))
    steps += list(range(20000, 140001, 10000))
    steps += [141000, 142000, 143000]
    return np.asarray(steps)


def module_subspace(covariance):
    graph = neuron_graph_from_covariance(covariance, minimum_firing_rate=0.0)
    operator, _ = symmetric_diffusion_operator(graph["kernel"])
    spectrum = diffusion_spectrum(graph["kernel"])
    count, _, _ = eigengap_module_count(spectrum["laplacian_eigenvalues"])
    count = max(int(count), 2)
    return spectrum["symmetric_vectors"][:, :count], count


def checkpoint(labels, generator):
    latents = generator.standard_normal((TOKENS, MODULES))
    values = np.sqrt(1 - STRENGTH) * generator.standard_normal((TOKENS, NEURONS))
    inside = labels >= 0
    values[:, inside] += np.sqrt(STRENGTH) * latents[:, labels[inside]]
    values[:, ~inside] += np.sqrt(STRENGTH) * generator.standard_normal((TOKENS, int(np.sum(~inside))))

    blocked = BlockedNeuronMoments(NEURONS, number_of_blocks=BLOCKS)
    size = TOKENS // BLOCKS
    for block in range(BLOCKS):
        blocked.add_batch(values[block * size:(block + 1) * size], block)

    vectors, count = module_subspace(blocked.merged().covariance())

    noise = []
    for _ in range(REPLICATES):
        replicate = blocked.bootstrap_replicate(generator).covariance()
        graph = neuron_graph_from_covariance(replicate, minimum_firing_rate=0.0)
        operator, _ = symmetric_diffusion_operator(graph["kernel"])
        noise.append(chordal_subspace_distance(vectors, leading_eigenvectors(operator, count)))
    return vectors, float(np.mean(noise))


def run(scenario, seed, steps):
    generator = np.random.default_rng(600 + seed)
    labels_before = np.full(NEURONS, -1)
    labels_before[:MODULE_NEURONS] = np.repeat(np.arange(MODULES), MODULE_NEURONS // MODULES)
    labels_after = labels_before.copy()
    swapped = generator.permutation(MODULE_NEURONS)[: MODULE_NEURONS // 2]
    labels_after[swapped] = generator.permutation(labels_after[swapped])

    subspaces = []
    noises = []
    for step in steps:
        labels = labels_before
        if scenario == "reshuffle" and step >= EVENT_STEP:
            labels = labels_after
        vectors, noise = checkpoint(labels, generator)
        subspaces.append(vectors)
        noises.append(noise)

    distances = np.array([chordal_subspace_distance(subspaces[i], subspaces[i + 1]) for i in range(len(steps) - 1)])
    return distances, np.asarray(noises)


def describe(name, values, steps):
    finite = np.where(np.isfinite(values), values, -np.inf)
    peak = int(np.argmax(finite))
    rest = np.delete(values, peak)
    rest = rest[np.isfinite(rest)]
    median = np.median(rest) if len(rest) else np.nan
    contrast = values[peak] / median if median > 0 else np.inf
    return f"{name:>16s}: peak at {steps[peak]:>6d}->{steps[peak + 1]:<6d} peak/median {contrast:10.1f}"


def main():
    steps = pythia_like_steps()
    times = training_time(steps)
    dt = np.diff(times)
    print(f"{len(steps)} checkpoints; dt in log10(step + 1) from {dt.min():.4f} to {dt.max():.3f} "
          f"(1 / dt^2 from {1 / dt.max() ** 2:.0f} to {1 / dt.min() ** 2:.0f})")
    for scenario in ["stable", "reshuffle"]:
        print(f"\n===== {scenario} =====")
        for seed in range(3):
            distances, noises = run(scenario, seed, steps)
            raw = fidelity_susceptibility(distances, times)
            ratio = subspace_change_signal_to_noise(distances, noises)
            floor = noises[:-1] + noises[1:]
            excess = np.maximum(distances - floor, 0.0) / dt ** 2
            print(f"seed {seed}: mean neighbour distance {np.mean(distances):.4f}, mean noise floor "
                  f"n_from + n_to {np.mean(noises[:-1] + noises[1:]):.4f}")
            print("   " + describe("raw d", distances, steps))
            print("   " + describe("raw chi", raw, steps))
            excess_positive = int(np.sum(excess > 0))
            print("   " + describe("chi excess", excess, steps) + f"   transitions above the floor: {excess_positive}")
            print("   " + describe("signal/noise", ratio, steps) + f"   largest {np.nanmax(ratio):.1f}, "
                  f"second {np.sort(ratio[np.isfinite(ratio)])[-2]:.1f}")


if __name__ == "__main__":
    main()
