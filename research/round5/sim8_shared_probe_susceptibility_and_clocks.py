"""
S8 (D50 follow-up). Does the shared probe text rescue the fidelity susceptibility, and how much
does the choice of clock matter?

S6 found chi = d / dt^2 useless on Pythia's spacing, but S6 drew FRESH tokens at every checkpoint.
The real pipeline uses the same probe text (common random numbers, S7), so the noise that blew up
under 1 / dt^2 is mostly gone. In finance the same thing is known for finite-difference
sensitivities ("bump and revalue"): with independent draws the variance of (f(x + h) - f(x)) / h
grows like 1 / h^2, with common random numbers it stays bounded for smooth payoffs (Glasserman,
Monte Carlo Methods in Financial Engineering, the chapter on estimating sensitivities; checked
only through secondary sources here).

What is left is real, smooth change of the network between checkpoints. Its size depends on the
clock: if the structure drifts smoothly in log(step), d ~ (drift rate x dt)^2 and chi is flat; if it
drifts in proportion to the number of steps (the late checkpoints are 1,000 steps apart, a lot of
training), d grows late and so does chi. The fidelity susceptibility is a metric: it depends on
the parametrisation (Gu 2010), which is why physicists quote it per unit of the coupling they
tune. Here the "coupling" is training time, and there is no unique choice of clock.

Test: Pythia's 37-checkpoint schedule from S6, the same 4,000 tokens at every checkpoint, strength
0.15, a reshuffle of half the module neurons between steps 2,000 and 3,000. The network drifts
between checkpoints by mixing in fresh variation of size
    log clock:     rate x (difference in log10(step + 1))
    step clock:    rate x (difference in steps) / 143,000
Readouts: raw d, chi = d / dt^2 (dt in log10(step + 1)), and d over the paired bootstrap standard
deviation (S7). 3 seeds per clock.

    python research/round5/sim8_shared_probe_susceptibility_and_clocks.py
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
from nds.spectra import chordal_subspace_distance
from nds.spectra import diffusion_spectrum
from nds.spectra import eigengap_module_count
from nds.spectra import leading_eigenvectors
from nds.spectra import symmetric_diffusion_operator
from nds.timeline import fidelity_susceptibility
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


def module_subspace(covariance, count=None):
    graph = neuron_graph_from_covariance(covariance, minimum_firing_rate=0.0)
    operator, _ = symmetric_diffusion_operator(graph["kernel"])
    if count is None:
        spectrum = diffusion_spectrum(graph["kernel"])
        found, _, _ = eigengap_module_count(spectrum["laplacian_eigenvalues"])
        count = max(int(found), 2)
        return spectrum["symmetric_vectors"][:, :count], count
    return leading_eigenvectors(operator, count), count


def blocked(values):
    moments = BlockedNeuronMoments(NEURONS, number_of_blocks=BLOCKS)
    size = TOKENS // BLOCKS
    for block in range(BLOCKS):
        moments.add_batch(values[block * size:(block + 1) * size], block)
    return moments


def run(clock, rate, seed, steps):
    generator = np.random.default_rng(800 + seed)
    labels = np.full(NEURONS, -1)
    labels[:MODULE_NEURONS] = np.repeat(np.arange(MODULES), MODULE_NEURONS // MODULES)
    labels_after = labels.copy()
    moved = generator.permutation(MODULE_NEURONS)[: MODULE_NEURONS // 2]
    labels_after[moved] = generator.permutation(labels_after[moved])

    latents = generator.standard_normal((TOKENS, MODULES))
    private = generator.standard_normal((TOKENS, NEURONS))
    free = generator.standard_normal((TOKENS, NEURONS))

    times = training_time(steps)
    moments = []
    for index, step in enumerate(steps):
        if index > 0:
            if clock == "log":
                amount = rate * (times[index] - times[index - 1])
            else:
                amount = rate * (steps[index] - steps[index - 1]) / 143000.0
            mix = min(amount, 1.0)
            latents = np.sqrt(1 - mix) * latents + np.sqrt(mix) * generator.standard_normal(latents.shape)
            private = np.sqrt(1 - mix) * private + np.sqrt(mix) * generator.standard_normal(private.shape)
            free = np.sqrt(1 - mix) * free + np.sqrt(mix) * generator.standard_normal(free.shape)
        current = labels
        if step >= EVENT_STEP:
            current = labels_after
        values = np.sqrt(1 - STRENGTH) * private
        inside = current >= 0
        values[:, inside] += np.sqrt(STRENGTH) * latents[:, current[inside]]
        values[:, ~inside] += np.sqrt(STRENGTH) * free[:, ~inside]
        moments.append(blocked(values))

    subspaces = []
    counts = []
    for moment in moments:
        vectors, count = module_subspace(moment.merged().covariance())
        subspaces.append(vectors)
        counts.append(count)

    distances = []
    spreads = []
    for index in range(len(steps) - 1):
        distances.append(chordal_subspace_distance(subspaces[index], subspaces[index + 1]))
        replicate_distances = []
        for _ in range(REPLICATES):
            chosen = generator.integers(0, BLOCKS, BLOCKS)
            vectors_from, _ = module_subspace(moments[index].merged(chosen).covariance(), counts[index])
            vectors_to, _ = module_subspace(moments[index + 1].merged(chosen).covariance(), counts[index + 1])
            replicate_distances.append(chordal_subspace_distance(vectors_from, vectors_to))
        spreads.append(np.std(replicate_distances, ddof=1))
    return np.asarray(distances), np.asarray(spreads)


def describe(name, values, steps, event):
    finite = np.where(np.isfinite(values), values, -np.inf)
    peak = int(np.argmax(finite))
    rest = np.delete(values, event)
    rest = rest[np.isfinite(rest)]
    contrast = values[event] / np.median(rest)
    second = np.max(rest) / np.median(rest)
    return (f"{name:>14s}: peak at {steps[peak]:>6d}->{steps[peak + 1]:<6d} event/median {contrast:9.1f}, "
            f"largest other/median {second:8.1f}")


def main():
    steps = pythia_like_steps()
    times = training_time(steps)
    event = int(np.searchsorted(steps, EVENT_STEP)) - 1
    for clock, rate in [("log", 0.02), ("step", 2.0)]:
        print(f"\n===== drift on the {clock} clock (rate {rate}); shared probe text =====")
        for seed in range(3):
            distances, spreads = run(clock, rate, seed, steps)
            chi = fidelity_susceptibility(distances, times)
            z = distances / spreads
            print(f"seed {seed}: d quiet median {np.median(np.delete(distances, event)):.5f}, d at event {distances[event]:.4f}")
            print("   " + describe("raw d", distances, steps, event))
            print("   " + describe("chi = d/dt^2", chi, steps, event))
            print("   " + describe("d / paired sd", z, steps, event))


if __name__ == "__main__":
    main()
