"""
S10 (D50). The cross-validated distance d_cv on Pythia's checkpoint spacing, with the things the
D50 review listed as unchecked: a change of module count, smooth structural drift, and how noisy
d_cv itself is.

Set-up: 37 checkpoints on Pythia's schedule (as S6/S8), 256 neurons, 8 modules of 24 plus 64 free,
strength 0.2, the SAME 4,000 tokens at every checkpoint (8 blocks), two kinds of slow change:
  token jitter      the activations drift on the log clock (rate 0.02 per decade), structure unchanged
  structural drift  every module neuron leaks out of its module along a private direction, the
                    angle growing smoothly in log-time to 0.35 rad at the end (real, slow change)
and three events:
  A  step 2,000 -> 3,000:     one neuron of 192 changes module
  B  step 20,000 -> 30,000:   a new module of 24 free neurons is born (k* changes from 8 to 9)
  C  step 141,000 -> 142,000: 5% of module neurons change module (an event late, where checkpoints
                              are 0.003 decades apart)

Readouts per transition (nds.analysis.crossvalidated_subspace_change, fixed even/odd block split, k
fixed per pair): plain d, d_cv, and d_cv / dt^2 (the fidelity susceptibility on the unbiased
distance). And, for the noise of d_cv: the same pair with the other natural split (blocks 0-3 vs
4-7). 2 seeds.

    python research/round5/sim10_crossvalidated_on_pythia_spacing.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from nds.analysis import crossvalidated_subspace_change
from nds.analysis import merged_settings
from nds.moments import BlockedNeuronMoments
from nds.spectra import crossvalidated_projector_distance
from nds.analysis import module_subspace_of
from nds.timeline import training_time

NEURONS = 256
MODULE_NEURONS = 192
MODULES = 8
TOKENS = 4000
BLOCKS = 8
STRENGTH = 0.2
SETTINGS = merged_settings({"minimum_firing_rate": 0.0})


def steps_schedule():
    steps = [0] + [2 ** power for power in range(10)]
    steps += list(range(1000, 10001, 1000))
    steps += list(range(20000, 140001, 10000))
    steps += [141000, 142000, 143000]
    return np.asarray(steps)


def blocked(values):
    moments = BlockedNeuronMoments(NEURONS, number_of_blocks=BLOCKS)
    size = TOKENS // BLOCKS
    for block in range(BLOCKS):
        moments.add_batch(values[block * size:(block + 1) * size], block)
    return moments


def other_split(moment_from, moment_to, count):
    halves = [[0, 1, 2, 3], [4, 5, 6, 7]]
    estimates = []
    for half in halves:
        a, ids_a, _ = module_subspace_of(moment_from.merged(half), SETTINGS, count)
        b, ids_b, _ = module_subspace_of(moment_to.merged(half), SETTINGS, count)
        estimates.extend([a, b])
    return crossvalidated_projector_distance(estimates[0], estimates[1], estimates[2], estimates[3])[0]


def run(seed, steps):
    generator = np.random.default_rng(1000 + seed)
    labels = np.full(NEURONS, -1)
    labels[:MODULE_NEURONS] = np.repeat(np.arange(MODULES), MODULE_NEURONS // MODULES)
    after_a = labels.copy()
    after_a[0] = 3
    after_c = after_a.copy()
    moved = generator.permutation(MODULE_NEURONS)[: int(0.05 * MODULE_NEURONS)]
    after_c[moved] = (after_c[moved] + 1 + generator.integers(0, MODULES - 1, len(moved))) % MODULES
    newborn = np.arange(MODULE_NEURONS, MODULE_NEURONS + 24)

    latents = generator.standard_normal((TOKENS, MODULES + 1))
    private = generator.standard_normal((TOKENS, NEURONS))
    leak = generator.standard_normal((TOKENS, NEURONS))
    times = training_time(steps)
    moments = []
    for index, step in enumerate(steps):
        if index > 0:
            mix = min(0.02 * (times[index] - times[index - 1]), 1.0)
            latents = np.sqrt(1 - mix) * latents + np.sqrt(mix) * generator.standard_normal(latents.shape)
            private = np.sqrt(1 - mix) * private + np.sqrt(mix) * generator.standard_normal(private.shape)
            leak = np.sqrt(1 - mix) * leak + np.sqrt(mix) * generator.standard_normal(leak.shape)
        current = labels
        if step >= 3000:
            current = after_a
        if step >= 142000:
            current = after_c
        angle = 0.35 * times[index] / times[-1]
        values = np.sqrt(1 - STRENGTH) * private
        inside = current >= 0
        module_part = np.cos(angle) * latents[:, current[inside]] + np.sin(angle) * leak[:, inside]
        values[:, inside] += np.sqrt(STRENGTH) * module_part
        free = ~inside
        values[:, free] += np.sqrt(STRENGTH) * leak[:, free]
        if step >= 30000:
            values[:, newborn] = np.sqrt(1 - STRENGTH) * private[:, newborn] + np.sqrt(STRENGTH) * latents[:, [MODULES]]
        moments.append(blocked(values))

    rows = []
    for index in range(len(steps) - 1):
        result = crossvalidated_subspace_change(moments[index], moments[index + 1], SETTINGS)
        second = other_split(moments[index], moments[index + 1], result["module_count"])
        top = int(result["neuron_ids"][np.argmax(result["per_neuron"])])
        rows.append((result["plain"], result["d_cv"], second, result["module_count"], top))
    return np.array([row[:4] for row in rows]), [row[4] for row in rows]


def main():
    steps = steps_schedule()
    times = training_time(steps)
    dt = np.diff(times)
    events = {"A (1 neuron)": int(np.searchsorted(steps, 3000)) - 1, "B (birth, k 8->9)": int(np.searchsorted(steps, 30000)) - 1, "C (5% late)": int(np.searchsorted(steps, 142000)) - 1}
    for seed in range(2):
        table, tops = run(seed, steps)
        plain = table[:, 0]
        crossed = table[:, 1]
        second = table[:, 2]
        chi = crossed / dt ** 2
        quiet = np.ones(len(plain), dtype=bool)
        for index in events.values():
            quiet[index] = False
        print(f"\n===== seed {seed} =====")
        print(f"quiet transitions: plain d median {np.median(plain[quiet]):.5f} (max {np.max(plain[quiet]):.5f}); "
              f"d_cv median {np.median(crossed[quiet]):.6f} (min {np.min(crossed[quiet]):.6f}, max {np.max(crossed[quiet]):.6f})")
        print(f"   two splits disagree on d_cv by (median |difference|) {np.median(np.abs(crossed - second)[quiet]):.6f}")
        early = quiet & (np.arange(len(plain)) < 10)
        late = quiet & (np.arange(len(plain)) > 25)
        print(f"   quiet d_cv early (steps < 1,000) median {np.median(crossed[early]):.6f}; late (> 50,000) median {np.median(crossed[late]):.6f}")
        print(f"   quiet d_cv / dt^2: early median {np.median(chi[early]):.4f}, late median {np.median(chi[late]):.4f}")
        for name, index in events.items():
            print(f"   event {name:18s} {steps[index]}->{steps[index + 1]}: plain {plain[index]:.5f}, d_cv {crossed[index]:.5f} "
                  f"(second split {second[index]:.5f}), d_cv / largest quiet d_cv {crossed[index] / np.max(crossed[quiet]):.1f}, "
                  f"k used {int(table[index, 3])}, top neuron by c_i: {tops[index]}")


if __name__ == "__main__":
    main()
