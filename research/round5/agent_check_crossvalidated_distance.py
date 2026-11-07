"""
Scratch check for the D50 lateral review (not a repo result).

Cross-validated ("crossnobis-style") subspace distance on the round-4 S7 set-up:
    d_cv = < P_from(A) - P_to(A) , P_from(B) - P_to(B) >_F / (2k)
with A and B two disjoint halves of the 8 token blocks (same blocks at both checkpoints),
averaged over random balanced splits. Plain d = ||P_from - P_to||_F^2 / (2k) on all blocks.
Also the lag structure: d and d_cv between checkpoints t and t + m, m = 1..5.
"""

import sys
from pathlib import Path

import numpy as np

ROUND4 = Path("/home/claude/pg/repo/neuron-diffusion-spectroscopy/research/round4")
sys.path.insert(0, str(ROUND4))

import sim7_shared_probe_paired_bootstrap as sim7
from sims_common import MODULE_NEURONS, MODULES, NEURONS

SPLITS = 12


def projector(vectors):
    q, _ = np.linalg.qr(vectors)
    return q @ q.T


def build_moments(fraction, seed, checkpoints=16, event=8):
    generator = np.random.default_rng(900 + seed)
    labels = np.full(NEURONS, -1)
    labels[:MODULE_NEURONS] = np.repeat(np.arange(MODULES), MODULE_NEURONS // MODULES)
    labels_after = labels.copy()
    if fraction > 0:
        count = max(1, int(round(fraction * MODULE_NEURONS)))
        moved = generator.permutation(MODULE_NEURONS)[:count]
        labels_after[moved] = (labels_after[moved] + 1 + generator.integers(0, MODULES - 1, len(moved))) % MODULES
    latents = generator.standard_normal((sim7.TOKENS, MODULES))
    private = generator.standard_normal((sim7.TOKENS, NEURONS))
    free = generator.standard_normal((sim7.TOKENS, NEURONS))
    moments = []
    for index in range(checkpoints):
        if index > 0:
            for array in (latents, private, free):
                array *= np.sqrt(1 - sim7.DRIFT)
                array += np.sqrt(sim7.DRIFT) * generator.standard_normal(array.shape)
        current = labels_after if index >= event else labels
        values = np.sqrt(1 - sim7.STRENGTH) * private
        inside = current >= 0
        values[:, inside] += np.sqrt(sim7.STRENGTH) * latents[:, current[inside]]
        values[:, ~inside] += np.sqrt(sim7.STRENGTH) * free[:, ~inside]
        moments.append(sim7.blocked(values))
    return moments, generator


def distances(moment_from, moment_to, generator):
    vectors_from, count = sim7.module_subspace(moment_from.merged().covariance())
    vectors_to, _ = sim7.module_subspace(moment_to.merged().covariance(), count)
    plain = np.sum((projector(vectors_from) - projector(vectors_to)) ** 2) / (2 * count)
    crossed = []
    for _ in range(SPLITS):
        order = generator.permutation(sim7.BLOCKS)
        half_a = order[: sim7.BLOCKS // 2]
        half_b = order[sim7.BLOCKS // 2:]
        difference = []
        for half in (half_a, half_b):
            a, _ = sim7.module_subspace(moment_from.merged(half).covariance(), count)
            b, _ = sim7.module_subspace(moment_to.merged(half).covariance(), count)
            difference.append(projector(a) - projector(b))
        crossed.append(np.sum(difference[0] * difference[1]) / (2 * count))
    return plain, float(np.mean(crossed)), float(np.std(crossed, ddof=1))


def main():
    print("Neighbour transitions, event at 7 -> 8")
    print(f"{'moved':>8s} {'seed':>4s} | {'d event':>9s} {'d quiet med':>11s} | {'dcv event':>10s} {'dcv quiet med':>13s} {'dcv quiet min':>13s} {'dcv quiet max':>13s}")
    for fraction in [0.0, 1.0 / 192, 0.05]:
        for seed in range(2):
            moments, generator = build_moments(fraction, seed)
            plain_list = []
            crossed_list = []
            for index in range(len(moments) - 1):
                plain, crossed, _ = distances(moments[index], moments[index + 1], generator)
                plain_list.append(plain)
                crossed_list.append(crossed)
            plain_list = np.asarray(plain_list)
            crossed_list = np.asarray(crossed_list)
            quiet_plain = np.delete(plain_list, 7)
            quiet_crossed = np.delete(crossed_list, 7)
            print(f"{fraction:8.4f} {seed:4d} | {plain_list[7]:9.5f} {np.median(quiet_plain):11.5f} | "
                  f"{crossed_list[7]:10.5f} {np.median(quiet_crossed):13.6f} {np.min(quiet_crossed):13.6f} {np.max(quiet_crossed):13.6f}")

    print()
    print("Lag structure in the quiet run (no event): mean over start points t = 0..5")
    moments, generator = build_moments(0.0, 0, checkpoints=12, event=99)
    print(f"{'lag':>4s} | {'d':>9s} | {'d_cv':>10s}")
    for lag in range(1, 6):
        plain_values = []
        crossed_values = []
        for start in range(0, 6):
            plain, crossed, _ = distances(moments[start], moments[start + lag], generator)
            plain_values.append(plain)
            crossed_values.append(crossed)
        print(f"{lag:4d} | {np.mean(plain_values):9.5f} | {np.mean(crossed_values):10.6f}")


if __name__ == "__main__":
    main()
