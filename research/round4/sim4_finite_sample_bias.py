"""
S4. Finite-sample bias of the readouts, and whether it can be extrapolated away.

Why it matters: the number of effective tokens T_eff = T (1 - rho^2) / (1 + rho^2) depends on the
lag-one correlation rho, which changes over training. If a readout depends on T_eff (noise adds a
random component to every correlation), it drifts over training for a reason that has nothing to
do with the network. The ratio that matters is q = neurons / T_eff (Marchenko-Pastur). Pythia-70m:
2,048 neurons and T_eff somewhere around 50,000 to 260,000 tokens, so q from about 0.04 down to 0.008.

Numerical analysis: if the bias goes like c / T, then f(T) = f_inf + c / T, and
    f_inf ~ 2 f(T) - f(T / 2)          (Richardson extrapolation, one step; equivalently a
                                        jackknife bias correction with two halves)
where f(T / 2) is the mean of the readout on the two halves of the tokens.

Test: fixed structure (256 neurons, 8 modules of 24 plus 64 free, strength 0.3), T from 1,000 to
64,000 (q from 0.26 to 0.004). Readouts: module-aligned localisation (IPR, L1), module count, and
the rotated value of the IPR readout (one rotation).

    python research/round4/sim4_finite_sample_bias.py
"""

import numpy as np

from sims_common import MODULE_NEURONS
from sims_common import MODULES
from sims_common import NEURONS

from nds.analysis import merged_settings
from nds.analysis import spectrum_numbers
from nds.controls import random_orthogonal_matrix
from nds.controls import rotate_covariance

SETTINGS = merged_settings({"minimum_firing_rate": 0.0, "fixed_bandwidth_dse": False})


def readouts(values, rotation):
    covariance = np.cov(values, rowvar=False)
    spectrum = spectrum_numbers(covariance, None, SETTINGS)
    rotated = spectrum_numbers(rotate_covariance(covariance, rotation), None, SETTINGS)
    return np.array([
        spectrum["module_aligned_localization"],
        spectrum["module_aligned_support"],
        spectrum["module_count"],
        rotated["module_aligned_localization"],
    ])


def main():
    names = ["IPR", "L1", "k*", "IPR rotated"]
    print(f"{'T':>6s} {'q':>6s} | " + " ".join(f"{name:>11s}" for name in names) + " | " + " ".join(f"{'R:' + name:>11s}" for name in names[:2]))
    for tokens in [1000, 2000, 4000, 8000, 16000, 32000, 64000]:
        results = []
        corrected = []
        for seed in range(3):
            generator = np.random.default_rng(400 + seed)
            labels = np.full(NEURONS, -1)
            labels[:MODULE_NEURONS] = np.repeat(np.arange(MODULES), MODULE_NEURONS // MODULES)
            latents = generator.standard_normal((tokens, MODULES))
            values = np.sqrt(0.7) * generator.standard_normal((tokens, NEURONS))
            inside = labels >= 0
            values[:, inside] += np.sqrt(0.3) * latents[:, labels[inside]]
            values[:, ~inside] += np.sqrt(0.3) * generator.standard_normal((tokens, int(np.sum(~inside))))
            rotation = random_orthogonal_matrix(NEURONS, np.random.default_rng(7))

            full = readouts(values, rotation)
            half_a = readouts(values[: tokens // 2], rotation)
            half_b = readouts(values[tokens // 2:], rotation)
            results.append(full)
            corrected.append(2 * full[:2] - 0.5 * (half_a[:2] + half_b[:2]))
        mean = np.mean(results, axis=0)
        mean_corrected = np.mean(corrected, axis=0)
        print(f"{tokens:6d} {NEURONS / tokens:6.3f} | " + " ".join(f"{value:11.3f}" for value in mean) + " | " + " ".join(f"{value:11.3f}" for value in mean_corrected))


if __name__ == "__main__":
    main()
