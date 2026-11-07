"""
S5. Is float32 accurate enough for the per-batch statistics?

Collection (activations.summarise_on_device) computes, per batch of 16 x 255 = 4,080 tokens on
the GPU in float32: the batch mean, then the scatter of the centred batch (X - mean)^T (X - mean).
Batches are merged in float64 on the CPU with the Chan-Golub-LeVeque update (moments.py). The
GTX 1060 (Pascal) is slow in float64, so float32 on the GPU is the plan (D23).

Numerical analysis: summing n products in float32 has relative error up to about n * 2^-24 in the
worst case and about sqrt(n) * 2^-24 typically (Higham, "Accuracy and Stability of Numerical
Algorithms", ch. 3). Centring first avoids the catastrophic cancellation of E[x^2] - E[x]^2 when a
neuron's mean is large compared with its spread (Chan, Golub & LeVeque 1983).

Test: 64 batches of 4,080 tokens (261,120 tokens, the probe size), 256 neurons with realistic
post-GELU trouble: a slow modular signal, neurons with a large mean and tiny spread (always-on
neurons), heavy tails, and a few "massive" neurons that are huge on 0.1% of tokens. Compared:
  float64 everywhere (reference)
  float32 per batch with centring (what the code does)
  float32 per batch without centring (what it would be without the CGL trick)
Reported: the largest error in any correlation, and in the resulting module-aligned localisation.

    python research/round4/sim5_float32_accumulation.py
"""

import numpy as np

import sims_common  # noqa: F401  (puts src/ on the path)

from nds.analysis import merged_settings
from nds.analysis import spectrum_numbers
from nds.moments import NeuronMoments

NEURONS = 256
BATCH = 4080
BATCHES = 64

SETTINGS = merged_settings({"minimum_firing_rate": 0.0, "fixed_bandwidth_dse": False})


def make_batch(generator):
    latents = generator.standard_normal((BATCH, 8))
    labels = np.repeat(np.arange(8), 32)
    values = 0.5 * latents[:, labels] + generator.standard_normal((BATCH, NEURONS))
    values = values * (values > -0.3)
    values[:, :16] = 40.0 + 0.01 * generator.standard_normal((BATCH, 16)) + 0.002 * latents[:, [0]]
    values[:, 16:32] = values[:, 16:32] * np.exp(generator.standard_normal((BATCH, 16)))
    spikes = generator.random((BATCH, 8)) < 0.001
    values[:, 32:40] = values[:, 32:40] + 3000.0 * spikes
    return values


def correlation(covariance):
    scale = np.sqrt(np.diag(covariance))
    return covariance / np.outer(scale, scale)


def main():
    generator = np.random.default_rng(5)
    reference = NeuronMoments(NEURONS)
    centred32 = NeuronMoments(NEURONS)
    raw_sum = np.zeros((NEURONS, NEURONS), dtype=np.float64)
    raw_mean_sum = np.zeros(NEURONS)
    count = 0

    for _ in range(BATCHES):
        batch = make_batch(generator)
        reference.add_batch(batch)

        batch32 = batch.astype(np.float32)
        mean32 = batch32.mean(axis=0)
        centred = batch32 - mean32
        scatter32 = centred.T @ centred
        centred32.add_batch_summary(
            BATCH,
            mean32.astype(np.float64),
            scatter32.astype(np.float64),
            np.sum(batch32 > 0, axis=0),
            np.abs(batch32).max(axis=0).astype(np.float64),
        )

        raw_sum += (batch32.T @ batch32).astype(np.float64)
        raw_mean_sum += batch32.sum(axis=0, dtype=np.float32).astype(np.float64)
        count += BATCH

    truth = correlation(reference.covariance())
    centred_result = correlation(centred32.covariance())
    mean = raw_mean_sum / count
    raw_covariance = (raw_sum - count * np.outer(mean, mean)) / (count - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        raw_result = correlation(raw_covariance)

    for name, result in [("float32, centred per batch (the code)", centred_result), ("float32, not centred", raw_result)]:
        error = np.abs(result - truth)
        finite = np.isfinite(error)
        print(f"{name:38s}: largest correlation error {np.nanmax(error):.2e} "
              f"(always-on neurons {np.nanmax(error[:16, :]):.2e}, massive {np.nanmax(error[32:40, :]):.2e}); "
              f"non-finite entries {int(np.sum(~finite))}")

    loc_truth = spectrum_numbers(reference.covariance(), None, SETTINGS)["module_aligned_localization"]
    loc_32 = spectrum_numbers(centred32.covariance(), None, SETTINGS)["module_aligned_localization"]
    print(f"module-aligned localisation: float64 {loc_truth:.6f}, float32 centred {loc_32:.6f}")


if __name__ == "__main__":
    main()
