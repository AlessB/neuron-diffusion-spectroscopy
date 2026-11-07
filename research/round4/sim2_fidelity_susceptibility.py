"""
S2. Fidelity susceptibility: the quantum-phase-transition detector that needs no order parameter,
applied to the neuron graph.

Physics (Zanardi & Paunkovic 2006; Gu 2010): the overlap F between ground states at neighbouring
parameter values drops at a transition; for small steps 1 - F ~ (d lambda)^2 chi_F / 2, and the
fidelity susceptibility chi_F peaks at the critical point. Perturbatively
    chi_F = sum_n |<n| dH/d lambda |0>|^2 / (E_n - E_0)^2,
so chi_F is large where a gap closes: an avoided crossing is exactly where it peaks, with height
proportional to 1 / (minimum gap)^2.

Ours: the "ground state" is the module subspace (modes 0 .. k*-1) or a single slow mode; the
parameter is training time t = log10(step + 1). With our subspace distance d (1 - mean cos^2 of
the principal angles, the multi-state version of 1 - F^2),
    chi(t) = d(t_i, t_i+1) / (t_i+1 - t_i)^2.
Dividing by the squared step corrects for Pythia's uneven checkpoint spacing.

Tests
  A. planted runs (sims_common): stable, growing (smooth crossover), forming, reshuffle. Does
     chi peak at the planted event and stay flat otherwise?
  B. the avoided-crossing toy (synthetic.three_module_path) at 41 positions, for three direct
     couplings g. For the two-level model the peak of chi for the slowest mode is (v / 4g)^2, so
     peak height should scale like 1 / g^2 and sit at the gap minimum.

    python research/round4/sim2_fidelity_susceptibility.py
"""

import numpy as np

from sims_common import CHECKPOINTS
from sims_common import EVENT
from sims_common import compare
from sims_common import make_run
from sims_common import summarise

from nds.analysis import merged_settings
from nds.analysis import spectrum_numbers
from nds.synthetic import three_module_path


def planted():
    print("A. chi between neighbouring checkpoints (evenly spaced here, so chi is proportional to d)")
    print("   transition index i means checkpoint i -> i+1; planted events at transition", EVENT - 1)
    for scenario in ["stable", "growing", "forming", "reshuffle"]:
        rows = []
        for seed in range(3):
            generator = np.random.default_rng(200 + seed)
            covariances = make_run(scenario, generator)
            summaries = [summarise(c, np.arange(c.shape[0])) for c in covariances]
            distances = [compare(summaries[i - 1], summaries[i])["subspace_distance"] for i in range(1, CHECKPOINTS)]
            rows.append(distances)
        mean = np.mean(rows, axis=0)
        peak = int(np.argmax(mean))
        others = np.delete(mean, peak)
        print(f"   {scenario:10s} " + " ".join(f"{value:6.3f}" for value in mean) + f"   peak at {peak}, peak / median of the rest = {mean[peak] / max(np.median(others), 1e-9):7.1f}")


def avoided_crossing():
    print("\nB. avoided-crossing toy, 41 positions: slowest non-trivial mode")
    settings = merged_settings({"fixed_bandwidth_dse": False, "minimum_firing_rate": 0.0})
    positions = np.linspace(0.0, 1.0, 41)
    step = positions[1] - positions[0]
    for coupling in [0.05, 0.1, 0.2]:
        vectors = []
        gaps = []
        for position in positions:
            activations, _ = three_module_path(position, direct_coupling=coupling)
            spectrum = spectrum_numbers(np.cov(activations, rowvar=False), None, settings)
            values = spectrum["laplacian_eigenvalues"]
            vectors.append(spectrum["symmetric_vectors"][:, 1:3])
            gaps.append(values[2] - values[1])
        single = []
        pair = []
        for index in range(1, len(positions)):
            overlap = abs(float(vectors[index - 1][:, 0] @ vectors[index][:, 0]))
            single.append((1.0 - overlap ** 2) / step ** 2)
            cosines = vectors[index - 1].T @ vectors[index]
            pair.append((1.0 - np.sum(cosines ** 2) / 2.0) / step ** 2)
        single = np.asarray(single)
        pair = np.asarray(pair)
        gaps = np.asarray(gaps)
        peak = int(np.argmax(single))
        gap_minimum = int(np.argmin(gaps))
        print(f"   g = {coupling:4.2f}: min gap {gaps.min():.4f} at position {positions[gap_minimum]:.3f}; "
              f"chi(mode 1) peak {single.max():9.1f} at position {0.5 * (positions[peak] + positions[peak + 1]):.3f}; "
              f"peak x gap^2 = {single.max() * gaps.min() ** 2:.4f}; chi(modes 1-2 together) max {pair.max():.2f}")


if __name__ == "__main__":
    planted()
    avoided_crossing()
