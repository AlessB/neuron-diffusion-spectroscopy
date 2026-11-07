"""
R2 check 4 (22 Oct 2026): heat-kernel readouts on synthetic neuron graphs.

  - Laplacian-RG specific heat C(tau) = tau^2 Var_{p_tau}(mu), p_tau(k) ~ exp(-tau mu_k)
    (Villegas et al. 2023), original vs rotated.
  - Graph diffusion distance (Hammond et al. 2013), on L_sym:
        GDD(G1, G2) = max_t || exp(-t L1) - exp(-t L2) ||_F
    between two "checkpoints" of the same neurons (module strength 0.3 -> 0.5), and for the
    rotated control with (a) the SAME rotation at both checkpoints, (b) a fresh rotation each time.

Run: PYTHONPATH=<repo>/src:<this dir> python3 check_r2_heat.py
"""

import numpy as np

from nds.controls import random_orthogonal_matrix, rotate_covariance
from nds.spectra import diffusion_spectrum

from check_r2_crossover import kernel_of
from check_r2_gap_ratio import covariance_from_latents


def specific_heat(mu, taus):
    out = []
    for tau in taus:
        w = np.exp(-tau * (mu - mu.min()))
        p = w / w.sum()
        mean = (p * mu).sum()
        out.append(tau ** 2 * (p * (mu - mean) ** 2).sum())
    return np.array(out)


def heat_kernel(spectrum, t):
    v, mu = spectrum["symmetric_vectors"], spectrum["laplacian_eigenvalues"]
    return (v * np.exp(-t * mu)) @ v.T


def gdd(s1, s2, ts):
    values = [np.linalg.norm(heat_kernel(s1, t) - heat_kernel(s2, t)) for t in ts]
    i = int(np.argmax(values))
    return values[i], ts[i]


def main():
    n, tokens = 1024, 20000
    sizes = [64] * 16
    taus = np.logspace(-1, 3, 81)
    ts = np.logspace(-1, 2.5, 36)

    # two "checkpoints": same latents and noise, module strength 0.3 then 0.5
    covs = [covariance_from_latents(tokens, n, sizes, s, np.random.default_rng(11)) for s in (0.3, 0.5)]
    q_fixed = random_orthogonal_matrix(n, np.random.default_rng(21))
    q_other = random_orthogonal_matrix(n, np.random.default_rng(22))

    spectra = {
        "orig": [diffusion_spectrum(kernel_of(c, neighbour_rank=10)) for c in covs],
        "rot same Q": [diffusion_spectrum(kernel_of(rotate_covariance(c, q_fixed), neighbour_rank=10)) for c in covs],
        "rot fresh Q": [diffusion_spectrum(kernel_of(rotate_covariance(covs[0], q_fixed), neighbour_rank=10)),
                        diffusion_spectrum(kernel_of(rotate_covariance(covs[1], q_other), neighbour_rank=10))],
    }

    print("== specific heat C(tau) at module strength 0.5 (1024 neurons, 16 modules x 64) ==")
    for name in ("orig", "rot same Q"):
        mu = spectra[name][1]["laplacian_eigenvalues"]
        c = specific_heat(mu, taus)
        peaks = [i for i in range(1, len(c) - 1) if c[i] > c[i - 1] and c[i] > c[i + 1]]
        desc = ", ".join(f"tau {taus[i]:.1f}: C {c[i]:.2f}" for i in peaks)
        print(f"{name:12s}: local maxima of C: {desc} | C(tau=1000) {c[-1]:.2f}")
        print(f"{'':12s}  slow mu 1..16: {np.round(mu[1:17], 3)}")

    print("\n== graph diffusion distance between strength 0.3 and 0.5 (same neurons) ==")
    for name, (s1, s2) in spectra.items():
        value, t_star = gdd(s1, s2, ts)
        print(f"{name:12s}: GDD {value:.3f} at t {t_star:.1f}")
    # scale reference: GDD between strength-0.5 original and its own token-bootstrap-like replicate
    cov_b = covariance_from_latents(tokens, n, sizes, 0.5, np.random.default_rng(12))
    sb = diffusion_spectrum(kernel_of(cov_b, neighbour_rank=10))
    value, t_star = gdd(spectra["orig"][1], sb, ts)
    print(f"{'noise ref':12s}: GDD {value:.3f} at t {t_star:.1f}  (strength 0.5 vs a fresh draw of the same "
          f"modules: same structure, new tokens; the size of pure sampling noise)")


if __name__ == "__main__":
    main()
