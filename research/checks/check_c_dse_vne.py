"""
check_c_dse_vne.py
==================

Extra cross-check: our diffusion_spectral_entropy versus
    (1) the published DSE function (ChenLiu-1996/DiffusionSpectralEntropy, dse_dsmi/dse.py), and
    (2) PHATE's von Neumann entropy used to pick t (KrishnaswamyLab/PHATE, Python/phate/vne.py).

(1) dse_dsmi builds G = exp(-D^2 / (2 sigma^2)) on raw vectors with ONE bandwidth. On z-scored neuron
    profiles (population std, p tokens) D = sqrt(p) * d, so sigma = sqrt(p) * b reproduces our
    fixed_bandwidth_gaussian_kernel(d^2, bandwidth=b) up to the constant 1/(sigma sqrt(2 pi)), which
    cancels. The two DSE values should agree to rounding.
(2) PHATE's VNE uses the SINGULAR values of the row-stochastic P (not |eigenvalues|) and natural log.
    For P = D^-1 K the singular values differ from the eigenvalues unless all degrees are equal.

Run:  python3 -I check_c_dse_vne.py
"""

import importlib.util
import sys

sys.path[:0] = [
    "/home/claude/refcode/ChenLiu-1996__DiffusionSpectralEntropy",
    "/home/claude/pg/repo/neuron-diffusion-spectroscopy/src",
]

import numpy as np

from dse_dsmi import diffusion_spectral_entropy as published_dse

from nds.graphs import adaptive_gaussian_kernel
from nds.graphs import correlation_distance_squared
from nds.graphs import correlation_from_covariance
from nds.graphs import fixed_bandwidth_gaussian_kernel
from nds.spectra import diffusion_spectral_entropy
from nds.spectra import diffusion_spectrum
from nds.synthetic import modular_activations

spec = importlib.util.spec_from_file_location("phate_vne", "/home/claude/refcode/KrishnaswamyLab__PHATE/Python/phate/vne.py")
phate_vne = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phate_vne)


def main():
    activations, _ = modular_activations(random_generator=np.random.default_rng(0))
    p = activations.shape[0]

    correlation = correlation_from_covariance(np.cov(activations, rowvar=False))
    squared_distances = correlation_distance_squared(correlation)

    centred = activations - activations.mean(axis=0)
    zscored = (centred / centred.std(axis=0)).T                      # neurons x tokens

    off_diagonal = squared_distances[~np.eye(len(squared_distances), dtype=bool)]
    bandwidth = float(np.sqrt(np.median(off_diagonal)))
    fixed_kernel = fixed_bandwidth_gaussian_kernel(squared_distances, bandwidth=bandwidth)
    fixed_eigenvalues = diffusion_spectrum(fixed_kernel)["diffusion_eigenvalues"]

    print("(1) fixed-bandwidth kernel, ours vs dse_dsmi (sigma = sqrt(p) * b)")
    for t in (1, 2, 4, 8):
        ours = diffusion_spectral_entropy(fixed_eigenvalues, diffusion_time=t)
        theirs = published_dse(zscored, gaussian_kernel_sigma=np.sqrt(p) * bandwidth, t=t, max_N=None)
        print("   t={:>2}: ours {:.10f}  dse_dsmi {:.10f}  diff {:.1e}".format(t, ours, theirs, abs(ours - theirs)))

    print("   dse_dsmi with its default sigma=10 on the same z-scored vectors: {:.4f} (t=1)".format(
        published_dse(zscored, max_N=None)))

    kernel = adaptive_gaussian_kernel(squared_distances, neighbour_rank=10, decay=2.0, keep_self_loops=False)
    eigenvalues = diffusion_spectrum(kernel)["diffusion_eigenvalues"]
    P = kernel / kernel.sum(axis=1, keepdims=True)
    singular_values = np.linalg.svd(P, compute_uv=False)

    vne = phate_vne.compute_von_neumann_entropy(P, t_max=64)          # vne[i] uses sigma^(i+1)
    print()
    print("(2) adaptive kernel (k=10, decay=2): our DSE (|eig|, log2) vs PHATE VNE (singular values, ln)")
    print("   max |sigma_i - |lambda_i||, sorted: {:.3e}".format(
        np.abs(np.sort(singular_values) - np.sort(np.abs(eigenvalues))).max()))
    for t in (1, 2, 4, 8, 16, 32, 64):
        ours_bits = diffusion_spectral_entropy(eigenvalues, diffusion_time=t)
        print("   t={:>2}: DSE {:.4f} bits = {:.4f} nats | PHATE vne[t-1] {:.4f} nats".format(
            t, ours_bits, ours_bits * np.log(2), vne[t - 1]))

    knee = phate_vne.find_knee_point(vne, np.arange(64))
    print("   PHATE knee on x = arange(t_max) -> t_opt = {}, but that entry was computed from sigma^{}".format(
        knee, knee + 1))


if __name__ == "__main__":
    main()
