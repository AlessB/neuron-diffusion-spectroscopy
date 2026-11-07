"""
R2 check 2 (22 Oct 2026): why <r> stayed GOE in check 1, and three readouts that do move.

  A. Cross-module coupling crossover (Rosenzweig-Porter-like): scale every cross-module kernel
     weight by eps and watch <r> and slow-mode IPR.
  B. Sparse kernels (alpha-decay k=5 d=40 with self loops; kNN-sparsified self-tuning k=10):
     <r>, exact-zero cross-module weights, components.
  C. IPR profile along the spectrum (median N*IPR/3 in index bins): a "mobility edge" picture.
  D. Random neuron subsampling: does slow-mode IPR scale like 1/N for both original and rotated?
  E. Toy multislice (M-PHATE-like) supra-graph over (checkpoint, neuron) on the avoided-crossing
     toy: which slow supra-modes are "temporal" (split early vs late) and which are "modular".

Run:  PYTHONPATH=/home/claude/pg/repo/neuron-diffusion-spectroscopy/src python3 check_r2_crossover.py
"""

import time

import numpy as np
from scipy.sparse.csgraph import connected_components

from nds.controls import random_orthogonal_matrix, rotate_covariance
from nds.graphs import (adaptive_gaussian_kernel, correlation_distance_squared,
                        correlation_from_covariance)
from nds.spectra import diffusion_spectrum
from nds.synthetic import three_module_path

from check_r2_gap_ratio import covariance_from_latents, gap_ratios


def kernel_of(cov, **kw):
    d2 = correlation_distance_squared(correlation_from_covariance(cov), signed=True)
    return adaptive_gaussian_kernel(d2, **kw)


def spectrum_numbers(kernel, n_slow):
    sp = diffusion_spectrum(kernel)
    mu, v = sp["laplacian_eigenvalues"], sp["symmetric_vectors"]
    m = len(mu)
    nipr = m * (v ** 4).sum(axis=0) / 3.0
    return mu, nipr, {
        "r_all": gap_ratios(mu[1:]).mean(),
        "r_bulk": gap_ratios(mu[m // 4: 3 * m // 4]).mean(),
        "slow_nIPR": nipr[1:n_slow].mean(),
        "bulk_nIPR_median": np.median(nipr[m // 4: 3 * m // 4]),
    }


def main():
    t0 = time.time()
    rng = np.random.default_rng(1)
    n, n_tokens = 2048, 20000
    sizes = [64] * 32
    labels = np.repeat(np.arange(32), 64)
    cov = covariance_from_latents(n_tokens, n, sizes, 0.7, rng)
    same = labels[:, None] == labels[None, :]
    q = random_orthogonal_matrix(n, np.random.default_rng(5))
    cov_rot = rotate_covariance(cov, q)

    print("== A. cross-module weights x eps (32 modules x 64, strength 0.7, self-tuning k=10) ==")
    k_full = kernel_of(cov, neighbour_rank=10)
    w_in, w_out = k_full[same & ~np.eye(n, dtype=bool)].mean(), k_full[~same].mean()
    print(f"mean kernel weight inside modules {w_in:.3e}, between modules {w_out:.3e}; "
          f"total between / total inside = {k_full[~same].sum() / k_full[same].sum():.2f}")
    for eps in (1.0, 0.3, 0.1, 0.03, 0.01, 1e-3, 1e-4, 0.0):
        k = np.where(same, k_full, eps * k_full)
        if eps == 0.0:
            # 32 disconnected pieces: mu = 0 is 32-fold degenerate; keep it, gaps of 0 are dropped
            pass
        _, _, s = spectrum_numbers(k, 33)
        print(f"eps {eps:7.4f}: r_all {s['r_all']:.3f} r_bulk {s['r_bulk']:.3f} | slow N*IPR/3 "
              f"{s['slow_nIPR']:.2f} | bulk median N*IPR/3 {s['bulk_nIPR_median']:.2f}")

    print("\n== B. sparse kernels, original vs rotated ==")
    for name, kw in [("self-tuning k=10 (dense)", dict(neighbour_rank=10)),
                     ("alpha-decay k=5 d=40 + self loops", dict(neighbour_rank=5, kernel_type="alpha_decay",
                                                               decay=40.0, keep_self_loops=True)),
                     ("alpha-decay k=10 d=40 + self loops", dict(neighbour_rank=10, kernel_type="alpha_decay",
                                                                decay=40.0, keep_self_loops=True))]:
        for tag, c in (("orig", cov), ("rot ", cov_rot)):
            k = kernel_of(c, **kw)
            ncomp = connected_components(k > 0, directed=False)[0]
            zero_frac = float((k[~same] == 0).mean())
            if ncomp > 1:
                print(f"{name:36s} {tag}: {ncomp} connected components -> skipped spectrum "
                      f"(cross-module weights exactly 0: {zero_frac:.3f})")
                continue
            _, _, s = spectrum_numbers(k, 33)
            print(f"{name:36s} {tag}: r_all {s['r_all']:.3f} r_bulk {s['r_bulk']:.3f} | slow N*IPR/3 "
                  f"{s['slow_nIPR']:.2f} | cross-module weights exactly 0: {zero_frac:.3f} | comps {ncomp}")
    # kNN-sparsified self-tuning: keep each neuron's 10 strongest links, symmetrise by max
    for tag, c in (("orig", cov), ("rot ", cov_rot)):
        k = kernel_of(c, neighbour_rank=10)
        keep = np.zeros_like(k, dtype=bool)
        top = np.argsort(-k, axis=1)[:, :10]
        keep[np.arange(n)[:, None], top] = True
        keep = keep | keep.T
        ks = np.where(keep, k, 0.0)
        ncomp = connected_components(ks > 0, directed=False)[0]
        if ncomp > 1:
            print(f"{'kNN-10 sparsified self-tuning':36s} {tag}: {ncomp} components -> skipped")
            continue
        _, _, s = spectrum_numbers(ks, 33)
        print(f"{'kNN-10 sparsified self-tuning':36s} {tag}: r_all {s['r_all']:.3f} r_bulk {s['r_bulk']:.3f} | "
              f"slow N*IPR/3 {s['slow_nIPR']:.2f} | comps {ncomp}")

    print("\n== C. IPR along the spectrum (median N*IPR/3 per eighth of the sorted modes) ==")
    for tag, c in (("orig", cov), ("rot ", cov_rot)):
        mu, nipr, _ = spectrum_numbers(kernel_of(c, neighbour_rank=10), 33)
        bins = np.array_split(np.arange(1, n), 8)
        print(f"{tag}: " + " ".join(f"[{mu[b[0]]:.3f}-{mu[b[-1]]:.3f}] {np.median(nipr[b]):5.2f}" for b in bins))

    print("\n== D. random neuron subsampling (modules shrink with N) ==")
    for n_sub in (512, 1024, 2048):
        keep = np.sort(np.random.default_rng(3).choice(n, n_sub, replace=False))
        sub = cov[np.ix_(keep, keep)]
        sub_rot = rotate_covariance(sub, random_orthogonal_matrix(n_sub, np.random.default_rng(4)))
        _, nipr_o, _ = spectrum_numbers(kernel_of(sub, neighbour_rank=10), 33)
        _, nipr_r, _ = spectrum_numbers(kernel_of(sub_rot, neighbour_rank=10), 33)
        print(f"N {n_sub:4d}: slow-mode IPR orig {3 * nipr_o[1:33].mean() / n_sub:.5f} "
              f"(N*IPR/3 {nipr_o[1:33].mean():.2f}) | rotated {3 * nipr_r[1:33].mean() / n_sub:.5f} "
              f"(N*IPR/3 {nipr_r[1:33].mean():.2f})")

    print("\n== E. toy multislice supra-graph: 12 slices x 120 neurons (avoided-crossing toy) ==")
    slices = 12
    positions = np.linspace(0, 1, slices)
    m = 120
    kernels, zs = [], []
    for p in positions:
        x, lab = three_module_path(p, number_of_tokens=3000)
        c = np.cov(x, rowvar=False)
        kernels.append(kernel_of(c, neighbour_rank=10))
        z = (x - x.mean(0)) / x.std(0)
        zs.append(z.T)                      # neurons x tokens, same tokens at every slice
    zs = np.stack(zs)                       # slices x neurons x tokens
    big = slices * m
    supra = np.zeros((big, big))
    for s, k in enumerate(kernels):
        supra[s * m:(s + 1) * m, s * m:(s + 1) * m] = k
    # interslice: M-PHATE-style Gaussian on each neuron's own z-vector across slices,
    # one fixed bandwidth = mean over neurons of the median 3rd-neighbour distance (toy has 12 slices)
    dists = np.stack([np.sqrt(((zs[:, v, None, :] - zs[None, :, v, :]) ** 2).sum(-1)) for v in range(m)])
    bw = np.mean([np.median(np.partition(d, 3, axis=1)[:, 3]) for d in dists])
    for v in range(m):
        a = np.exp(-(dists[v] / bw) ** 2)
        np.fill_diagonal(a, 0.0)
        idx = np.arange(slices) * m + v
        supra[np.ix_(idx, idx)] = a
    sp = diffusion_spectrum(supra)
    v = sp["symmetric_vectors"]
    mu = sp["laplacian_eigenvalues"]
    print(f"interslice bandwidth {bw:.2f}; slowest supra mu: {np.round(mu[:6], 4)}")
    for j in range(1, 5):
        w = v[:, j].reshape(slices, m)
        time_profile = (w ** 2).sum(axis=1)                   # weight per slice
        signed_slice = w.sum(axis=1)
        module_profile = [(w[:, lab == g] ** 2).sum() for g in range(3)]
        sign_flip = np.sign(signed_slice[0]) != np.sign(signed_slice[-1])
        print(f"supra mode {j}: weight per module A,B,C {np.round(module_profile, 2)} | "
              f"weight first/last third of slices {time_profile[:4].sum():.2f}/{time_profile[-4:].sum():.2f} | "
              f"slice-sum sign early vs late differs: {bool(sign_flip)}")
    # memory arithmetic for the real case (float64 values + int32 column indices = 12 bytes per nnz)
    print("\nmemory, sparse CSR at 12 bytes per stored weight:")
    for checkpoints in (25, 154):
        nodes = 2048 * checkpoints
        intra_dense = checkpoints * 2048 ** 2
        intra_knn = checkpoints * 2048 * 2 * 10
        inter_full = 2048 * checkpoints * (checkpoints - 1)
        dense_supra_gb = nodes ** 2 * 8 / 1e9
        print(f"  {checkpoints:3d} checkpoints: {nodes:,} nodes | dense supra matrix {dense_supra_gb:,.0f} GB | "
              f"intraslice dense {intra_dense * 12 / 1e9:.1f} GB, kNN-10 {intra_knn * 12 / 1e6:.0f} MB | "
              f"interslice all-pairs {inter_full * 12 / 1e6:.0f} MB")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
