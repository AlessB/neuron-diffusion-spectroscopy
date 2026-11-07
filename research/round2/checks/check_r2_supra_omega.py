"""
R2 check 3 (22 Oct 2026): toy multislice supra-graph, interslice weight scaled by omega.
Which slow supra-modes are "temporal" (constant within a slice, varying across slices) and which
are "modular" (varying across neurons within a slice)? Radicchi-Arenas-type crossover.
Run: PYTHONPATH=<repo>/src:<this dir> python3 check_r2_supra_omega.py
"""
import numpy as np
from nds.spectra import diffusion_spectrum
from nds.synthetic import three_module_path
from check_r2_crossover import kernel_of

slices, m = 12, 120
kernels, zs = [], []
for p in np.linspace(0, 1, slices):
    # same tokens and same latent/noise draws at every slice (like one probe text at every checkpoint)
    x, lab = three_module_path(p, number_of_tokens=3000, random_generator=np.random.default_rng(1))
    c = np.cov(x, rowvar=False)
    kernels.append(kernel_of(c, neighbour_rank=10))
    zs.append(((x - x.mean(0)) / x.std(0)).T)
zs = np.stack(zs)
dists = np.stack([np.sqrt(((zs[:, v, None, :] - zs[None, :, v, :]) ** 2).sum(-1)) for v in range(m)])
bw = np.mean([np.median(np.partition(d, 3, axis=1)[:, 3]) for d in dists])
inter = [np.exp(-(d / bw) ** 2) for d in dists]
for a in inter:
    np.fill_diagonal(a, 0.0)
print(f"mean intraslice weight {np.mean([k.mean() for k in kernels]):.3f}, mean interslice weight {np.mean([a.mean() for a in inter]):.3f}")
for omega in (0.001, 0.01, 0.1, 1.0, 3.0, 10.0, 30.0):
    big = slices * m
    supra = np.zeros((big, big))
    for s, k in enumerate(kernels):
        supra[s * m:(s + 1) * m, s * m:(s + 1) * m] = k
    for v in range(m):
        idx = np.arange(slices) * m + v
        supra[np.ix_(idx, idx)] = omega * inter[v]
    sp = diffusion_spectrum(supra)
    vec, mu = sp["symmetric_vectors"], sp["laplacian_eigenvalues"]
    desc = []
    for j in range(1, 17):
        w = vec[:, j].reshape(slices, m)
        temporal = (m * (w.mean(axis=1) ** 2).sum()) / (w ** 2).sum()   # share of the mode that is a slice mean
        module_means = np.array([w[:, lab == g].mean(axis=1) for g in range(3)])  # 3 x slices
        modular = (40 * (module_means ** 2).sum() - m * (w.mean(axis=1) ** 2).sum()) / (w ** 2).sum()
        desc.append(f"mu {mu[j]:.4f} T {temporal:.2f} M {modular:.2f}")
    first_modular = next((j for j in range(1, 60) if (lambda w: (40 * (np.array([w[:, lab == g].mean(axis=1) for g in range(3)]) ** 2).sum() - m * (w.mean(axis=1) ** 2).sum()) / (w ** 2).sum())(vec[:, j].reshape(slices, m)) > 0.5), None)
    print(f"omega {omega:6.3f}: first mode with M > 0.5 is mode {first_modular} (mu {mu[first_modular]:.4f}); modes 1-16:")
    print("    " + "\n    ".join(desc))
print("T = share of the mode's squared weight explained by slice means (temporal);")
print("M = extra share explained by module-by-slice means (modular). 1 - T - M = within-module detail.")
