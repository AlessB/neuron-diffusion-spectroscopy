"""
R2 check (22 Oct 2026): adjacent-gap ratio <r> of neuron-graph Laplacian spectra,
modular vs randomly rotated, plus IPR and finite-size scaling of the slow modes.

Run:  PYTHONPATH=/home/claude/pg/repo/neuron-diffusion-spectroscopy/src python3 check_r2_gap_ratio.py
Uses our own nds code for covariance -> kernel -> spectrum -> rotation.

r_n = min(s_n, s_{n+1}) / max(s_n, s_{n+1}),  s_n = e_{n+1} - e_n  (Oganesyan & Huse 2007;
reference values Atas et al. 2013: Poisson 2 ln 2 - 1 = 0.3863, GOE 0.5307 (numerics)).
"""

import time

import numpy as np

from nds.controls import rotate_live_covariance
from nds.graphs import neuron_graph_from_covariance
from nds.spectra import diffusion_spectrum


def gap_ratios(eigenvalues):
    e = np.sort(np.asarray(eigenvalues))
    s = np.diff(e)
    a, b = s[:-1], s[1:]
    big = np.maximum(a, b)
    ok = big > 0
    return np.minimum(a, b)[ok] / big[ok]


def window_r(mu, lo_frac, hi_frac):
    """<r> over Laplacian eigenvalues with sorted index in [lo_frac, hi_frac) of the spectrum."""
    n = len(mu)
    i0, i1 = int(lo_frac * n), int(hi_frac * n)
    return float(gap_ratios(mu[i0:i1]).mean())


def covariance_from_latents(n_tokens, n_neurons, module_sizes, strength, rng, heavy=False):
    """Modules of neurons listening to one latent each; neurons beyond the modules are pure noise.
    heavy=True passes everything through a GELU-like rectifier to mimic post-GELU sparsity."""
    n_mod = len(module_sizes)
    latents = rng.standard_normal((n_tokens, n_mod)).astype(np.float32)
    x = rng.standard_normal((n_tokens, n_neurons)).astype(np.float32)
    start = 0
    for m, size in enumerate(module_sizes):
        load = (0.7 + 0.6 * rng.random(size)).astype(np.float32)
        x[:, start:start + size] = (np.sqrt(strength) * latents[:, [m]] * load
                                    + np.sqrt(1 - strength) * x[:, start:start + size])
        start += size
    if heavy:
        x = x * 0.5 * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))  # GELU
    x = x - x.mean(axis=0)
    cov = (x.T.astype(np.float64) @ x.astype(np.float64)) / (n_tokens - 1)
    return cov


def readouts(cov, n_slow):
    g = neuron_graph_from_covariance(cov, neighbour_rank=10)
    sp = diffusion_spectrum(g["kernel"])
    mu = sp["laplacian_eigenvalues"]
    v = sp["symmetric_vectors"]
    m = len(mu)
    ipr = (v ** 4).sum(axis=0)
    out = {
        "r_all": float(gap_ratios(mu[1:]).mean()),
        "r_bulk": window_r(mu, 0.25, 0.75),
        "r_slow50": float(gap_ratios(mu[1:51]).mean()),
        "nIPR_slow": float(m * ipr[1:n_slow].mean() / 3.0),
        "nIPR_bulk": float(m * ipr[m // 4: 3 * m // 4].mean() / 3.0),
        "r_cov": float(gap_ratios(np.linalg.eigvalsh(cov)).mean()),
        "mu_slow": mu[1:n_slow],
    }
    return out


def fmt(d):
    return (f"r_all {d['r_all']:.3f} | r_bulk {d['r_bulk']:.3f} | r_slow50 {d['r_slow50']:.3f} | "
            f"N*IPR/3 slow {d['nIPR_slow']:.2f} bulk {d['nIPR_bulk']:.2f} | r_cov {d['r_cov']:.3f}")


def main():
    rng = np.random.default_rng(0)
    t0 = time.time()

    print("== references ==")
    n = 2048
    print(f"Poisson (sorted uniforms)      <r> = {gap_ratios(rng.random(n)).mean():.4f}   (theory 0.3863)")
    goe = rng.standard_normal((n, n)); goe = (goe + goe.T) / 2
    print(f"GOE {n}                       <r> = {gap_ratios(np.linalg.eigvalsh(goe)).mean():.4f}   (Atas 0.5307)")
    for k in (2, 8, 32):
        blocks = []
        for _ in range(k):
            b = rng.standard_normal((n // k, n // k)); b = (b + b.T) / 2
            blocks.append(np.linalg.eigvalsh(b) / np.sqrt(n // k))
        print(f"superposition of {k:2d} independent GOE blocks <r> = {gap_ratios(np.concatenate(blocks)).mean():.4f}")
    # sampling spread of <r> for a single GOE spectrum of 1023 levels (one bulk window)
    rs = []
    for _ in range(20):
        a = rng.standard_normal((1024, 1024)); a = (a + a.T) / 2
        rs.append(gap_ratios(np.linalg.eigvalsh(a)).mean())
    print(f"spread of <r> over 20 GOE(1024) draws: mean {np.mean(rs):.4f} sd {np.std(rs):.4f}")
    rs = []
    for _ in range(200):
        a = rng.standard_normal((52, 52)); a = (a + a.T) / 2
        rs.append(gap_ratios(np.linalg.eigvalsh(a)).mean())
    print(f"spread of <r> over 200 GOE(52) draws (50 ratios, like r_slow50): mean {np.mean(rs):.4f} sd {np.std(rs):.4f}")

    n_tokens = 20000
    configs = [
        ("no modules (pure noise)", [], 0.0, False),
        ("8 modules x 64, strength 0.7, rest noise", [64] * 8, 0.7, False),
        ("32 modules x 64 (every neuron), strength 0.7", [64] * 32, 0.7, False),
        ("32 modules x 64, strength 0.2", [64] * 32, 0.2, False),
        ("32 modules x 64, strength 0.7, GELU", [64] * 32, 0.7, True),
    ]
    print(f"\n== neuron graphs: {n} neurons, {n_tokens} tokens, self-tuning kernel k=10 ==")
    for name, sizes, strength, heavy in configs:
        cov = covariance_from_latents(n_tokens, n, sizes, strength, rng, heavy)
        n_slow = max(len(sizes), 1) + 1
        orig = readouts(cov, n_slow)
        print(f"{name}")
        print(f"   original : {fmt(orig)}")
        for rep in range(2):
            is_dead = np.zeros(n, dtype=bool)
            rot = readouts(rotate_live_covariance(cov, is_dead, np.random.default_rng(100 + rep)), n_slow)
            print(f"   rotated {rep}: {fmt(rot)}")
        print(f"   slow mu (orig) first 5: {np.round(orig['mu_slow'][:5], 4)}  | elapsed {time.time() - t0:.0f}s")

    print("\n== module strength sweep (32 modules x 64): does <r> move as modules form? ==")
    for strength in (0.0, 0.05, 0.1, 0.2, 0.4, 0.7):
        cov = covariance_from_latents(n_tokens, n, [64] * 32, strength, rng, False)
        o = readouts(cov, 33)
        rot = readouts(rotate_live_covariance(cov, np.zeros(n, bool), np.random.default_rng(7)), 33)
        print(f"strength {strength:4.2f}: orig r_all {o['r_all']:.3f} r_bulk {o['r_bulk']:.3f} "
              f"slow N*IPR/3 {o['nIPR_slow']:.2f} || rotated r_all {rot['r_all']:.3f} "
              f"r_bulk {rot['r_bulk']:.3f} slow N*IPR/3 {rot['nIPR_slow']:.2f}")

    print("\n== finite-size scaling: subsample N neurons (modules of 64 kept whole), IPR of slow modes ==")
    cov = covariance_from_latents(n_tokens, n, [64] * 32, 0.7, rng, False)
    for n_sub in (256, 512, 1024, 2048):
        keep = np.arange(n_sub)          # first n_sub/64 modules, whole
        sub = cov[np.ix_(keep, keep)]
        k = n_sub // 64
        o = readouts(sub, k + 1)
        rot = readouts(rotate_live_covariance(sub, np.zeros(n_sub, bool), np.random.default_rng(9)), k + 1)
        print(f"N {n_sub:4d}: slow-mode IPR orig {o['nIPR_slow'] * 3 / n_sub:.4f} (1/64 = {1/64:.4f}) "
              f"rotated {rot['nIPR_slow'] * 3 / n_sub:.5f} (3/N = {3 / n_sub:.5f}) | r_all orig {o['r_all']:.3f} rot {rot['r_all']:.3f}")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
