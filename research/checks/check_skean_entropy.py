"""
check_skean_entropy.py
======================

Skean et al. (2025) "prompt entropy", re-implemented in numpy from:
  /home/claude/refcode/OFSkean__information_flow/experiments/utils/metrics/metric_functions.py
      normalize()          lines 47-53   centre over tokens, then L2-normalise each token row
      compute_entropy()    lines 179-207 Gram (N<=D: N x N, else D x D), clamp(min=0) ELEMENTWISE,
                                         divide by trace, repitl matrixAlphaEntropy
      entropy_normalization() lines 8-36 default 'maxEntropy': divide by min(log N, log D)
  /home/claude/refcode/OFSkean__information_flow/experiments/utils/metrics/metric_calling.py
      lines 93-101  normalize() is applied per prompt (pad tokens removed), then averaged over prompts
  /home/claude/refcode/uk-cliplab__representation-itl/src/repitl/matrix_itl.py
      vonNeumannEntropy  lines 183-199  eigh, keep eigenvalues > 0, renormalise, -sum p log p (nats)
      generalizedInformationPotential lines 11-34; frobeniusGIP lines 37-52 (alpha == 2 shortcut)

and compared with nds.baselines.matrix_entropy (Renyi entropy of the covariance eigenvalues).
"""

import sys

import numpy as np

sys.path.insert(0, "/home/claude/pg/repo/neuron-diffusion-spectroscopy/src")
sys.path.insert(0, "/home/claude/pg/research/checks")

from nds.baselines import covariance_eigenvalues, matrix_entropy, rankme_li  # noqa: E402
from synthetic_activations import gelu_neurons  # noqa: E402


def skean_normalize(tokens):
    centred = tokens - tokens.mean(axis=0)
    norms = np.linalg.norm(centred, axis=1, keepdims=True)
    return centred / norms


def von_neumann_entropy(matrix):
    eigenvalues = np.linalg.eigvalsh(matrix)
    kept = eigenvalues[eigenvalues > 0.0]
    p = kept / kept.sum()
    return float(-np.sum(p * np.log(p)))


def renyi_entropy_eig(matrix, alpha):
    eigenvalues = np.linalg.eigvalsh(matrix)
    kept = eigenvalues[eigenvalues > 0.0]
    p = kept / kept.sum()
    return float(np.log(np.sum(np.exp(alpha * np.log(p)))) / (1.0 - alpha))


def renyi2_frobenius_shortcut(matrix):
    gip = np.sum(matrix ** 2) / matrix.shape[0] ** 2       # repitl frobeniusGIP
    return float(-np.log(gip))


def skean_prompt_entropy(tokens, clamp=True, alpha=1.0, normalization="raw"):
    hidden = skean_normalize(tokens)
    n_tokens, dim = hidden.shape
    if n_tokens > dim:
        gram = hidden.T @ hidden
    else:
        gram = hidden @ hidden.T
    if clamp:
        gram = np.maximum(gram, 0.0)
    gram = gram / np.trace(gram)
    entropy = von_neumann_entropy(gram) if alpha == 1.0 else renyi_entropy_eig(gram, alpha)
    if normalization == "maxEntropy":
        entropy /= min(np.log(n_tokens), np.log(dim))
    return entropy


if __name__ == "__main__":
    data = gelu_neurons(number_of_tokens=200 * 64, number_of_neurons=256, seed=7)
    prompts = data.reshape(200, 64, 256)          # 200 "prompts" of 64 tokens, 256 neurons

    # 1. Skean's code path vs the same thing without the elementwise clamp
    with_clamp = np.mean([skean_prompt_entropy(p) for p in prompts])
    without_clamp = np.mean([skean_prompt_entropy(p, clamp=False) for p in prompts])

    # 2. without clamp, the token-Gram entropy equals the covariance-eigenvalue entropy of the
    #    row-normalised data's second-moment matrix H^T H (shared nonzero spectrum with H H^T).
    #    Note H is NOT re-centred after row normalisation, so np.cov(H) would differ slightly.
    ours_on_row_normalised = np.mean([
        matrix_entropy(covariance_eigenvalues(skean_normalize(p).T @ skean_normalize(p)))
        for p in prompts
    ])
    ours_on_row_normalised_cov = np.mean([
        matrix_entropy(covariance_eigenvalues(np.cov(skean_normalize(p), rowvar=False)))
        for p in prompts
    ])
    # 3. our actual baseline: covariance of raw (not row-normalised) activations, per prompt and pooled
    ours_raw_per_prompt = np.mean([matrix_entropy(covariance_eigenvalues(np.cov(p, rowvar=False))) for p in prompts])
    pooled_eigs = covariance_eigenvalues(np.cov(data, rowvar=False))
    ours_pooled = matrix_entropy(pooled_eigs)

    print("--- Skean prompt entropy (alpha=1, 'raw' normalisation), mean over 200 prompts of 64 tokens")
    print(f"  Skean code (with clamp)                      : {with_clamp:.6f}")
    print(f"  Skean without clamp                          : {without_clamp:.6f}")
    print(f"  nds matrix_entropy, eig(H^T H), H row-normed : {ours_on_row_normalised:.6f}  "
          f"(diff to no-clamp {ours_on_row_normalised - without_clamp:.2e})")
    print(f"  nds matrix_entropy, np.cov of row-normalised : {ours_on_row_normalised_cov:.6f}  (re-centred columns)")
    print(f"  nds matrix_entropy, raw cov, per prompt      : {ours_raw_per_prompt:.6f}")
    print(f"  nds matrix_entropy, raw cov, pooled tokens   : {ours_pooled:.6f}")
    print(f"  maxEntropy-normalised Skean (÷ min(log N, log D)) : {np.mean([skean_prompt_entropy(p, normalization='maxEntropy') for p in prompts]):.6f}")
    print(f"  log rankme_li(pooled) - matrix_entropy(pooled): {np.log(rankme_li(pooled_eigs)) - ours_pooled:.3e}")

    # 4. alpha = 2 path in repitl: frobenius shortcut on an already trace-normalised matrix
    p = prompts[0]
    hidden = skean_normalize(p)
    gram = np.maximum(hidden @ hidden.T, 0.0)
    gram_normalised = gram / np.trace(gram)
    exact = renyi_entropy_eig(gram_normalised, 2.0)
    shortcut = renyi2_frobenius_shortcut(gram_normalised)
    print("--- repitl alpha=2 shortcut on a trace-normalised Gram (what compute_entropy passes in)")
    print(f"  eigenvalue Renyi-2 = {exact:.6f}   frobeniusGIP Renyi-2 = {shortcut:.6f}   "
          f"diff = {shortcut - exact:.6f}   2 log N = {2 * np.log(gram.shape[0]):.6f}")
    print(f"  ours matrix_entropy(order=2) on same eigenvalues = "
          f"{matrix_entropy(np.sort(np.linalg.eigvalsh(gram_normalised))[::-1], order=2.0):.6f}")
