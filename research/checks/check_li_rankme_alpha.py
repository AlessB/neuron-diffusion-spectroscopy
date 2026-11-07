"""
check_li_rankme_alpha.py
========================

Re-implements, from the cloned code, Li et al. (2025) utils/powerlaw.py:
    get_eigenspectrum   (lines 89-95)   sklearn PCA(svd_solver='full') -> explained_variance_ratio_
    rankme              (lines 133-140) eps = 1e-7 on eigenvalue ratios
    stringer_get_powerlaw (lines 33-55) weighted log-log LSQ, weights 1/rank, IN-PLACE `trange += 1`
(/home/claude/refcode/melodylizx__Tracing-the-representation-geometry-of-language-models/utils/powerlaw.py)

and compares with nds.baselines.rankme_li / alpha_req, fed the covariance built by
nds.moments.NeuronMoments from the same data.
"""

import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

sys.path.insert(0, "/home/claude/pg/repo/neuron-diffusion-spectroscopy/src")
sys.path.insert(0, "/home/claude/pg/research/checks")

from nds.baselines import alpha_req, covariance_eigenvalues, rankme_li, matrix_entropy  # noqa: E402
from nds.moments import NeuronMoments  # noqa: E402
from synthetic_activations import gelu_neurons, power_law_gaussian  # noqa: E402


# ---------------------------------------------------------------- Li et al. code, verbatim logic
def li_get_eigenspectrum(activations_np, max_eigenvals=2048):
    feats = activations_np.reshape(activations_np.shape[0], -1)
    feats_center = feats - feats.mean(axis=0)
    pca = PCA(n_components=min(max_eigenvals, feats_center.shape[0], feats_center.shape[1]), svd_solver="full")
    pca.fit(feats_center)
    return pca.explained_variance_ratio_


def li_rankme(eigen):
    l1 = np.sum(np.abs(eigen))
    eps = 1e-7
    scores = eigen / l1 + eps
    entropy = -np.sum(scores * np.log(scores))
    return np.exp(entropy)


def li_stringer_get_powerlaw(ss, trange):
    logss = np.log(np.abs(ss))
    y = logss[trange][:, np.newaxis]
    trange += 1                                   # mutates the caller's array
    nt = trange.size
    x = np.concatenate((-np.log(trange)[:, np.newaxis], np.ones((nt, 1))), axis=1)
    w = 1.0 / trange.astype(np.float32)[:, np.newaxis]
    b = np.linalg.solve(x.T @ (x * w), (w * x).T @ y).flatten()
    allrange = np.arange(0, ss.size).astype(int) + 1
    x = np.concatenate((-np.log(allrange)[:, np.newaxis], np.ones((ss.size, 1))), axis=1)
    ypred = np.exp((x * b).sum(axis=1))
    alpha = b[0]
    max_range = 500 if len(ss) >= 512 else len(ss) - 10
    fit_R2 = r2_score(y_true=logss[trange[0]:max_range], y_pred=np.log(np.abs(ypred))[trange[0]:max_range])
    try:
        fit_R2_100 = r2_score(y_true=logss[trange[0]:100], y_pred=np.log(np.abs(ypred))[trange[0]:100])
    except Exception:
        fit_R2_100 = None
    return alpha, ypred, fit_R2, fit_R2_100


# ---------------------------------------------------------------- our pipeline
def ours(data, batch_size=500):
    moments = NeuronMoments(data.shape[1])
    for start in range(0, len(data), batch_size):
        moments.add_batch(data[start:start + batch_size])
    eigenvalues = covariance_eigenvalues(moments.covariance())
    return eigenvalues


def compare(name, data):
    li_eigen = li_get_eigenspectrum(data)
    our_eigen = ours(data)
    our_ratio = our_eigen / our_eigen.sum()

    k = len(li_eigen)
    li_alpha, _, li_r2, li_r2_100 = li_stringer_get_powerlaw(li_eigen, np.arange(11, 100))
    our_alpha, our_r2 = alpha_req(our_eigen)

    print(f"--- {name}: tokens={data.shape[0]} neurons={data.shape[1]} (Li keeps {k} components)")
    print(f"  max |eigen ratio Li - ours| over Li's {k}  : {np.max(np.abs(li_eigen - our_ratio[:k])):.3e}")
    print(f"  RankMe  Li={li_rankme(li_eigen):.6f}  ours(rankme_li)={rankme_li(our_eigen):.6f}  "
          f"diff={li_rankme(li_eigen) - rankme_li(our_eigen):.3e}")
    print(f"  alpha   Li={li_alpha:.10f}  ours={our_alpha:.10f}  diff={li_alpha - our_alpha:.3e}")
    print(f"  R2      Li fit_R2(idx 12..499)={li_r2:.5f}  Li fit_R2_100(idx 12..99)={li_r2_100:.5f}  "
          f"ours(idx 11..99)={our_r2:.5f}")
    print(f"  log(rankme_li) - matrix_entropy(order 1) = {np.log(rankme_li(our_eigen)) - matrix_entropy(our_eigen):.3e}")


if __name__ == "__main__":
    np.set_printoptions(precision=4)

    compare("power-law Gaussian alpha=1.0", power_law_gaussian(6000, 512, alpha=1.0, seed=1))
    compare("GELU neurons, 512", gelu_neurons(6000, 512, seed=2))

    # d_mlp > 2048 (Pythia-160m has 3072): Li's PCA keeps only 2048 components
    compare("GELU neurons, 3072 (> 2048 cap)", gelu_neurons(4000, 3072, latent_dimension=1024, seed=3))

    # the in-place mutation gotcha
    trange = np.arange(11, 100)
    eig = li_get_eigenspectrum(power_law_gaussian(3000, 256, alpha=1.0, seed=4))
    first = li_stringer_get_powerlaw(eig, trange)[0]
    second = li_stringer_get_powerlaw(eig, trange)[0]
    print("--- in-place trange mutation")
    print(f"  same array reused: alpha call1={first:.6f}  call2={second:.6f}  trange now starts at {trange[0]}")
