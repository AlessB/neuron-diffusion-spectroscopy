"""
check_fastssl_alpha_ranges.py
=============================

alpha-ReQ as in the original fastssl repo (Agrawal et al. 2022),
/home/claude/refcode/kumarkrishna__fastssl/fastssl/utils/powerlaw.py:
    stringer_get_powerlaw  (lines 32-54)  byte-identical to Li et al.'s copy
    fit_powerlaw / robust_fit_powerlaw (lines 9-30)  unweighted sliding-window fits, median slope
Fit ranges used in fastssl:
    scripts/train_model.py:607-608  np.arange(3, 100)   (full-test-set alpha)
    scripts/train_model.py:625      np.arange(5, 50)    (per-batch alpha)
    nb/plot_alpha.ipynb             np.arange(10, 100)
and Li et al. use np.arange(11, 100).

Checks that nds.baselines.alpha_req(first_index, last_index) reproduces each range exactly, and shows
how much alpha moves with the range on realistic (non-pure power-law) spectra.
"""

import sys

import numpy as np
from sklearn.linear_model import LinearRegression

sys.path.insert(0, "/home/claude/pg/repo/neuron-diffusion-spectroscopy/src")
sys.path.insert(0, "/home/claude/pg/research/checks")

from nds.baselines import alpha_req, covariance_eigenvalues  # noqa: E402
from check_li_rankme_alpha import li_get_eigenspectrum, li_stringer_get_powerlaw  # noqa: E402
from synthetic_activations import gelu_neurons, power_law_gaussian  # noqa: E402


def fastssl_fit_powerlaw(arr, start, end):
    x_range = np.arange(start, end + 1).astype(int)
    y_range = arr[x_range - 1]
    reg = LinearRegression().fit(np.log(x_range).reshape(-1, 1), np.log(y_range).reshape(-1, 1))
    return -reg.coef_[0][0]


def fastssl_robust_fit_powerlaw(arr, start, end):
    window = int((end - start) / 10)
    slope_arr = np.array([fastssl_fit_powerlaw(arr, idx, idx + window) for idx in range(start, end + 1)])
    return np.median(slope_arr)


RANGES = {
    "fastssl train_model (3,100)": (3, 100),
    "fastssl per-batch (5,50)": (5, 50),
    "fastssl notebook (10,100)": (10, 100),
    "Li et al. (11,100)": (11, 100),
}


def run(name, data):
    ratio = li_get_eigenspectrum(data)
    eig = covariance_eigenvalues(np.cov(data, rowvar=False))
    print(f"--- {name}")
    for label, (lo, hi) in RANGES.items():
        theirs = li_stringer_get_powerlaw(ratio, np.arange(lo, hi))[0]
        mine, _ = alpha_req(eig, first_index=lo, last_index=hi - 1)
        print(f"  {label:30s} theirs={theirs:.6f}  ours={mine:.6f}  diff={theirs - mine:.2e}")
    print(f"  robust_fit_powerlaw(ranks 12..100, median of windows) = {fastssl_robust_fit_powerlaw(ratio, 12, 100):.6f}")


if __name__ == "__main__":
    run("power-law Gaussian alpha=1.0 (512 neurons)", power_law_gaussian(6000, 512, alpha=1.0, seed=1))
    run("GELU neurons (512)", gelu_neurons(6000, 512, seed=2))
    run("GELU neurons, steeper latent (decay 1.5)", gelu_neurons(6000, 512, latent_decay=1.5, seed=5))
