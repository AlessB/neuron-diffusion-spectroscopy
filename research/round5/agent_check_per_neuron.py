"""
Scratch check: per-neuron cross-validated displacement c_i = [(P_from(A)-P_to(A)) (P_from(B)-P_to(B))]_ii,
averaged over splits. Does its largest entry find the moved neuron(s)? Event transition only.
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/claude/pg/repo/neuron-diffusion-spectroscopy/research/round4")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim7_shared_probe_paired_bootstrap as sim7
from check_crossvalidated_distance import build_moments, projector

for count in [1, 2, 10]:
    for seed in range(2):
        fraction = count / 192
        g0 = np.random.default_rng(900 + seed)
        # rebuild the moved set exactly as build_moments draws it
        moved = g0.permutation(192)[:count]
        moments, g = build_moments(fraction, seed)
        m_from, m_to = moments[7], moments[8]
        _, k = sim7.module_subspace(m_from.merged().covariance())
        per_neuron = np.zeros(m_from.merged().covariance().shape[0])
        for _ in range(12):
            order = g.permutation(sim7.BLOCKS)
            diffs = []
            for half in (order[:4], order[4:]):
                a, _ = sim7.module_subspace(m_from.merged(half).covariance(), k)
                b, _ = sim7.module_subspace(m_to.merged(half).covariance(), k)
                diffs.append(projector(a) - projector(b))
            per_neuron += np.diag(diffs[0] @ diffs[1]) / 12
        top = np.argsort(per_neuron)[::-1][:count]
        others = np.delete(per_neuron, moved)
        print(f"moved {count:2d} seed {seed}: top-{count} hits {len(set(top) & set(moved))}/{count}; "
              f"moved mean c {per_neuron[moved].mean():.5f}, others median {np.median(others):.6f}, others max {others.max():.5f}, "
              f"share of total on moved {per_neuron[moved].sum() / per_neuron.sum():.2f}")
