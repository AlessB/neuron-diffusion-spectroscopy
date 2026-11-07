"""
Scratch check: structural drift (module loadings rotate slowly toward a second module), shared tokens,
token-level jitter as in S7. Lag structure of d and d_cv, and a one-neuron event on top.
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/claude/pg/repo/neuron-diffusion-spectroscopy/research/round4")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim7_shared_probe_paired_bootstrap as sim7
from sims_common import MODULE_NEURONS, MODULES, NEURONS
from check_crossvalidated_distance import distances

def build(rate, moved_count, seed, checkpoints=14, event=8):
    g = np.random.default_rng(1300 + seed)
    labels = np.full(NEURONS, -1)
    labels[:MODULE_NEURONS] = np.repeat(np.arange(MODULES), MODULE_NEURONS // MODULES)
    after = labels.copy()
    if moved_count:
        moved = g.permutation(MODULE_NEURONS)[:moved_count]
        after[moved] = (after[moved] + 1 + g.integers(0, MODULES - 1, moved_count)) % MODULES
    second = (labels + 1 + g.integers(0, MODULES - 1, NEURONS)) % MODULES
    speed = g.uniform(0, 1, NEURONS)
    lat = g.standard_normal((sim7.TOKENS, MODULES)); pri = g.standard_normal((sim7.TOKENS, NEURONS)); fre = g.standard_normal((sim7.TOKENS, NEURONS))
    moments = []
    for index in range(checkpoints):
        if index > 0:
            for a in (lat, pri, fre):
                a *= np.sqrt(1 - sim7.DRIFT); a += np.sqrt(sim7.DRIFT) * g.standard_normal(a.shape)
        cur = after if index >= event else labels
        theta = rate * index * speed
        v = np.sqrt(1 - sim7.STRENGTH) * pri
        inside = cur >= 0
        v[:, inside] += np.sqrt(sim7.STRENGTH) * (np.cos(theta[inside]) * lat[:, cur[inside]] + np.sin(theta[inside]) * lat[:, second[inside]])
        v[:, ~inside] += np.sqrt(sim7.STRENGTH) * fre[:, ~inside]
        moments.append(sim7.blocked(v))
    return moments, g

for rate in [0.02]:
    moments, g = build(rate, 0, 0)
    print(f"structural drift rate {rate} rad/checkpoint (scaled by a per-neuron speed in [0,1]), no event")
    print(" lag |        d |      d_cv")
    for lag in range(1, 6):
        p = []; c = []
        for s in range(0, 6):
            a, b, _ = distances(moments[s], moments[s + lag], g); p.append(a); c.append(b)
        print(f"{lag:4d} | {np.mean(p):8.5f} | {np.mean(c):9.6f}")
    for moved in [0, 1]:
        for seed in range(2):
            moments, g = build(rate, moved, seed)
            d = []; c = []
            for i in range(len(moments) - 1):
                a, b, _ = distances(moments[i], moments[i + 1], g); d.append(a); c.append(b)
            d = np.array(d); c = np.array(c)
            print(f"moved {moved} seed {seed}: d event {d[7]:.5f} quiet median {np.median(np.delete(d,7)):.5f} | "
                  f"d_cv event {c[7]:.5f} quiet median {np.median(np.delete(c,7)):.6f} quiet max {np.max(np.delete(c,7)):.6f}")
