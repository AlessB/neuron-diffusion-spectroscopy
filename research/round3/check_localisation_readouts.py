"""
Which localisation readout sees which kind of module structure? (22 Oct 2026, testing round 3)

Synthetic layers with known structure, each analysed by nds.analysis.analyse_layer exactly as a
real checkpoint would be (5 rotations, 5 bootstrap replicates, Wishart null). For three readouts
we print the original value and the largest rotated value:

  subspace   first_ten_subspace_localization (D33 as implemented on 21 Oct)
  IPR        module_aligned_localization (modes 0 .. k*-1, varimax, IPR x m / 3)
  L1         module_aligned_support      (same basis, L1 support ratio)

A readout "sees" the structure when the original is clearly above the rotated range.

    python research/round3/check_localisation_readouts.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from nds.analysis import analyse_layer
from nds.moments import BlockedNeuronMoments
from nds.synthetic import modular_activations

CASES = [
    ("no structure", (1,) * 512, 0.0),
    ("2 large modules, all neurons (2 x 256)", (256, 256), 0.5),
    ("4 modules, all neurons (4 x 128)", (128,) * 4, 0.5),
    ("8 modules, all neurons (8 x 64)", (64,) * 8, 0.5),
    ("4 x 32 modules + 384 free neurons", (32,) * 4 + (1,) * 384, 0.5),
    ("8 x 16 modules + 384 free neurons", (16,) * 8 + (1,) * 384, 0.5),
    ("2 x 100 modules + 312 free neurons", (100, 100) + (1,) * 312, 0.5),
    ("8 x 64, weak (strength 0.2)", (64,) * 8, 0.2),
]

READOUTS = [
    ("subspace", "first_ten_subspace_localization"),
    ("IPR", "module_aligned_localization"),
    ("L1", "module_aligned_support"),
]


def main():
    generator = np.random.default_rng(0)
    print(f"{'case':42s} {'k*':>4s} " + " ".join(f"{name + ' orig/rot':>18s}" for name, _ in READOUTS))
    for name, sizes, strength in CASES:
        activations, _ = modular_activations(8000, sizes, strength, generator)
        activations = activations * (activations > -0.5)

        blocked = BlockedNeuronMoments(activations.shape[1], number_of_blocks=8)
        for block in range(8):
            blocked.add_batch(activations[block * 1000:(block + 1) * 1000], block)

        result = analyse_layer(blocked, settings={"minimum_firing_rate": 0.0}, effective_tokens=8000.0)

        cells = []
        for _, key in READOUTS:
            original = result["spectrum"][key]
            rotated = max(rotation["spectrum"][key] for rotation in result["rotations"])
            mark = "*" if original > 1.2 * rotated else " "
            cells.append(f"{original:7.2f} / {rotated:5.2f} {mark}")
        print(f"{name:42s} {result['spectrum']['module_count']:4d} " + " ".join(f"{cell:>18s}" for cell in cells))
    print("* = original above 1.2 x the largest rotated value")


if __name__ == "__main__":
    main()
