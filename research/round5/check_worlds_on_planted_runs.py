"""
Check of the rotated-worlds implementation (D41 = E, nds.worlds and reanalyse.py --worlds) on
planted runs saved exactly as collect.py saves a real run, block moments included (D48).

Three runs of 8 checkpoints, 256 neurons (8 modules of 24 plus 64 free), 4,000 fresh tokens per
checkpoint in 8 blocks, a dead-neuron set that changes at checkpoint 2:
  stable     nothing changes
  reshuffle  at checkpoint 4 half the module neurons change module
  forming    modules switch on at checkpoint 4
Then 19 worlds per run through the real reanalyse code, and the rank table (scripts/worlds_report).

What should come out, if the implementation is right (research/round4, S1 did it outside the
pipeline):
  - localisation (IPR and L1) beats all 19 worlds wherever modules exist, and not before forming;
  - the ARI and lineage strengths beat all worlds at the reshuffle, and nowhere in the stable run.

    python research/round5/check_worlds_on_planted_runs.py
"""

import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

RUNS = tempfile.mkdtemp(prefix="nds-worlds-check-")
os.environ["NDS_RUNS_DIR"] = RUNS

from nds import storage
from nds.analysis import analyse_layer
from nds.analysis import layer_summary
from nds.moments import BlockedNeuronMoments
from nds.worlds import world_rank_table

spec = importlib.util.spec_from_file_location("reanalyse", ROOT / "scripts" / "reanalyse.py")
reanalyse = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reanalyse)

NEURONS = 256
MODULE_NEURONS = 192
MODULES = 8
TOKENS = 4000
BLOCKS = 8
CHECKPOINTS = 8
EVENT = 4
WORLDS = 19
SETTINGS = {"minimum_firing_rate": 0.0, "number_of_bootstrap_replicates": 3, "number_of_rotations": 2, "fixed_bandwidth_dse": False}


def write_run(name, scenario, seed):
    generator = np.random.default_rng(seed)
    labels = np.full(NEURONS, -1)
    labels[:MODULE_NEURONS] = np.repeat(np.arange(MODULES), MODULE_NEURONS // MODULES)
    labels_after = labels.copy()
    swapped = generator.permutation(MODULE_NEURONS)[: MODULE_NEURONS // 2]
    labels_after[swapped] = generator.permutation(labels_after[swapped])
    dead_early = generator.permutation(NEURONS)[: NEURONS // 5]
    dead_late = generator.permutation(NEURONS)[: NEURONS // 10]

    folder = storage.run_folder_for(name)
    for index in range(CHECKPOINTS):
        step = index * 1000
        current = labels
        strength = 0.5
        if scenario == "reshuffle" and index >= EVENT:
            current = labels_after
        if scenario == "forming" and index < EVENT:
            strength = 0.0
        latents = generator.standard_normal((TOKENS, MODULES))
        values = np.sqrt(1 - strength) * generator.standard_normal((TOKENS, NEURONS))
        inside = current >= 0
        values[:, inside] += np.sqrt(strength) * latents[:, current[inside]]
        values[:, ~inside] += np.sqrt(strength) * generator.standard_normal((TOKENS, int(np.sum(~inside))))
        dead = dead_early
        if index >= 2:
            dead = dead_late
        values[:, dead] = 0.0

        blocked = BlockedNeuronMoments(NEURONS, number_of_blocks=BLOCKS)
        size = TOKENS // BLOCKS
        for block in range(BLOCKS):
            blocked.add_batch(values[block * size:(block + 1) * size], block)

        result = analyse_layer(blocked, settings=SETTINGS, effective_tokens=TOKENS)
        step_folder = storage.step_folder(folder, step)
        storage.save_layer_moments(step_folder, 0, blocked.merged())
        storage.save_block_moments(step_folder, 0, blocked)
        storage.save_layer_spectrum(step_folder, 0, result["spectrum"])
        summary = {
            "model_name": "planted",
            "step": step,
            "tokens_seen": step,
            "probe_loss": float("nan"),
            "induction": {"induction_loss_drop": 0.0},
            "layers": {0: layer_summary(result)},
        }
        storage.save_summary(step_folder, summary)


def main():
    print(f"runs in {RUNS}")
    for scenario in ["stable", "reshuffle", "forming"]:
        name = f"planted-{scenario}"
        write_run(name, scenario, seed=40)
        source = storage.run_folder_for(name)
        for world in range(WORLDS + 1):
            output = storage.run_folder_for(f"{name}_world{world:02d}")
            reanalyse.reanalyse_run(source, output, SETTINGS, name, world=world, worlds_seed=0)
        table = world_rank_table(name, 0, WORLDS)
        print(f"\n===== {scenario} (event between steps {(EVENT - 1) * 1000} and {EVENT * 1000}) =====")
        steps = table["step"].astype(int)
        for key in ["module_aligned_localization", "module_aligned_support", "one_minus_ari_strength", "lineage_loss_strength"]:
            entry = table[key]
            cells = []
            for row in range(len(steps)):
                real = entry["real"][row]
                if not np.isfinite(real):
                    cells.append("    -    ")
                    continue
                mark = " "
                if entry["p"][row] <= 0.05 + 1e-12:
                    mark = "*"
                cells.append(f"{real:5.2f}/{entry['world_maximum'][row]:5.2f}{mark}")
            print(f"{key:30s} " + " ".join(cells))
        print("(real / largest of 19 worlds; * = beats every world, p = 0.05)")


if __name__ == "__main__":
    main()
