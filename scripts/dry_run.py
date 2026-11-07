"""
dry_run.py
==========

The whole pipeline on fake data, no model and no GPU: a pretend "training run" where three neuron
modules slowly rearrange (the avoided-crossing toy), saved exactly like collect.py saves a real
run, then analysed. Use it to check an installation, or to see what the outputs look like before
the first real run.

    python scripts/dry_run.py
    # then look in runs/dry-run/analysis/

Takes well under a minute.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nds.baselines import knowledge_entropy
from nds import checkpoints
from nds import storage
from nds.analysis import analyse_layer
from nds.analysis import layer_summary
from nds.moments import BlockedNeuronMoments
from nds.noise import effective_token_count
from nds.synthetic import three_module_path
from nds.token_sample import token_sample_baselines


def main():
    run_folder = storage.run_folder_for("dry-run")

    steps = checkpoints.week_one_steps()

    positions = np.linspace(0.0, 1.0, len(steps))

    for step, position in zip(steps, positions):
        activations, _ = three_module_path(position, direct_coupling=0.1)

        number_of_tokens, number_of_neurons = activations.shape

        blocked = BlockedNeuronMoments(number_of_neurons, number_of_blocks=8)

        rows_per_block = number_of_tokens // 8
        for block_index in range(8):
            rows = activations[block_index * rows_per_block:(block_index + 1) * rows_per_block]
            blocked.add_batch(rows, block_index)

        # synthetic tokens are independent, so the lag-one correlation is 0 and T_eff = T
        effective_tokens = effective_token_count(number_of_tokens, 0.0)

        result = analyse_layer(
            blocked,
            settings={"minimum_firing_rate": 0.0},
            effective_tokens=effective_tokens,
        )

        folder = storage.step_folder(run_folder, step)
        storage.save_layer_moments(folder, 0, blocked.merged())
        storage.save_layer_spectrum(folder, 0, result["spectrum"])

        # the fixed token sample (here simply the first 512 synthetic tokens)
        sample_rows = np.arange(512)
        sample = activations[sample_rows]
        storage.save_token_sample(folder, 0, sample, np.zeros(512, dtype=int), sample_rows)

        summary = {
            "model_name": "synthetic-three-modules",
            "step": step,
            "tokens_seen": checkpoints.tokens_seen(step),
            "probe_loss": float("nan"),
            "induction": {"induction_loss_drop": 0.0},
            "layers": {
                0: layer_summary(result),
            },
        }

        summary["layers"][0]["baselines_token_sample"] = token_sample_baselines(sample)
        summary["layers"][0]["lag_one_correlation"] = 0.0
        summary["layers"][0]["baselines_knowledge_entropy"] = knowledge_entropy(np.abs(activations).mean(axis=0))

        storage.save_summary(folder, summary)

        print(f"step {step:6d}: modules {result['spectrum']['module_count']}, "
              f"slow-subspace localisation {result['spectrum']['slow_subspace_localization']:.2f}")

    config_path = run_folder / "config.yaml"
    config_path.write_text("run_name: dry-run\n")

    print(f"\nnow run:  python scripts/analyse.py --config {config_path}")


if __name__ == "__main__":
    main()
