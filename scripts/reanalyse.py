"""
reanalyse.py
============

Re-run the analysis with different settings on covariances already saved by collect.py, without
touching the model again. This is how the robustness checks (k = 30, unsigned distance, the
graphtools alpha-decay kernel, massive neurons excluded) are done.

    python scripts/reanalyse.py --config configs/robustness_k30_standard.yaml

The bootstrap needs the per-block statistics; runs that saved them (collection.save_block_moments,
D48) get a full band, others have none (number_of_bootstrap_replicates is forced to 0).

Rotated worlds (D41 = E): rerun a finished run in fixed-rotation worlds, from its saved moments:

    python scripts/reanalyse.py --config configs/pythia70m-standard_week1.yaml --worlds 0-19

writes <run>_world00 (the real neurons, restricted to the neurons alive at every checkpoint) and
<run>_world01 .. _world19 (the same neurons under one fixed random rotation each; nds.worlds).
Each world is an ordinary run folder, so analyse.py and every timeline function work on it.
`--worlds-seed` changes which rotations are drawn (default 0).
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nds import storage
from nds.analysis import analyse_layer
from nds.analysis import layer_summary
from nds.moments import BlockedNeuronMoments
from nds import worlds as world_tools


def moments_from_saved(saved):
    """
    Rebuild a one-block BlockedNeuronMoments from a saved layer<L>_moments.npz.
    """
    covariance = saved["covariance"].astype(np.float64)
    count = int(saved["count"])

    number_of_neurons = covariance.shape[0]

    blocked = BlockedNeuronMoments(number_of_neurons, number_of_blocks=1)

    block = blocked.blocks[0]
    block.count = count
    block.mean = saved["mean"].astype(np.float64)
    block.scatter = covariance * (count - 1)
    block.active_count = np.round(saved["firing_rate"].astype(np.float64) * count).astype(np.int64)
    block.max_abs_activation = saved["max_abs_activation"].astype(np.float64)

    return blocked


def load_layer_blocks(source_step, layer):
    """
    The saved moments of one layer: every block if they were saved, else one block (no bootstrap).
    Returns (BlockedNeuronMoments, has_blocks).
    """
    block_path = Path(source_step) / f"layer{layer}_block_moments.npz"
    if block_path.exists():
        return storage.load_block_moments(source_step, layer), True

    saved = storage.load_layer_moments(source_step, layer)
    return moments_from_saved(saved), False


def reanalyse_run(source_folder, output_folder, settings, source_name, world=None, worlds_seed=0):
    """
    Analyse every finished step of source_folder with `settings` into output_folder. With `world`
    given, each layer's moments are first restricted to the neurons alive at every checkpoint and
    rotated by that world's fixed rotation (nds.worlds).
    """
    world_neurons = {}
    world_rotations = {}

    for step in storage.finished_steps(source_folder):
        source_step = storage.step_folder(source_folder, step)
        source_summary = storage.load_summary(source_step)

        layer_results = {}

        for layer_key in source_summary["layers"].keys():
            layer = int(layer_key)

            blocked, has_blocks = load_layer_blocks(source_step, layer)

            layer_settings = dict(settings)

            if world is not None:
                if layer not in world_neurons:
                    neurons, mean_live = world_tools.always_alive_neurons(source_folder, layer)
                    world_neurons[layer] = neurons
                    world_rotations[layer] = world_tools.world_rotation(len(neurons), world, worlds_seed)
                    print(f"world {world:02d} layer {layer}: {len(neurons)} neurons alive at every checkpoint "
                          f"(mean live per checkpoint {mean_live:.0f})")
                blocked = world_tools.world_moments(blocked, world_neurons[layer], world_rotations[layer])
                layer_settings = world_tools.world_settings(layer_settings)

            if not has_blocks:
                layer_settings["number_of_bootstrap_replicates"] = 0

            # the noise level measured at collection, for the null (D36)
            source_layer = source_summary["layers"].get(layer_key, {})
            effective_tokens = source_layer.get("effective_tokens")

            result = analyse_layer(blocked, settings=layer_settings, effective_tokens=effective_tokens)

            storage.save_layer_spectrum(storage.step_folder(output_folder, step), layer, result["spectrum"])

            layer_results[layer] = layer_summary(result)

            # no blocks saved: no bootstrap band. Repeat the main value so tables stay readable (the
            # band then has zero width, which the figures show honestly)
            if not has_blocks:
                layer_results[layer]["bootstrap"] = [
                    {
                        "module_count": result["spectrum"]["module_count"],
                    }
                ]

            # token-side baselines do not depend on the kernel or the neuron basis: carry them over
            for key in ["baselines_prompt_entropy", "baselines_token_sample", "baselines_knowledge_entropy", "neuron_statistics", "lag_one_correlation", "effective_tokens"]:
                if key in source_layer:
                    layer_results[layer][key] = source_layer[key]

        summary = dict(source_summary)
        summary["layers"] = layer_results
        summary["reanalysed_from"] = source_name
        if world is not None:
            summary["world"] = int(world)
            summary["worlds_seed"] = int(worlds_seed)

        storage.save_summary(storage.step_folder(output_folder, step), summary)

        label = "re-analysed"
        if world is not None:
            label = f"world {world:02d}"
        print(f"step {step}: {label}")


def parse_worlds(text):
    """
    "0-19" -> [0, ..., 19]; "3" -> [3]; "0,5,7" -> [0, 5, 7].
    """
    chosen = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            first, last = part.split("-")
            chosen.extend(range(int(first), int(last) + 1))
        else:
            chosen.append(int(part))
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--worlds", default=None, help="rotated worlds to build from this run, e.g. 0-19 (D41)")
    parser.add_argument("--worlds-seed", type=int, default=0)
    arguments = parser.parse_args()

    with open(arguments.config, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    settings = dict(config.get("analysis", {}))

    if arguments.worlds is None:
        source_folder = storage.run_folder_for(config["source_run"])
        output_folder = storage.run_folder_for(config["run_name"])
        output_folder.mkdir(parents=True, exist_ok=True)
        shutil.copy(arguments.config, output_folder / "config.yaml")
        reanalyse_run(source_folder, output_folder, settings, config["source_run"])
        return

    # worlds of a run: the source is the run itself (or its source_run for a re-analysed config)
    source_name = config.get("source_run", config["run_name"])
    source_folder = storage.run_folder_for(source_name)

    for world in parse_worlds(arguments.worlds):
        output_name = world_tools.world_run_name(source_name, world)
        output_folder = storage.run_folder_for(output_name)
        output_folder.mkdir(parents=True, exist_ok=True)
        shutil.copy(arguments.config, output_folder / "config.yaml")
        reanalyse_run(source_folder, output_folder, settings, source_name, world=world, worlds_seed=arguments.worlds_seed)


if __name__ == "__main__":
    main()
