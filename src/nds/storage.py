"""
storage.py
==========

Where results go on disk, and how they are written and read back.

Layout of one run (everything under runs/, which git ignores, or under $NDS_RUNS_DIR):

    runs/<run_name>/
        config.yaml                     the exact settings used (copied at the start)
        step<NNNNNN>/
            layer<L>_moments.npz        covariance (float32), mean, firing rate, max |activation|,
                                        token count. Lets us re-analyse with other settings
                                        without running the model again. ~16 MB per layer for 70M.
            layer<L>_spectrum.npz       ALL Laplacian eigenvalues (16 KB for 2,048 neurons), the
                                        first 50 eigenvectors, and the model neuron ids the
                                        eigenvector rows refer to
            summary.json                every scalar: module count, DSE(t), IPR, baselines,
                                        rotation and bootstrap replicates, probe loss, induction
                                        scores, timings

Zero-padded step numbers (step000000, step001000, ...) keep folders in training order when sorted.
"""

import json
import os
from pathlib import Path

import numpy as np


def runs_root():
    """
    The folder that holds every run. Default: `runs/` inside the repo.

    Set the environment variable NDS_RUNS_DIR to put runs somewhere else, for example on the
    shared LAN drive (DECISIONS D22), so every machine sees the same results:

        export NDS_RUNS_DIR=/mnt/nds-share/runs          # Fedora or WSL2

    Large results never go to GitHub; this keeps them off the repo's disk too.
    """
    configured = os.environ.get("NDS_RUNS_DIR", "").strip()

    if configured:
        return Path(configured)

    return Path("runs")


def run_folder_for(run_name):
    return runs_root() / run_name


def step_folder(run_folder, step):
    return Path(run_folder) / f"step{int(step):06d}"


def to_jsonable(value):
    """
    Convert numpy numbers and arrays (and dicts/lists of them) into plain Python for json.dump.
    """
    if isinstance(value, dict):
        converted = {}
        for key, item in value.items():
            converted[str(key)] = to_jsonable(item)
        return converted

    if isinstance(value, (list, tuple)):
        converted_list = []
        for item in value:
            converted_list.append(to_jsonable(item))
        return converted_list

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    return value


def save_layer_moments(folder, layer, moments):
    """
    Save the full-data moments of one layer (merged over blocks).
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    path = folder / f"layer{layer}_moments.npz"

    np.savez(
        path,
        covariance=moments.covariance().astype(np.float32),
        mean=moments.mean.astype(np.float32),
        firing_rate=moments.firing_rate().astype(np.float32),
        max_abs_activation=moments.max_abs_activation.astype(np.float32),
        count=np.array(moments.count),
    )

    return path


def load_layer_moments(folder, layer):
    path = Path(folder) / f"layer{layer}_moments.npz"

    with np.load(path) as data:
        loaded = {}
        for key in data.files:
            loaded[key] = data[key]

    return loaded


def save_layer_spectrum(folder, layer, spectrum):
    """
    Save eigenvalues/eigenvectors of one layer's neuron graph.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    path = folder / f"layer{layer}_spectrum.npz"

    arrays = {
        "laplacian_eigenvalues": spectrum["laplacian_eigenvalues"],
        "all_laplacian_eigenvalues": spectrum.get("all_laplacian_eigenvalues", spectrum["laplacian_eigenvalues"]),
        "symmetric_vectors": spectrum["symmetric_vectors"].astype(np.float32),
        "all_mode_localization": spectrum.get("all_mode_localization", np.array([])),
        "live_neuron_ids": spectrum["live_neuron_ids"],
    }

    # module membership per live neuron (analysis.py, settings "module_partition")
    if "module_labels" in spectrum:
        arrays["module_labels"] = spectrum["module_labels"]
    if "module_attachment" in spectrum:
        arrays["module_attachment"] = np.asarray(spectrum["module_attachment"], dtype=np.float32)

    np.savez(path, **arrays)

    return path


def load_layer_spectrum(folder, layer):
    path = Path(folder) / f"layer{layer}_spectrum.npz"

    with np.load(path) as data:
        loaded = {}
        for key in data.files:
            loaded[key] = data[key]

    return loaded


def save_summary(folder, summary):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    path = folder / "summary.json"

    with open(path, "w", encoding="utf-8") as summary_file:
        json.dump(to_jsonable(summary), summary_file, indent=1)

    return path


def load_summary(folder):
    path = Path(folder) / "summary.json"

    with open(path, "r", encoding="utf-8") as summary_file:
        return json.load(summary_file)


def finished_steps(run_folder):
    """
    Steps that already have a summary.json (so a stopped run can resume where it left off).
    """
    run_folder = Path(run_folder)

    steps = []

    if not run_folder.exists():
        return steps

    for folder in sorted(run_folder.glob("step*")):
        if (folder / "summary.json").exists():
            steps.append(int(folder.name.replace("step", "")))

    return steps


def save_token_sample(folder, layer, activations, sequence_index, position):
    """
    The fixed token sample of one layer (DECISIONS D31): activations in float16 (tokens x neurons)
    and where each token came from. ~8 MB per layer of pythia-70m for 2,048 tokens.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    path = folder / f"layer{layer}_token_sample.npz"

    np.savez(
        path,
        activations=np.asarray(activations).astype(np.float16),
        sequence_index=np.asarray(sequence_index, dtype=np.int32),
        position=np.asarray(position, dtype=np.int32),
    )

    return path


def save_neuron_statistics(folder, layer, arrays):
    """
    Per-neuron statistics of one layer (neuron_stats.layer_neuron_statistics): a few arrays of
    length 2,048, float32, about 50 KB.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    path = folder / f"layer{layer}_neuron_stats.npz"

    stored = {}
    for name, values in arrays.items():
        stored[name] = np.asarray(values).astype(np.float32)

    np.savez(path, **stored)

    return path


def load_neuron_statistics(folder, layer):
    path = Path(folder) / f"layer{layer}_neuron_stats.npz"

    with np.load(path) as data:
        loaded = {}
        for key in data.files:
            loaded[key] = data[key]

    return loaded


def load_token_sample(folder, layer):
    path = Path(folder) / f"layer{layer}_token_sample.npz"

    with np.load(path) as data:
        loaded = {}
        for key in data.files:
            loaded[key] = data[key]

    return loaded


def save_block_moments(folder, layer, blocked_moments):
    """
    Every bootstrap block's moments (count, mean, scatter in float32), so the bootstrap can be redone
    later with as many replicates as wanted (D48, proposed). Large: blocks x 16 MB per layer of
    pythia-70m, so this is off unless the config asks for it (collection.save_block_moments).
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    path = folder / f"layer{layer}_block_moments.npz"

    counts = []
    means = []
    scatters = []
    active_counts = []
    max_abs = []

    for block in blocked_moments.blocks:
        counts.append(block.count)
        means.append(block.mean.astype(np.float32))
        scatters.append(block.scatter.astype(np.float32))
        active_counts.append(block.active_count)
        max_abs.append(block.max_abs_activation.astype(np.float32))

    np.savez(
        path,
        count=np.asarray(counts, dtype=np.int64),
        mean=np.stack(means),
        scatter=np.stack(scatters),
        active_count=np.stack(active_counts),
        max_abs_activation=np.stack(max_abs),
    )

    return path


def load_block_moments(folder, layer):
    """
    Rebuild a BlockedNeuronMoments from save_block_moments.
    """
    from nds.moments import BlockedNeuronMoments

    path = Path(folder) / f"layer{layer}_block_moments.npz"

    with np.load(path) as data:
        counts = data["count"]
        means = data["mean"].astype(np.float64)
        scatters = data["scatter"].astype(np.float64)
        active_counts = data["active_count"]
        max_abs = data["max_abs_activation"].astype(np.float64)

    number_of_blocks = len(counts)
    number_of_neurons = means.shape[1]

    blocked = BlockedNeuronMoments(number_of_neurons, number_of_blocks=number_of_blocks)

    for index in range(number_of_blocks):
        block = blocked.blocks[index]
        block.count = int(counts[index])
        block.mean = means[index]
        block.scatter = scatters[index]
        block.active_count = active_counts[index].astype(np.int64)
        block.max_abs_activation = max_abs[index]

    return blocked
