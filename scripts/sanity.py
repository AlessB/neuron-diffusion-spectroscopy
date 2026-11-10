"""
sanity.py  (P1.4)
=================

One table per run, to check the numbers make sense before reading anything into them, and a second
table about small modules: groups of only a few neurons that split off from the rest of the layer.

    python scripts/sanity.py --config configs/pythia70m-standard_week1.yaml

Reads what collect.py saved (summary.json, layer<L>_spectrum.npz, layer<L>_neuron_stats.npz) and
writes into runs/<run_name>/analysis/:
    sanity.csv                one row per step and layer: probe loss, induction drop, dead fraction,
                              massive neurons, gap ratio, module count and sizes, module-aligned
                              localisation and the rotated maximum
    sanity_small_modules.csv  one row per step and layer whose smallest module holds less than
                              --small-fraction of the live neurons (default 5%): which neurons, how
                              many were in the small module of the previous checkpoint too, and how
                              they behave compared with the rest of the layer (kurtosis, how often
                              they are positive, how strongly they react at position 0)

It also prints both tables and a step-0 check: at step 0 the model is random, so the probe loss
should be near log(vocabulary size) and the induction drop near 0.

Why the second table (9 Nov 2026): in layer 2 of both week-1 runs the module-aligned localisation
jumps to 26 to 35 between steps 3,000 and 48,000 while the module sizes become about [2033, 15].
A small module of rarely active neurons gives a high localisation by itself, so this table says
who those neurons are before the number is read as structure.
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nds import storage

# Pythia's tokenizer has 50,277 tokens (the embedding is padded to 50,304)
PYTHIA_VOCABULARY_SIZE = 50277


def layer_row(summary, layer_key):
    """
    The scalar sanity numbers of one layer at one step, from summary.json.
    """
    layer = summary["layers"][layer_key]
    spectrum = layer["spectrum"]

    rotated_values = []
    for rotation in layer.get("rotations", []):
        rotated_values.append(rotation["spectrum"]["module_aligned_localization"])

    if rotated_values:
        rotated_maximum = max(rotated_values)
    else:
        rotated_maximum = float("nan")

    module_sizes = sorted(spectrum.get("module_sizes", []), reverse=True)

    row = {
        "step": int(summary["step"]),
        "layer": int(layer_key),
        "probe_loss": summary["probe_loss"],
        "induction_loss_drop": summary["induction"]["induction_loss_drop"],
        "dead_fraction": spectrum.get("dead_fraction", float("nan")),
        "massive_neurons": len(layer.get("massive_neurons", [])),
        "gap_ratio": spectrum.get("gap_ratio", float("nan")),
        "module_count": spectrum.get("module_count", 0),
        "module_sizes": " ".join(str(size) for size in module_sizes),
        "module_aligned_localization": spectrum.get("module_aligned_localization", float("nan")),
        "rotated_module_aligned_max": rotated_maximum,
    }

    return row


def small_module_neurons(spectrum_arrays, small_fraction):
    """
    The neuron ids (original numbering) of the smallest module, if it holds less than small_fraction
    of the live neurons; otherwise an empty array. Also returns the mask over live neurons.
    """
    if "module_labels" not in spectrum_arrays:
        return np.array([], dtype=np.int64), None

    labels = np.asarray(spectrum_arrays["module_labels"])
    live_ids = np.asarray(spectrum_arrays["live_neuron_ids"])

    if labels.size == 0:
        return np.array([], dtype=np.int64), None

    sizes = np.bincount(labels)
    sizes_present = np.where(sizes > 0, sizes, np.iinfo(np.int64).max)
    smallest_label = int(np.argmin(sizes_present))
    smallest_size = int(sizes[smallest_label])

    if smallest_size >= small_fraction * labels.size:
        return np.array([], dtype=np.int64), None

    in_small_module = labels == smallest_label

    return live_ids[in_small_module], in_small_module


def median_or_nan(values):
    values = np.asarray(values, dtype=np.float64)

    if values.size == 0:
        return float("nan")

    return float(np.median(values))


def small_module_row(step, layer, neuron_ids, neuron_statistics, previous_ids):
    """
    How the small module's neurons differ from the rest of their layer.
    neuron_statistics arrays are over all neurons of the layer (original numbering), so the
    comparison uses neuron_ids directly.
    """
    number_of_neurons = None
    for values in neuron_statistics.values():
        number_of_neurons = len(values)
        break

    if number_of_neurons is None:
        rest_mask = None
    else:
        rest_mask = np.ones(number_of_neurons, dtype=bool)
        rest_mask[neuron_ids] = False

    row = {
        "step": int(step),
        "layer": int(layer),
        "size": int(len(neuron_ids)),
        "neurons": " ".join(str(int(neuron)) for neuron in neuron_ids),
    }

    if previous_ids is None:
        row["also_in_previous_small_module"] = ""
    else:
        shared = np.intersect1d(neuron_ids, previous_ids)
        row["also_in_previous_small_module"] = int(len(shared))

    for name in [
        "activation_kurtosis",
        "positive_fraction",
        "position_zero_mean_absolute",
    ]:
        if name not in neuron_statistics or rest_mask is None:
            row[f"{name}_small"] = float("nan")
            row[f"{name}_rest"] = float("nan")
            continue
        values = np.asarray(neuron_statistics[name], dtype=np.float64)
        row[f"{name}_small"] = median_or_nan(values[neuron_ids])
        row[f"{name}_rest"] = median_or_nan(values[rest_mask])

    return row


def step_zero_check(rows):
    """
    At step 0 the network is random: probe loss near log(V), induction drop near 0.
    Returns a list of plain-language lines.
    """
    lines = []

    step_zero_rows = [row for row in rows if row["step"] == 0]

    if not step_zero_rows:
        lines.append("step 0: not in this run, nothing to check")
        return lines

    row = step_zero_rows[0]
    expected_loss = math.log(PYTHIA_VOCABULARY_SIZE)
    loss_ok = abs(row["probe_loss"] - expected_loss) < 0.5
    induction_ok = abs(row["induction_loss_drop"]) < 0.1

    loss_word = "ok" if loss_ok else "CHECK"
    induction_word = "ok" if induction_ok else "CHECK"

    lines.append(f"step 0: probe loss {row['probe_loss']:.3f}, log(V) = {expected_loss:.3f}: {loss_word}")
    lines.append(f"step 0: induction drop {row['induction_loss_drop']:.3f} (random network: near 0): {induction_word}")

    return lines


def write_csv(rows, path):
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as csv_file:
            csv_file.write("")
        return

    names = list(rows[0].keys())

    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_number(value):
    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return str(value)
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return f"{value:.3g}"


def print_table(rows, names):
    print("  ".join(names))
    for row in rows:
        values = []
        for name in names:
            values.append(format_number(row[name]))
        print("  ".join(values))


def sanity_tables(run_folder, small_fraction=0.05):
    """
    Build both tables for one run folder. Returns (rows, small_module_rows).
    """
    run_folder = Path(run_folder)
    steps = storage.finished_steps(run_folder)

    rows = []
    small_module_rows = []
    previous_small_ids = {}

    for step in steps:
        folder = storage.step_folder(run_folder, step)
        summary = storage.load_summary(folder)

        layer_keys = sorted(summary["layers"].keys(), key=int)

        for layer_key in layer_keys:
            rows.append(layer_row(summary, layer_key))

            spectrum_path = folder / f"layer{layer_key}_spectrum.npz"
            statistics_path = folder / f"layer{layer_key}_neuron_stats.npz"

            if not spectrum_path.exists():
                continue

            spectrum_arrays = storage.load_layer_spectrum(folder, layer_key)
            neuron_ids, _ = small_module_neurons(spectrum_arrays, small_fraction)

            if len(neuron_ids) == 0:
                previous_small_ids[layer_key] = None
                continue

            if statistics_path.exists():
                neuron_statistics = storage.load_neuron_statistics(folder, layer_key)
            else:
                neuron_statistics = {}

            small_module_rows.append(
                small_module_row(
                    step,
                    layer_key,
                    neuron_ids,
                    neuron_statistics,
                    previous_small_ids.get(layer_key),
                )
            )
            previous_small_ids[layer_key] = neuron_ids

    return rows, small_module_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--small-fraction", type=float, default=0.05)
    arguments = parser.parse_args()

    with open(arguments.config, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    run_folder = storage.run_folder_for(config["run_name"])
    output_folder = run_folder / "analysis"
    output_folder.mkdir(parents=True, exist_ok=True)

    rows, small_module_rows = sanity_tables(run_folder, arguments.small_fraction)

    write_csv(rows, output_folder / "sanity.csv")
    write_csv(small_module_rows, output_folder / "sanity_small_modules.csv")

    print(f"run {config['run_name']}: {len(rows)} rows (step x layer)")
    print()
    print_table(
        rows,
        [
            "step",
            "layer",
            "probe_loss",
            "induction_loss_drop",
            "dead_fraction",
            "massive_neurons",
            "gap_ratio",
            "module_sizes",
            "module_aligned_localization",
            "rotated_module_aligned_max",
        ],
    )
    print()
    for line in step_zero_check(rows):
        print(line)
    print()
    print(f"small modules (under {arguments.small_fraction:.0%} of the live neurons): {len(small_module_rows)} step x layer")
    if small_module_rows:
        print_table(
            small_module_rows,
            [
                "step",
                "layer",
                "size",
                "also_in_previous_small_module",
                "activation_kurtosis_small",
                "activation_kurtosis_rest",
                "positive_fraction_small",
                "positive_fraction_rest",
                "position_zero_mean_absolute_small",
                "position_zero_mean_absolute_rest",
            ],
        )
    print()
    print(f"written to {output_folder / 'sanity.csv'} and {output_folder / 'sanity_small_modules.csv'}")


if __name__ == "__main__":
    main()
