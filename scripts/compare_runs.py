"""
compare_runs.py
===============

Put two or more runs on the same axes, one figure per layer: the standard and deduped models
(DECISIONS D27), the primary kernel and the alpha-decay check (D32), or k = 10 and k = 30.

    python scripts/compare_runs.py --configs configs/pythia70m-standard_week1.yaml configs/pythia70m-deduped_week1.yaml
    python scripts/compare_runs.py --configs A.yaml B.yaml --columns first_ten_subspace_localization rankme_li

Reads what analyse.py would read (the run folders under NDS_RUNS_DIR); it does not need analyse.py
to have run first. Writes runs/comparisons/<name A>__vs__<name B>/layer<L>_compare.png and a CSV of
the chosen columns with one block of columns per run.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nds import storage
from nds.timeline import partition_stability_series
from nds.timeline import scalar_table
from nds.timeline import subspace_change_series
from nds.timeline import training_time


DEFAULT_COLUMNS = [
    "first_ten_subspace_localization",
    "rotated_first_ten_subspace_localization",
    "module_count",
    "partition_stability",
    "slow_subspace_change",
    "rankme_li",
    "alpha_req",
    "dse_t1",
    "bulk_spacing_ratio",
    "probe_loss",
]


def load_run(config_path):
    """
    The run name and the folder its results are in.
    """
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    run_name = config["run_name"]

    run_folder = storage.run_folder_for(run_name)

    return run_name, run_folder


def layer_table(run_folder, layer):
    """
    scalar_table plus the two series that need the saved eigenvectors.
    """
    table = scalar_table(run_folder, layer)

    table["partition_stability"] = partition_stability_series(run_folder, layer)
    table["slow_subspace_change"] = subspace_change_series(run_folder, layer)

    return table


def comparison_figure(tables, run_names, columns, layer, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(len(columns), 1, figsize=(9, 2.2 * len(columns)), sharex=True)

    if len(columns) == 1:
        axes = [axes]

    for axis, column in zip(axes, columns):
        for table, run_name in zip(tables, run_names):
            if column not in table:
                continue

            times = training_time(table["step"])

            axis.plot(times, table[column], marker=".", label=run_name)

        axis.set_ylabel(column, fontsize=7)

    axes[0].set_title(f"layer {layer}")
    axes[0].legend(loc="best", fontsize=7)
    axes[-1].set_xlabel("log10(step + 1)")

    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def write_comparison_csv(tables, run_names, columns, path):
    """
    One row per step that any run has; a run without that step gets an empty cell.
    """
    all_steps = set()
    for table in tables:
        for step in table["step"]:
            all_steps.add(int(step))

    ordered_steps = sorted(all_steps)

    header = ["step"]
    for run_name in run_names:
        for column in columns:
            header.append(f"{run_name}:{column}")

    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)

        for step in ordered_steps:
            row = [step]

            for table in tables:
                matches = np.flatnonzero(table["step"] == step)

                for column in columns:
                    if len(matches) == 0 or column not in table:
                        row.append("")
                    else:
                        row.append(table[column][matches[0]])

            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--columns", nargs="+", default=DEFAULT_COLUMNS)
    arguments = parser.parse_args()

    run_names = []
    run_folders = []
    for config_path in arguments.configs:
        run_name, run_folder = load_run(config_path)
        run_names.append(run_name)
        run_folders.append(run_folder)

    output_folder = storage.runs_root() / "comparisons" / "__vs__".join(run_names)
    output_folder.mkdir(parents=True, exist_ok=True)

    # the layers every run has
    common_layers = None
    for run_folder in run_folders:
        steps = storage.finished_steps(run_folder)

        if len(steps) == 0:
            print(f"no finished checkpoints in {run_folder}")
            return

        first_summary = storage.load_summary(storage.step_folder(run_folder, steps[0]))

        layers = set()
        for layer in first_summary["layers"].keys():
            layers.add(int(layer))

        if common_layers is None:
            common_layers = layers
        else:
            common_layers = common_layers & layers

    for layer in sorted(common_layers):
        tables = []
        for run_folder in run_folders:
            tables.append(layer_table(run_folder, layer))

        comparison_figure(
            tables,
            run_names,
            arguments.columns,
            layer,
            output_folder / f"layer{layer}_compare.png",
        )

        write_comparison_csv(
            tables,
            run_names,
            arguments.columns,
            output_folder / f"layer{layer}_compare.csv",
        )

        print(f"layer {layer}: compared {len(tables)} runs")

    print(f"written to {output_folder}")


if __name__ == "__main__":
    main()
