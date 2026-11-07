"""
analyse.py
==========

After collect.py: line the checkpoints up, find candidate events, and draw the standard figure
for every layer (baselines always on the same figure, as PLAN.md requires).

    python scripts/analyse.py --config configs/pythia70m-standard_week1.yaml

Writes into runs/<run_name>/analysis/:
    layer<L>_table.csv            one row per checkpoint, every scalar
    layer<L>_crossings.json       avoided-crossing candidates with two-level fits
    layer<L>_flow.npz             mode overlaps between neighbouring checkpoints
    layer<L>_overview.png         the standard 8-panel figure
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nds import storage
from nds.known_events import KNOWN_EVENTS
from nds.known_events import transitions_in_window
from nds.timeline import circular_shift_p_value
from nds.timeline import avoided_crossing_candidates
from nds.timeline import bootstrap_noise_scale
from nds.timeline import change_points
from nds.timeline import partition_stability_series
from nds.timeline import scalar_table
from nds.timeline import spectral_flow
from nds.timeline import subspace_change_signal_to_noise_series
from nds.timeline import identity_drift_series
from nds.timeline import module_lineage
from nds.timeline import subspace_change_series
from nds.timeline import training_time


def write_table_csv(table, path):
    names = list(table.keys())
    number_of_rows = len(table["step"])

    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(names)

        for row in range(number_of_rows):
            values = []
            for name in names:
                values.append(table[name][row])
            writer.writerow(values)


def overview_figure(table, flow, crossings, layer, path, induction_step=None):
    """
    Eight panels sharing the training-time axis:
      1  Laplacian eigenvalues mu_1..mu_10
      2  module count: original, rotated, bootstrap range
      3  DSE at several diffusion times
      4  baselines: RankMe (Li et al. version) and alpha-ReQ, original and rotated
      5  localisation (PRIMARY, D33): the first-10-mode subspace version with its bootstrap band,
         rotated range and null; and the module subspace after varimax, by IPR and by L1 support,
         each with its rotated range (proposed D49)
      6  partition stability (SECONDARY, D33): adjusted Rand index with the previous checkpoint,
         how far the slow subspace (modes 1-10) turned since the previous checkpoint, and the
         median neuron identity drift (1 - self-correlation on the token sample), and the
         weakest module's overlap with its predecessor (timeline.module_lineage)
      7  token-side baselines: prompt entropy, token-graph DSE (t = 1), TwoNN dimension, and
         knowledge entropy (neuron basis, not a graph) with its rotated control
      8  probe loss and induction loss drop
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = training_time(table["step"])

    figure, axes = plt.subplots(8, 1, figsize=(9, 21), sharex=True)

    # 1. eigenvalues
    for mode in range(1, 11):
        name = f"mu{mode}"
        if name in table:
            axes[0].plot(times, table[name], linewidth=1)
    for crossing in crossings:
        axes[0].axvline(training_time(crossing["step"]), color="grey", alpha=0.3, linestyle=":")
    axes[0].set_ylabel("Laplacian eigenvalue mu_k")
    axes[0].set_title(f"layer {layer}")

    # 2. module count
    axes[1].fill_between(
        times,
        table["bootstrap_module_count_min"],
        table["bootstrap_module_count_max"],
        alpha=0.25,
        label="bootstrap range",
    )
    axes[1].plot(times, table["module_count"], marker="o", label="neuron graph")
    axes[1].plot(times, table["rotated_module_count"], marker="x", label="rotated control")
    axes[1].set_ylabel("module count (eigengap)")
    axes[1].legend(loc="upper left", fontsize=8)

    # 3. DSE
    for name in table:
        if name.startswith("dse_t"):
            axes[2].plot(times, table[name], label=name.replace("dse_", ""))
    axes[2].set_ylabel("DSE (bits)")
    axes[2].legend(loc="upper left", fontsize=7, ncol=4)

    # 4. baselines
    axes[3].plot(times, table["rankme_li"], label="RankMe (Li et al.)")
    axes[3].plot(times, table["rotated_rankme_li"], linestyle="--", label="RankMe, rotated")
    twin = axes[3].twinx()
    twin.plot(times, table["alpha_req"], color="tab:red", label="alpha-ReQ")
    axes[3].set_ylabel("RankMe")
    twin.set_ylabel("alpha-ReQ")
    axes[3].legend(loc="upper left", fontsize=8)

    # 5. localization: the primary readout, which the rotation control must kill
    axes[4].fill_between(
        times,
        table["bootstrap_first_ten_localization_min"],
        table["bootstrap_first_ten_localization_max"],
        alpha=0.25,
        label="bootstrap range",
    )
    axes[4].fill_between(
        times,
        table["rotated_first_ten_localization_min"],
        table["rotated_first_ten_localization_max"],
        alpha=0.25,
        color="tab:orange",
        label="rotated controls (range)",
    )
    axes[4].plot(times, table["first_ten_subspace_localization"], marker="o", label="first 10 modes")
    axes[4].plot(times, table["module_aligned_localization"], marker="D", color="tab:red", label="module subspace, varimax-aligned")
    axes[4].fill_between(
        times,
        table["rotated_module_aligned_min"],
        table["rotated_module_aligned_max"],
        alpha=0.2,
        color="tab:red",
        label="rotated, module-aligned (range)",
    )
    axes[4].plot(times, table["module_aligned_support"], marker="v", color="tab:purple", label="module subspace, L1 support")
    axes[4].fill_between(
        times,
        table["rotated_module_support_min"],
        table["rotated_module_support_max"],
        alpha=0.2,
        color="tab:purple",
        label="rotated, L1 support (range)",
    )
    axes[4].plot(times, table["null_first_ten_subspace_localization"], color="grey", linestyle="--", label="independent-neuron null")
    axes[4].axhline(1.2, color="grey", linestyle=":", label="random subspace (1 + 2/10)")
    axes[4].set_ylabel("subspace localization")
    axes[4].legend(loc="upper left", fontsize=8)

    # 6. partition stability
    axes[5].plot(times, table["partition_stability"], marker="o", color="tab:purple", label="ARI with previous")
    axes[5].set_ylabel("ARI with previous")
    axes[5].set_ylim(-0.1, 1.05)
    twin_change = axes[5].twinx()
    twin_change.plot(times, table["slow_subspace_change"], color="tab:brown", marker=".", label="slow subspace turned")
    twin_change.plot(times, table["median_identity_drift"], color="tab:gray", marker=".", label="median neuron drift")
    axes[5].plot(times, table["module_lineage_min_overlap"], color="tab:olive", marker="s", markersize=3, label="weakest module carried over")
    twin_change.set_ylabel("subspace distance / drift")
    twin_change.set_ylim(-0.05, 1.05)
    axes[5].legend(loc="lower left", fontsize=8)
    twin_change.legend(loc="lower right", fontsize=8)

    # 7. token-side baselines (rotation-invariant)
    axes[6].plot(times, table["prompt_entropy"], label="prompt entropy (normalised)")
    axes[6].plot(times, table["token_graph_dse_t1"] / 11.0, label="token-graph DSE t=1 / 11 bits")
    axes[6].plot(times, table["knowledge_entropy"], color="tab:red", label="knowledge entropy / log m")
    axes[6].plot(times, table["knowledge_entropy_sample_rotated"], color="tab:red", linestyle="--", label="knowledge entropy, rotated (sample)")
    axes[6].set_ylabel("entropies")
    twin_tokens = axes[6].twinx()
    twin_tokens.plot(times, table["twonn_dimension"], color="tab:cyan", label="TwoNN dimension (right axis)")
    twin_tokens.set_ylabel("TwoNN dimension")
    axes[6].legend(loc="upper left", fontsize=8)

    # 8. loss and induction
    axes[7].plot(times, table["probe_loss"], label="probe loss")
    axes[7].plot(times, table["induction_loss_drop"], label="induction loss drop")
    axes[7].set_ylabel("nats")
    axes[7].set_xlabel("log10(step + 1)")
    axes[7].legend(loc="upper right", fontsize=8)

    if induction_step is not None:
        for axis in axes:
            axis.axvline(training_time(induction_step), color="black", alpha=0.4)

    # what other papers report (known_events.py): solid for 70m, dotted for other sizes
    for event in KNOWN_EVENTS:
        if "70m" in event["model"]:
            line_style = "-"
        else:
            line_style = ":"

        for axis in axes:
            if event["first_step"] == event["last_step"]:
                axis.axvline(training_time(event["first_step"]), color="tab:olive", alpha=0.35, linestyle=line_style, linewidth=0.8)
            else:
                axis.axvspan(training_time(event["first_step"]), training_time(event["last_step"]), color="tab:olive", alpha=0.08)

    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()

    with open(arguments.config, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    run_folder = storage.run_folder_for(config["run_name"])
    output_folder = run_folder / "analysis"
    output_folder.mkdir(parents=True, exist_ok=True)

    steps = storage.finished_steps(run_folder)
    if len(steps) < 3:
        print(f"only {len(steps)} finished checkpoints in {run_folder}; need at least 3")
        return

    first_summary = storage.load_summary(storage.step_folder(run_folder, steps[0]))
    layers = sorted(int(layer) for layer in first_summary["layers"].keys())

    for layer in layers:
        table = scalar_table(run_folder, layer)
        table["partition_stability"] = partition_stability_series(run_folder, layer)
        table["slow_subspace_change"] = subspace_change_series(run_folder, layer)
        # in units of the token-noise floor; the fidelity susceptibility d / dt^2 is not used on
        # Pythia's uneven spacing (timeline.subspace_change_signal_to_noise, research/round4/sim6)
        table["subspace_change_signal_to_noise"] = subspace_change_signal_to_noise_series(run_folder, layer)

        # module lineage: how much of each module's membership survives to the next checkpoint
        lineage_minimum = [np.nan]
        lineage_born = [np.nan]
        for transition in module_lineage(run_folder, layer):
            lineage_minimum.append(float(np.min(transition["overlap"])))
            lineage_born.append(transition["born"])
        table["module_lineage_min_overlap"] = np.asarray(lineage_minimum)
        table["modules_born"] = np.asarray(lineage_born, dtype=np.float64)

        # median neuron identity drift between neighbouring checkpoints (needs the token samples)
        try:
            _, drift = identity_drift_series(run_folder, layer)
            median_drift = []
            for row in drift:
                if np.all(np.isnan(row)):
                    median_drift.append(np.nan)
                else:
                    median_drift.append(float(np.nanmedian(row)))
            table["median_identity_drift"] = np.asarray(median_drift)
        except FileNotFoundError:
            table["median_identity_drift"] = np.full(len(table["step"]), np.nan)
        write_table_csv(table, output_folder / f"layer{layer}_table.csv")

        flow = spectral_flow(run_folder, layer)

        overlaps = np.array([transition["overlap"] for transition in flow])
        np.savez(
            output_folder / f"layer{layer}_flow.npz",
            step_from=np.array([transition["step_from"] for transition in flow]),
            step_to=np.array([transition["step_to"] for transition in flow]),
            overlap=overlaps,
        )

        crossings = avoided_crossing_candidates(table)

        with open(output_folder / f"layer{layer}_crossings.json", "w", encoding="utf-8") as json_file:
            json.dump(storage.to_jsonable(crossings), json_file, indent=1)

        # change points (series are standardised inside change_points); one penalty for all,
        # so localization and the baselines are judged by the same rule
        penalty = config.get("analysis_events", {}).get("change_point_penalty", 3.0)

        module_breaks = change_points(table["module_count"], penalty)
        localization_noise = bootstrap_noise_scale(
            table,
            "bootstrap_first_ten_localization_min",
            "bootstrap_first_ten_localization_max",
        )
        localization_breaks = change_points(
            table["first_ten_subspace_localization"],
            penalty,
            noise_scale=localization_noise,
        )
        rankme_breaks = change_points(table["rankme_li"], penalty)

        if module_breaks is None:
            print("ruptures not installed: change points not computed (pip install ruptures)")

        overview_figure(
            table,
            flow,
            crossings,
            layer,
            output_folder / f"layer{layer}_overview.png",
            induction_step=config.get("induction", {}).get("known_onset_step"),
        )

        # is the primary readout's change concentrated in the 512 to 1.6k window (He et al.)?
        # changes per unit of log training time, circular-shift null (timeline.circular_shift_p_value)
        steps_list = [int(step) for step in table["step"]]
        window = transitions_in_window(steps_list, 512, 1600)
        rates = np.diff(table["first_ten_subspace_localization"]) / np.diff(training_time(table["step"]))
        enrichment, p_value = circular_shift_p_value(rates, window)
        rankme_rates = np.diff(table["rankme_li"]) / np.diff(training_time(table["step"]))
        rankme_enrichment, rankme_p = circular_shift_p_value(rankme_rates, window)

        print(
            f"layer {layer}: localization change in steps 512-1.6k: {enrichment:.2f} x average "
            f"(circular-shift p = {p_value:.3f}); RankMe {rankme_enrichment:.2f} x (p = {rankme_p:.3f})"
        )

        print(
            f"layer {layer}: {len(table['step'])} checkpoints, "
            f"{len(crossings)} crossing candidates, "
            f"change points (row numbers): modules {module_breaks}, "
            f"localization {localization_breaks}, RankMe {rankme_breaks}"
        )


if __name__ == "__main__":
    main()
