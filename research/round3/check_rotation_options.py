"""
D41: which way of drawing the rotated control is best? (22 Oct 2026, asked by AB)

Four options for the random-rotation control, compared on synthetic training runs:

  A  fresh:      5 new Haar rotations at every checkpoint, of the neurons alive there (the code now)
  B  one fixed:  1 rotation, the same at every checkpoint, of all neurons (dead ones included)
  C  mixed:      A for numbers measured at one checkpoint, B for comparisons between checkpoints
  D  five fixed: 5 rotations, the same 5 at every checkpoint, of the neurons alive at EVERY
                 checkpoint (known only after the run, so computed at analysis time)

C uses A's numbers for single checkpoints and B's for comparisons, so it is reported through
those two.

Two synthetic runs of 6 checkpoints, 256 neurons (8 modules of 24 plus 64 free neurons), fresh
tokens at every checkpoint, and a set of dead neurons that changes at checkpoint 2 (20% dead,
then a different 10%), as dead neurons do during real training:
  stable     the modules never change: every change seen is noise
  reshuffle  at checkpoint 4 half of the module neurons swap modules (a neuron-level event)

What is measured
  single checkpoint, stable run:
    level    mean rotated value of the localisation readouts (the baseline the original must beat)
    spread   range over the rotations at one checkpoint (A and D have 5, B has 1)
    jitter   typical change of the rotated mean from one checkpoint to the next (a control that
             jumps around makes "the original jumped more than the control" easy to fake or hide)
  between checkpoints:
    stable run: the change the CONTROL reports when nothing changed (1 - ARI, subspace distance)
                -> the noise floor an event must exceed
    reshuffle run: the change the ORIGINAL reports at the event, against that floor

Repeated over 3 random seeds.

    python research/round3/check_rotation_options.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from nds.analysis import merged_settings
from nds.analysis import spectrum_numbers
from nds.controls import random_orthogonal_matrix
from nds.controls import rotate_covariance
from nds.flow import restrict_to_common_neurons
from nds.spectra import varimax_rotation
from nds.timeline import partition_stability
from nds.timeline import subspace_distance

NEURONS = 256
TOKENS = 4000
CHECKPOINTS = 6
EVENT = 4
ROTATIONS = 5

SETTINGS = merged_settings({"module_partition": True, "fixed_bandwidth_dse": False})

READOUTS = ["module_aligned_localization", "module_aligned_support", "first_ten_subspace_localization"]


def make_run(scenario, generator):
    labels_before = np.full(NEURONS, -1)
    labels_before[:192] = np.repeat(np.arange(8), 24)
    labels_after = labels_before.copy()
    swapped = generator.permutation(192)[:96]
    labels_after[swapped] = generator.permutation(labels_after[swapped])

    dead_early = generator.permutation(NEURONS)[: NEURONS // 5]
    dead_late = generator.permutation(NEURONS)[: NEURONS // 10]

    covariances = []
    for index in range(CHECKPOINTS):
        labels = labels_before
        if scenario == "reshuffle" and index >= EVENT:
            labels = labels_after

        latents = generator.standard_normal((TOKENS, 8))
        values = np.sqrt(0.5) * generator.standard_normal((TOKENS, NEURONS))
        in_module = labels >= 0
        values[:, in_module] += np.sqrt(0.5) * latents[:, labels[in_module]]
        values[:, ~in_module] += np.sqrt(0.5) * generator.standard_normal((TOKENS, int(np.sum(~in_module))))

        if index < 2:
            values[:, dead_early] = 0.0
        else:
            values[:, dead_late] = 0.0

        covariances.append(np.cov(values, rowvar=False))

    return covariances


def summarise(covariance, neuron_ids):
    """
    Graph, spectrum and partition of one (possibly rotated) covariance. neuron_ids names the rows,
    so units can be matched across checkpoints.
    """
    spectrum = spectrum_numbers(covariance, None, SETTINGS)
    live = spectrum["live_indices"]
    module_count = max(int(spectrum["module_count"]), 2)
    basis = varimax_rotation(spectrum["symmetric_vectors"][:, 0:module_count])
    return {
        "values": {name: spectrum[name] for name in READOUTS},
        "labels": spectrum["module_labels"],
        "ids": np.asarray(neuron_ids)[live],
        "basis": basis,
    }


def compare(first, second):
    ari = partition_stability(first["labels"], first["ids"], second["labels"], second["ids"])
    a, b, _, _ = restrict_to_common_neurons(first["basis"], first["ids"], second["basis"], second["ids"])
    return {"one_minus_ari": 1.0 - ari, "subspace_distance": subspace_distance(a, b)}


def analyse_run(covariances, generator):
    always_alive = np.arange(NEURONS)
    for covariance in covariances:
        always_alive = np.intersect1d(always_alive, np.flatnonzero(np.diag(covariance) > 1e-10))

    fixed_full = random_orthogonal_matrix(NEURONS, generator)
    fixed_alive = []
    for _ in range(ROTATIONS):
        fixed_alive.append(random_orthogonal_matrix(len(always_alive), generator))

    per_checkpoint = {"original": [], "A": [], "B": [], "D": []}

    for covariance in covariances:
        per_checkpoint["original"].append([summarise(covariance, np.arange(NEURONS))])

        # A: fresh rotations of the neurons alive at this checkpoint; rotated units have no identity
        alive_here = np.flatnonzero(np.diag(covariance) > 1e-10)
        block = covariance[np.ix_(alive_here, alive_here)]
        fresh = []
        for _ in range(ROTATIONS):
            rotation = random_orthogonal_matrix(len(alive_here), generator)
            fresh.append(summarise(rotate_covariance(block, rotation), np.arange(len(alive_here))))
        per_checkpoint["A"].append(fresh)

        # B: one fixed rotation of all neurons, dead ones included
        per_checkpoint["B"].append([summarise(rotate_covariance(covariance, fixed_full), np.arange(NEURONS))])

        # D: five fixed rotations of the neurons alive at every checkpoint
        block = covariance[np.ix_(always_alive, always_alive)]
        fixed = []
        for rotation in fixed_alive:
            fixed.append(summarise(rotate_covariance(block, rotation), np.arange(len(always_alive))))
        per_checkpoint["D"].append(fixed)

    return per_checkpoint


def main():
    results = {"stable": [], "reshuffle": []}
    for seed in range(3):
        for scenario in results:
            generator = np.random.default_rng(seed)
            covariances = make_run(scenario, generator)
            results[scenario].append(analyse_run(covariances, generator))
        print(f"seed {seed} done", flush=True)

    print("\n== single checkpoint, stable run (mean over seeds and checkpoints) ==")
    print(f"{'readout':34s} {'option':9s} {'level':>7s} {'spread':>7s} {'jitter':>7s} {'original - max rotated':>24s}")
    for name in READOUTS:
        for option in ["A", "B", "D"]:
            levels, spreads, jitters, margins = [], [], [], []
            for run in results["stable"]:
                means = []
                for index in range(CHECKPOINTS):
                    rotated = [entry["values"][name] for entry in run[option][index]]
                    original = run["original"][index][0]["values"][name]
                    means.append(np.mean(rotated))
                    spreads.append(np.max(rotated) - np.min(rotated))
                    margins.append(original - np.max(rotated))
                levels.append(np.mean(means))
                jitters.append(np.mean(np.abs(np.diff(means))))
            print(f"{name:34s} {option:9s} {np.mean(levels):7.3f} {np.mean(spreads):7.3f} {np.mean(jitters):7.3f} {np.mean(margins):24.3f}")
        originals = []
        for run in results["stable"]:
            for index in range(CHECKPOINTS):
                originals.append(run["original"][index][0]["values"][name])
        print(f"{name:34s} {'original':9s} {np.mean(originals):7.3f}")

    print("\n== between checkpoints (mean over seeds) ==")
    print("stable run: change reported between neighbouring checkpoints where nothing changed")
    print("reshuffle run: change reported at the event (checkpoint 3 -> 4)")
    print(f"{'measure':18s} {'who':10s} {'stable, no event':>18s} {'at the event':>14s}")
    for measure in ["one_minus_ari", "subspace_distance"]:
        for who in ["original", "A", "B", "D"]:
            stable_values = []
            for run in results["stable"]:
                for index in range(1, CHECKPOINTS):
                    for entry_from, entry_to in zip(run[who][index - 1], run[who][index]):
                        stable_values.append(compare(entry_from, entry_to)[measure])
            event_values = []
            for run in results["reshuffle"]:
                for entry_from, entry_to in zip(run[who][EVENT - 1], run[who][EVENT]):
                    event_values.append(compare(entry_from, entry_to)[measure])
            print(f"{measure:18s} {who:10s} {np.mean(stable_values):18.3f} {np.mean(event_values):14.3f}")


if __name__ == "__main__":
    main()
