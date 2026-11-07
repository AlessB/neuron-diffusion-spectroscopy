"""
Shared pieces for the round-4 simulations (22 Oct 2026): synthetic training runs with planted
events, and one function that summarises a (possibly rotated) covariance the way the pipeline does.

Synthetic runs: 256 neurons, 8 modules of 24 plus 64 free neurons, fresh tokens at every
checkpoint, a dead-neuron set that changes at checkpoint 2.
  stable     nothing changes
  reshuffle  at EVENT half of the module neurons swap modules (membership changes, spectrum not)
  forming    modules switch on at EVENT (strength 0 before, 0.5 after)
  growing    module strength rises smoothly from 0 to 0.5 over the run (a crossover, no jump)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from nds.analysis import merged_settings
from nds.analysis import spectrum_numbers
from nds.flow import restrict_to_common_neurons
from nds.spectra import varimax_rotation
from nds.timeline import partition_stability
from nds.timeline import subspace_distance

NEURONS = 256
MODULE_NEURONS = 192
MODULES = 8
TOKENS = 4000
CHECKPOINTS = 8
EVENT = 4

SETTINGS = merged_settings({"module_partition": True, "fixed_bandwidth_dse": False})


def make_run(scenario, generator, tokens=TOKENS, checkpoints=CHECKPOINTS, strength=0.5):
    """
    Returns the list of covariances (one per checkpoint) and the token matrices' sizes.
    """
    labels_before = np.full(NEURONS, -1)
    labels_before[:MODULE_NEURONS] = np.repeat(np.arange(MODULES), MODULE_NEURONS // MODULES)
    labels_after = labels_before.copy()
    swapped = generator.permutation(MODULE_NEURONS)[: MODULE_NEURONS // 2]
    labels_after[swapped] = generator.permutation(labels_after[swapped])

    dead_early = generator.permutation(NEURONS)[: NEURONS // 5]
    dead_late = generator.permutation(NEURONS)[: NEURONS // 10]

    covariances = []
    for index in range(checkpoints):
        labels = labels_before
        current_strength = strength
        if scenario == "reshuffle" and index >= EVENT:
            labels = labels_after
        if scenario == "forming" and index < EVENT:
            current_strength = 0.0
        if scenario == "growing":
            current_strength = strength * index / (checkpoints - 1)

        latents = generator.standard_normal((tokens, MODULES))
        values = np.sqrt(1.0 - current_strength) * generator.standard_normal((tokens, NEURONS))
        in_module = labels >= 0
        values[:, in_module] += np.sqrt(current_strength) * latents[:, labels[in_module]]
        values[:, ~in_module] += np.sqrt(current_strength) * generator.standard_normal((tokens, int(np.sum(~in_module))))

        if index < 2:
            values[:, dead_early] = 0.0
        else:
            values[:, dead_late] = 0.0

        covariances.append(np.cov(values, rowvar=False))

    return covariances


def always_alive(covariances):
    alive = np.arange(covariances[0].shape[0])
    for covariance in covariances:
        alive = np.intersect1d(alive, np.flatnonzero(np.diag(covariance) > 1e-10))
    return alive


def summarise(covariance, neuron_ids):
    """
    What the pipeline computes for one checkpoint: readouts, partition, the varimax module basis,
    the module count and the Laplacian eigenvalues.
    """
    spectrum = spectrum_numbers(covariance, None, SETTINGS)
    live = spectrum["live_indices"]
    module_count = max(int(spectrum["module_count"]), 2)
    basis = varimax_rotation(spectrum["symmetric_vectors"][:, 0:module_count])
    for column in range(basis.shape[1]):
        largest = np.argmax(np.abs(basis[:, column]))
        if basis[largest, column] < 0:
            basis[:, column] = -basis[:, column]
    return {
        "module_aligned_localization": spectrum["module_aligned_localization"],
        "module_aligned_support": spectrum["module_aligned_support"],
        "module_count": int(spectrum["module_count"]),
        "labels": spectrum["module_labels"],
        "ids": np.asarray(neuron_ids)[live],
        "basis": basis,
        "eigenvalues": np.asarray(spectrum["laplacian_eigenvalues"]),
        "vectors": spectrum["symmetric_vectors"],
    }


def compare(first, second):
    """
    Changes between two checkpoints: 1 - ARI, module-subspace distance, and 1 - the weakest
    module's best overlap with the previous checkpoint (module lineage).
    """
    ari = partition_stability(first["labels"], first["ids"], second["labels"], second["ids"])
    a, b, _, _ = restrict_to_common_neurons(first["basis"], first["ids"], second["basis"], second["ids"])
    overlaps = np.abs(a.T @ b)
    weakest = float(np.min(np.max(overlaps, axis=0)))
    return {
        "one_minus_ari": 1.0 - ari,
        "subspace_distance": subspace_distance(a, b),
        "lineage_loss": 1.0 - weakest,
    }
