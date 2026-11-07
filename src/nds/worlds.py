"""
worlds.py
=========

Rotated worlds (D41, option E, AB's idea): the whole analysis repeated on randomly rotated
neurons, one fixed rotation per world, so that every readout, comparison and event search has a
null that went through exactly the same pipeline.

How a world is built
--------------------
A rotation Q mixes neurons, so it needs one fixed set of neurons across the run. We use the
neurons that are alive and not massive at every checkpoint of the run (`always_alive_neurons`),
restrict each checkpoint's moments to that set, and rotate them:

    mean'    = Q^T mean
    scatter' = Q^T scatter Q        (so covariance' = Q^T C Q)

for every bootstrap block, so the bootstrap still works inside a world when blocks were saved
(D48). Firing rates and maximum activations do not rotate, so worlds are analysed with the dead-
and massive-neuron filters switched off (everything in the set is alive by construction).

World 0 is the same neuron set with Q = identity: the real neurons, restricted to the same set and
analysed with the same switches. The rank test compares world 0 with worlds 1 to 19, so the only
difference between them is the rotation. (The main analysis of the real run, on all live neurons,
stays as it is for the figures.)

The rank test
-------------
For a readout where "larger means more neuron-level structure", the real value r and the world
values w_1 .. w_W give

    p = (1 + #{w_i >= r}) / (1 + W)

the smallest possible p with W = 19 worlds is 0.05 (Phipson & Smyth 2010 for the +1). On the
planted runs (research/round4, S1) localisation, ARI changes and module lineage passed this test at
real events; the subspace distance did not separate (a fixed rotation keeps the angles between
subspaces), so it is not used for neuron-level claims.

Credit
------
- F. Mezzadri (2007). How to generate random matrices from the classical compact groups. Notices
  of the AMS 54(5). (Haar rotations; nds.controls.random_orthogonal_matrix)
- B. Phipson, G. K. Smyth (2010). Permutation p-values should never be zero. Statistical
  Applications in Genetics and Molecular Biology 9(1).
"""

from pathlib import Path

import numpy as np

from nds import storage
from nds.controls import random_orthogonal_matrix
from nds.moments import BlockedNeuronMoments

NUMBER_OF_WORLDS = 19


def always_alive_neurons(run_folder, layer):
    """
    Model neuron numbers that are in the analysed graph (alive, not massive) at every finished
    checkpoint of the run, from the saved spectra. Returns (neuron ids, mean number of live neurons
    per checkpoint), so the coverage of the worlds can be reported.
    """
    run_folder = Path(run_folder)

    common = None
    live_counts = []

    for step in storage.finished_steps(run_folder):
        spectrum = storage.load_layer_spectrum(storage.step_folder(run_folder, step), layer)
        ids = np.asarray(spectrum["live_neuron_ids"], dtype=np.int64)
        live_counts.append(len(ids))
        if common is None:
            common = ids
        else:
            common = np.intersect1d(common, ids)

    if common is None:
        return np.zeros(0, dtype=np.int64), 0.0

    return np.sort(common), float(np.mean(live_counts))


def world_rotation(size, world_index, worlds_seed=0):
    """
    The fixed rotation of one world: identity for world 0, otherwise a Haar rotation drawn from a
    generator seeded with (worlds_seed, world_index), so any world can be rebuilt on its own.
    """
    if int(world_index) == 0:
        return np.eye(size)

    generator = np.random.default_rng([int(worlds_seed), int(world_index)])

    return random_orthogonal_matrix(size, generator)


def world_moments(blocked_moments, neuron_ids, rotation):
    """
    The moments of one checkpoint restricted to `neuron_ids` and rotated by `rotation`, block by
    block. Firing rates are set to 1 and maximum activations to 0 (they do not rotate; worlds are
    analysed with those filters off).
    """
    neuron_ids = np.asarray(neuron_ids, dtype=np.int64)
    size = len(neuron_ids)

    result = BlockedNeuronMoments(size, number_of_blocks=blocked_moments.number_of_blocks)

    for index in range(blocked_moments.number_of_blocks):
        source = blocked_moments.blocks[index]
        target = result.blocks[index]

        target.count = int(source.count)

        mean = np.asarray(source.mean, dtype=np.float64)[neuron_ids]
        target.mean = rotation.T @ mean

        scatter = np.asarray(source.scatter, dtype=np.float64)[np.ix_(neuron_ids, neuron_ids)]
        rotated = rotation.T @ scatter @ rotation
        target.scatter = (rotated + rotated.T) / 2.0

        target.active_count = np.full(size, int(source.count), dtype=np.int64)
        target.max_abs_activation = np.zeros(size, dtype=np.float64)

    return result


def world_settings(settings):
    """
    The analysis settings inside a world: same graph and readouts, dead and massive filters off,
    no per-checkpoint rotations (the worlds are the rotation control).
    """
    inside = dict(settings or {})
    inside["exclude_massive_neurons"] = False
    inside["minimum_firing_rate"] = 0.0
    inside["number_of_rotations"] = 0
    return inside


def world_rank_p_values(real_values, world_values):
    """
    Rank test of the real run (world 0) against the rotated worlds, per checkpoint (or per
    transition).

    real_values    (n,) the readout in world 0
    world_values   (W, n) the same readout in worlds 1 .. W
    Returns (n,) p-values, (1 + number of worlds at least as large) / (1 + number of worlds with a
    value); NaN where the real value is missing.
    """
    real = np.asarray(real_values, dtype=np.float64)
    worlds = np.atleast_2d(np.asarray(world_values, dtype=np.float64))

    p_values = np.full(real.shape, np.nan)

    for index in range(len(real)):
        if not np.isfinite(real[index]):
            continue
        column = worlds[:, index]
        column = column[np.isfinite(column)]
        at_least = int(np.sum(column >= real[index]))
        p_values[index] = (1.0 + at_least) / (1.0 + len(column))

    return p_values


def world_run_name(run_name, world_index):
    """
    Folder name of one world of a run: <run>_world00 is the real neurons on the common set.
    """
    return f"{run_name}_world{int(world_index):02d}"


def change_strength(changes, floor=0.01):
    """
    Each transition's change divided by the run's typical quiet change, max(median, floor). The
    floor matters: ARI between identical partitions changes by exactly 0, which made the ratio
    explode on the planted runs (research/round4, S1).
    """
    changes = np.asarray(changes, dtype=np.float64)

    finite = changes[np.isfinite(changes)]

    typical = floor
    if len(finite) > 0:
        typical = max(float(np.median(finite)), floor)

    return changes / typical


def world_readouts(run_folder, layer):
    """
    The readouts the rank test uses, for one world (a run folder). Single-checkpoint readouts
    (larger = more neuron-level structure) and changes between neighbouring checkpoints (larger =
    more change), all aligned with the checkpoints (changes: first entry NaN).
    """
    from nds.timeline import module_lineage
    from nds.timeline import partition_stability_series
    from nds.timeline import scalar_table

    table = scalar_table(run_folder, layer)

    stability = partition_stability_series(run_folder, layer)

    lineage_loss = [np.nan]
    for transition in module_lineage(run_folder, layer):
        lineage_loss.append(1.0 - float(np.min(transition["overlap"])))

    return {
        "step": np.asarray(table["step"]),
        "single": {
            "module_aligned_localization": np.asarray(table["module_aligned_localization"], dtype=np.float64),
            "module_aligned_support": np.asarray(table["module_aligned_support"], dtype=np.float64),
        },
        "change": {
            "one_minus_ari": 1.0 - np.asarray(stability, dtype=np.float64),
            "lineage_loss": np.asarray(lineage_loss, dtype=np.float64),
        },
    }


def world_rank_table(run_name, layer, number_of_worlds=NUMBER_OF_WORLDS):
    """
    For every readout, the rank-test p-value of world 0 (the real neurons) against worlds 1 .. W at
    each checkpoint. Changes are compared as strengths (change_strength), so each world judges its
    own changes against its own quiet stretches.

    Returns a dict: step, and for each readout name (single-checkpoint names, and
    "<change>_strength") a dict with real, world_minimum, world_maximum, p.
    """
    world_folders = []
    for world in range(number_of_worlds + 1):
        folder = storage.run_folder_for(world_run_name(run_name, world))
        if not folder.exists():
            raise FileNotFoundError(f"missing world {world}: run reanalyse.py --worlds 0-{number_of_worlds} first ({folder})")
        world_folders.append(folder)

    readouts = [world_readouts(folder, layer) for folder in world_folders]

    table = {"step": readouts[0]["step"]}

    for name in readouts[0]["single"].keys():
        real = readouts[0]["single"][name]
        others = np.array([readout["single"][name] for readout in readouts[1:]])
        table[name] = {
            "real": real,
            "world_minimum": np.nanmin(others, axis=0),
            "world_maximum": np.nanmax(others, axis=0),
            "p": world_rank_p_values(real, others),
        }

    # a change of module membership is only meaningful where modules exist: the real neurons must
    # beat every world on IPR or L1 localisation at BOTH ends of the transition. Without this gate
    # a run that is noisy early (no modules yet, partitions random) and quiet later has a small
    # median change, so its early noise looks like strong events, while every world is noisy
    # throughout (found on the gallery's synthetic run, research/round5, 23 Oct 2026).
    smallest = 1.0 / (1.0 + number_of_worlds) + 1e-12
    structured = (table["module_aligned_localization"]["p"] <= smallest) | (table["module_aligned_support"]["p"] <= smallest)
    both_ends = np.zeros(len(structured), dtype=bool)
    both_ends[1:] = structured[1:] & structured[:-1]

    for name in readouts[0]["change"].keys():
        real = change_strength(readouts[0]["change"][name])
        others = np.array([change_strength(readout["change"][name]) for readout in readouts[1:]])
        p_values = world_rank_p_values(real, others)
        gated = np.where(both_ends, p_values, np.nan)
        table[name + "_strength"] = {
            "real": real,
            "world_minimum": np.nanmin(others, axis=0),
            "world_maximum": np.nanmax(others, axis=0),
            "p": p_values,
            "p_where_modules_exist": gated,
        }

    return table
