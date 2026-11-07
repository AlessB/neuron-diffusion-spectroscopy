"""
Data for the D43 results gallery (AB: "make artifacts exploring all possible different ways of
presenting and interpreting the results; then we refine for the project paper").

A SYNTHETIC training run, built so that every kind of event we expect is present once, and analysed
with the real pipeline (nds), so the gallery shows what our figures would look like on real data.
Nothing here is a result about Pythia.

Three layers on Pythia's week-1 grid (25 checkpoints), 192 neurons each, the same 3,000 probe tokens
at every checkpoint (shared probe text), the network drifting slowly on the log clock:
  layer 0  modules FORM: 6 modules of 24 neurons switch on gradually between steps 32 and 512
           (an "explained" event: the eigenvalue baseline RankMe moves too)
  layer 1  modules exist from step 64; at the transition 1,000 -> 2,000 half the module neurons
           change module (a "hidden" event: spectrum and RankMe unchanged)
  layer 2  two modules HYBRIDISE: 12 bridge neurons rotate from module A to module B between steps
           2,000 and 16,000 (the avoided-crossing picture); a NEW module of 24 free neurons is born
           between steps 24,000 and 48,000
9 rotated worlds per layer (the real analysis uses 19).

    python research/round5/gallery_data.py      -> research/round5/gallery_data.json
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from nds.analysis import merged_settings
from nds.analysis import spectrum_numbers
from nds.baselines import baseline_summary
from nds.checkpoints import week_one_steps
from nds.controls import random_orthogonal_matrix
from nds.controls import rotate_covariance
from nds.flow import restrict_to_common_neurons
from nds.moments import BlockedNeuronMoments
from nds.spectra import chordal_subspace_distance
from nds.spectra import varimax_rotation
from nds.timeline import partition_stability
from nds.timeline import training_time
from nds.worlds import change_strength
from nds.worlds import world_rank_p_values

NEURONS = 192
TOKENS = 3000
BLOCKS = 6
WORLDS = 9
DRIFT_RATE = 0.02
SETTINGS = merged_settings({"minimum_firing_rate": 0.0, "module_partition": True, "fixed_bandwidth_dse": False})
STEPS = week_one_steps()
TIMES = training_time(STEPS)


def ramp(time, start_step, end_step):
    start = np.log10(start_step + 1.0)
    end = np.log10(end_step + 1.0)
    return float(np.clip((time - start) / (end - start), 0.0, 1.0))


def loadings(layer, index, generator_state):
    """
    Loading matrix (neurons x latents) of a layer at a checkpoint. Latents: 8 module latents.
    """
    time = TIMES[index]
    step = STEPS[index]
    matrix = np.zeros((NEURONS, 8))
    labels = np.full(NEURONS, -1)
    if layer == 0:
        strength = 0.45 * ramp(time, 32, 512)
        labels[:144] = np.repeat(np.arange(6), 24)
        for neuron in range(144):
            matrix[neuron, labels[neuron]] = np.sqrt(strength)
    if layer == 1:
        strength = 0.35 * ramp(time, 16, 64)
        labels[:144] = np.repeat(np.arange(6), 24)
        if step >= 2000:
            labels = generator_state["reshuffled"]
        for neuron in range(144):
            matrix[neuron, labels[neuron]] = np.sqrt(strength)
    if layer == 2:
        strength = 0.35
        labels[:120] = np.repeat(np.arange(5), 24)
        for neuron in range(120):
            matrix[neuron, labels[neuron]] = np.sqrt(strength)
        angle = (np.pi / 2.0) * ramp(time, 2000, 16000)
        bridge = np.arange(12)
        matrix[bridge, 0] = np.sqrt(strength) * np.cos(angle)
        matrix[bridge, 1] = np.sqrt(strength) * np.sin(angle)
        newborn = np.arange(144, 168)
        matrix[newborn, 5] = np.sqrt(0.35 * ramp(time, 24000, 48000))
    return matrix


def make_layer(layer, seed):
    generator = np.random.default_rng(seed)
    reshuffled = np.full(NEURONS, -1)
    reshuffled[:144] = np.repeat(np.arange(6), 24)
    moved = generator.permutation(144)[:72]
    reshuffled[moved] = generator.permutation(reshuffled[moved])
    state = {"reshuffled": reshuffled}

    latents = generator.standard_normal((TOKENS, 8))
    private = generator.standard_normal((TOKENS, NEURONS))
    moments = []
    for index in range(len(STEPS)):
        if index > 0:
            mix = min(DRIFT_RATE * (TIMES[index] - TIMES[index - 1]), 1.0)
            latents = np.sqrt(1 - mix) * latents + np.sqrt(mix) * generator.standard_normal(latents.shape)
            private = np.sqrt(1 - mix) * private + np.sqrt(mix) * generator.standard_normal(private.shape)
        matrix = loadings(layer, index, state)
        signal = np.sum(matrix ** 2, axis=1)
        values = latents @ matrix.T + np.sqrt(np.clip(1.0 - signal, 0.05, None)) * private
        blocked = BlockedNeuronMoments(NEURONS, number_of_blocks=BLOCKS)
        size = TOKENS // BLOCKS
        for block in range(BLOCKS):
            blocked.add_batch(values[block * size:(block + 1) * size], block)
        moments.append(blocked)
    return moments, generator


def summarise(covariance):
    spectrum = spectrum_numbers(covariance, None, SETTINGS)
    count = max(int(spectrum["module_count"]), 2)
    basis = varimax_rotation(spectrum["symmetric_vectors"][:, :count])
    for column in range(basis.shape[1]):
        largest = int(np.argmax(np.abs(basis[:, column])))
        if basis[largest, column] < 0:
            basis[:, column] = -basis[:, column]
    return {
        "loc": float(spectrum["module_aligned_localization"]),
        "l1": float(spectrum["module_aligned_support"]),
        "count": int(spectrum["module_count"]),
        "labels": np.asarray(spectrum["module_labels"]),
        "attached": np.asarray(spectrum["module_attachment"]) >= 0.5,
        "basis": basis,
        "subspace": spectrum["symmetric_vectors"][:, :count],
        "eigenvalues": np.asarray(spectrum["all_laplacian_eigenvalues"])[:14],
        "mode_ipr": np.asarray(spectrum["all_mode_localization"])[:14],
    }


def changes(summaries):
    ids = np.arange(NEURONS)
    one_minus_ari = [np.nan]
    lineage_loss = [np.nan]
    distance = [np.nan]
    for index in range(1, len(summaries)):
        before = summaries[index - 1]
        after = summaries[index]
        one_minus_ari.append(1.0 - partition_stability(before["labels"][before["attached"]], ids[before["attached"]], after["labels"][after["attached"]], ids[after["attached"]]))
        overlaps = np.abs(before["basis"].T @ after["basis"])
        lineage_loss.append(1.0 - float(np.min(np.max(overlaps, axis=1))))
        distance.append(chordal_subspace_distance(before["subspace"], after["subspace"]))
    return np.asarray(one_minus_ari), np.asarray(lineage_loss), np.asarray(distance)


def lineage_links(summaries):
    """
    For each transition, every (module before, module after, overlap) above 0.3: the streams of
    an alluvial diagram.
    """
    links = []
    for index in range(1, len(summaries)):
        overlaps = np.abs(summaries[index - 1]["basis"].T @ summaries[index]["basis"])
        rows = []
        for before in range(overlaps.shape[0]):
            for after in range(overlaps.shape[1]):
                if overlaps[before, after] > 0.3:
                    rows.append([before, after, round(float(overlaps[before, after]), 3)])
        links.append(rows)
    return links


def aligned_membership(summaries):
    """
    Module label of every neuron at every checkpoint, relabelled so that a module keeps its colour
    when it continues (greedy matching of each checkpoint's modules to the previous one's).
    """
    from scipy.optimize import linear_sum_assignment
    result = []
    previous = None
    next_free = 0
    mapping_previous = {}
    for summary in summaries:
        labels = summary["labels"]
        mapping = {}
        if previous is None:
            for label in np.unique(labels):
                mapping[int(label)] = next_free
                next_free += 1
        else:
            current_labels = np.unique(labels)
            previous_labels = np.unique(previous)
            overlap = np.zeros((len(current_labels), len(previous_labels)))
            for row, label in enumerate(current_labels):
                for column, old in enumerate(previous_labels):
                    overlap[row, column] = np.sum((labels == label) & (previous == old))
            rows, columns = linear_sum_assignment(-overlap)
            used = set()
            for row, column in zip(rows, columns):
                if overlap[row, column] >= 0.5 * np.sum(labels == current_labels[row]):
                    mapping[int(current_labels[row])] = mapping_previous[int(previous_labels[column])]
                    used.add(int(current_labels[row]))
            for label in current_labels:
                if int(label) not in mapping:
                    mapping[int(label)] = next_free
                    next_free += 1
        result.append([mapping[int(label)] for label in labels])
        previous = labels
        mapping_previous = mapping
    return result


def embedding(matrix, dimensions=2):
    """
    Diffusion map of the checkpoints from the matrix of subspace distances between all pairs.
    """
    scale = np.median(matrix[matrix > 0])
    kernel = np.exp(-(matrix / scale) ** 2)
    degree = kernel.sum(axis=1)
    symmetric = kernel / np.sqrt(np.outer(degree, degree))
    values, vectors = np.linalg.eigh(symmetric)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order] / np.sqrt(degree)[:, None]
    coordinates = vectors[:, 1:dimensions + 1] * values[1:dimensions + 1]
    return coordinates


def main():
    output = {"steps": [int(step) for step in STEPS], "layers": [], "worlds": WORLDS}
    for layer in range(3):
        moments, generator = make_layer(layer, seed=500 + layer)
        covariances = [moment.merged().covariance() for moment in moments]

        real = [summarise(covariance) for covariance in covariances]
        rotations = [random_orthogonal_matrix(NEURONS, np.random.default_rng([7, world])) for world in range(1, WORLDS + 1)]
        worlds = []
        for rotation in rotations:
            worlds.append([summarise(rotate_covariance(covariance, rotation)) for covariance in covariances])

        bootstrap_loc = []
        bootstrap_l1 = []
        for moment in moments:
            values_loc = []
            values_l1 = []
            for _ in range(4):
                replicate = summarise(moment.bootstrap_replicate(generator).covariance())
                values_loc.append(replicate["loc"])
                values_l1.append(replicate["l1"])
            bootstrap_loc.append([min(values_loc), max(values_loc)])
            bootstrap_l1.append([min(values_l1), max(values_l1)])

        rankme = [float(baseline_summary(covariance)["rankme_li"]) for covariance in covariances]

        real_ari, real_lineage, real_distance = changes(real)
        world_changes = [changes(world) for world in worlds]

        pairwise = np.zeros((len(STEPS), len(STEPS)))
        for first in range(len(STEPS)):
            for second in range(first + 1, len(STEPS)):
                value = chordal_subspace_distance(real[first]["subspace"], real[second]["subspace"])
                pairwise[first, second] = value
                pairwise[second, first] = value

        loc_worlds = np.array([[summary["loc"] for summary in world] for world in worlds])
        l1_worlds = np.array([[summary["l1"] for summary in world] for world in worlds])
        ari_strength_worlds = np.array([change_strength(change[0]) for change in world_changes])
        lineage_strength_worlds = np.array([change_strength(change[1]) for change in world_changes])

        layer_data = {
            "name": ["modules form", "hidden reshuffle", "hybridise, then a module is born"][layer],
            "module_count": [summary["count"] for summary in real],
            "loc": [summary["loc"] for summary in real],
            "l1": [summary["l1"] for summary in real],
            "loc_worlds": loc_worlds.tolist(),
            "l1_worlds": l1_worlds.tolist(),
            "loc_bootstrap": bootstrap_loc,
            "l1_bootstrap": bootstrap_l1,
            "loc_p": world_rank_p_values([summary["loc"] for summary in real], loc_worlds).tolist(),
            "l1_p": world_rank_p_values([summary["l1"] for summary in real], l1_worlds).tolist(),
            "rankme": rankme,
            "one_minus_ari": real_ari.tolist(),
            "lineage_loss": real_lineage.tolist(),
            "subspace_distance": real_distance.tolist(),
            "subspace_distance_worlds": [change[2].tolist() for change in world_changes],
            "ari_strength": change_strength(real_ari).tolist(),
            "ari_strength_worlds": ari_strength_worlds.tolist(),
            "ari_p": world_rank_p_values(change_strength(real_ari), ari_strength_worlds).tolist(),
            "lineage_strength": change_strength(real_lineage).tolist(),
            "lineage_p": world_rank_p_values(change_strength(real_lineage), lineage_strength_worlds).tolist(),
            "eigenvalues": [summary["eigenvalues"].tolist() for summary in real],
            "eigenvalues_world": [summary["eigenvalues"].tolist() for summary in worlds[0]],
            "mode_ipr": [summary["mode_ipr"].tolist() for summary in real],
            "mode_ipr_world": [summary["mode_ipr"].tolist() for summary in worlds[0]],
            "membership": aligned_membership(real),
            "attached": [summary["attached"].astype(int).tolist() for summary in real],
            "lineage_links": lineage_links(real),
            "pairwise_distance": pairwise.tolist(),
            "embedding": embedding(pairwise).tolist(),
        }
        output["layers"].append(layer_data)
        print(f"layer {layer}: counts {layer_data['module_count']}")

    def clean(value):
        if isinstance(value, float):
            if not np.isfinite(value):
                return None
            return round(value, 5)
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        return value

    path = Path(__file__).with_name("gallery_data.json")
    path.write_text(json.dumps(clean(output)), encoding="utf-8")
    print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
