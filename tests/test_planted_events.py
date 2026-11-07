"""
A synthetic "training run" with a planted event, through the real pipeline: moments ->
analyse_layer -> storage -> scalar_table and the across-checkpoint series. Two kinds of event:

- "forming":   no modules, then 4 modules of 32 neurons switch on at checkpoint index 2.
               Expected: module count, RankMe and ALIGNED localisation all jump; the rotated
               control's aligned localisation does not; ARI drops once.
- "reshuffle": the same 4 modules throughout, but at index 2 half of the neurons swap modules.
               Expected: RankMe, module count and localisation stay flat (same spectrum, same
               concentration), ARI drops once, the slow subspace turns once. A "hidden" event.

Found while writing this (22 Oct 2026): subspace_localization is blind to a partition that
covers every neuron evenly (its leverage is uniform), which is why aligned_localization exists.
"""

import numpy as np

from nds import storage
from nds.analysis import analyse_layer
from nds.analysis import layer_summary
from nds.moments import BlockedNeuronMoments
from nds.timeline import partition_stability_series
from nds.timeline import scalar_table
from nds.timeline import subspace_change_series

NEURONS = 128
TOKENS = 4096
BLOCKS = 4
STEPS = [0, 1000, 2000, 3000, 4000]
ONSET = 2


def planted_activations(scenario, index, generator, labels_before, labels_after):
    latents = generator.standard_normal((TOKENS, 4))
    noise = generator.standard_normal((TOKENS, NEURONS))

    if scenario == "forming":
        labels = labels_before
        if index < ONSET:
            strength = 0.0
        else:
            strength = 0.6
    else:
        strength = 0.6
        if index < ONSET:
            labels = labels_before
        else:
            labels = labels_after

    values = np.sqrt(strength) * latents[:, labels] + np.sqrt(1.0 - strength) * noise

    return values


def write_run(scenario, run_folder):
    generator = np.random.default_rng(0)

    labels_before = np.repeat(np.arange(4), NEURONS // 4)
    labels_after = labels_before.copy()
    swapped = generator.permutation(NEURONS)[: NEURONS // 2]
    labels_after[swapped] = generator.permutation(labels_after[swapped])

    for index, step in enumerate(STEPS):
        values = planted_activations(scenario, index, generator, labels_before, labels_after)

        blocked = BlockedNeuronMoments(NEURONS, number_of_blocks=BLOCKS)
        rows_per_block = TOKENS // BLOCKS
        for block in range(BLOCKS):
            blocked.add_batch(values[block * rows_per_block:(block + 1) * rows_per_block], block)

        result = analyse_layer(
            blocked,
            settings={"module_partition": True, "minimum_firing_rate": 0.0, "number_of_rotations": 2, "number_of_bootstrap_replicates": 2},
            effective_tokens=float(TOKENS),
        )

        folder = storage.step_folder(run_folder, step)
        storage.save_layer_spectrum(folder, 0, result["spectrum"])
        storage.save_summary(
            folder,
            {
                "model_name": scenario,
                "step": step,
                "tokens_seen": 0,
                "probe_loss": 1.0,
                "induction": {"induction_loss_drop": 0.0},
                "layers": {0: layer_summary(result)},
            },
        )


def test_forming_modules_are_seen_by_aligned_localization_and_not_by_the_rotated_control(tmp_path):
    run_folder = tmp_path / "forming"
    write_run("forming", run_folder)

    table = scalar_table(run_folder, 0)
    ari = partition_stability_series(run_folder, 0)

    before = table["module_aligned_localization"][:ONSET]
    after = table["module_aligned_localization"][ONSET:]
    rotated_after = table["rotated_module_aligned_max"][ONSET:]

    # 4 modules of 32 neurons out of 128: m / (3 s) = 128 / 96
    assert np.allclose(after, 128.0 / 96.0, atol=0.02)
    assert before.max() < 1.15
    assert rotated_after.max() < 1.15
    assert table["module_count"][ONSET:].min() == 4
    assert table["rankme_li"][ONSET:].max() < table["rankme_li"][:ONSET].min() * 0.7

    # the old measure cannot see a complete, even partition (documented limitation)
    assert abs(table["first_ten_subspace_localization"][ONSET:].mean() - table["first_ten_subspace_localization"][:ONSET].mean()) < 0.15

    # membership changes once, then is stable
    assert ari[ONSET] < 0.3
    assert np.all(ari[ONSET + 1:] > 0.95)


def test_reshuffled_modules_are_a_hidden_event(tmp_path):
    run_folder = tmp_path / "reshuffle"
    write_run("reshuffle", run_folder)

    table = scalar_table(run_folder, 0)
    ari = partition_stability_series(run_folder, 0)
    turn = subspace_change_series(run_folder, 0)

    # the covariance baselines and the count see nothing
    rankme = table["rankme_li"]
    assert np.ptp(rankme) < 0.1 * rankme.mean()
    assert np.ptp(table["module_count"]) == 0

    # localisation stays flat too: same modules, same concentration
    aligned = table["module_aligned_localization"]
    assert np.ptp(aligned) < 0.02

    # the partition and the subspace change exactly once, at the onset
    assert ari[ONSET] < 0.5
    assert np.all(ari[[1, 3, 4]] > 0.95)
    assert turn[ONSET] > 3 * np.nanmax(turn[[1, 3, 4]])


def test_module_lineage_follows_modules_and_sees_the_reshuffle(tmp_path):
    from nds.timeline import module_lineage

    run_folder = tmp_path / "reshuffle"
    write_run("reshuffle", run_folder)

    transitions = module_lineage(run_folder, 0)

    # steady transitions: every module continues with overlap near 1
    for index in [0, 2, 3]:
        assert transitions[index]["born"] == 0
        assert np.min(transitions[index]["overlap"]) > 0.95

    # the reshuffle (into index 2): each module keeps about 5/8 of its neurons (half are reassigned
    # at random, a quarter of those land back in the same module), so the overlap falls to about
    # 0.6 to 0.7 while the spectrum and the module count do not move
    assert np.max(transitions[ONSET - 1]["overlap"]) < 0.8
