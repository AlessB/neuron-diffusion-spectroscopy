"""
A fake run folder (synthetic avoided-crossing data saved like collect.py saves real data), read
back through timeline.py.
"""

import numpy as np

from nds import storage
from nds.analysis import analyse_layer
from nds.analysis import layer_summary
from nds.moments import BlockedNeuronMoments
from nds.synthetic import three_module_path
from nds.timeline import avoided_crossing_candidates
from nds.timeline import scalar_table
from nds.timeline import spectral_flow


def write_fake_run(run_folder, steps):
    positions = np.linspace(0.0, 1.0, len(steps))

    for step, position in zip(steps, positions):
        activations, _ = three_module_path(position, direct_coupling=0.1)

        number_of_neurons = activations.shape[1]
        blocked = BlockedNeuronMoments(number_of_neurons, number_of_blocks=4)

        rows_per_block = activations.shape[0] // 4
        for block_index in range(4):
            rows = activations[block_index * rows_per_block:(block_index + 1) * rows_per_block]
            blocked.add_batch(rows, block_index)

        result = analyse_layer(
            blocked,
            settings={
                "minimum_firing_rate": 0.0,
                "number_of_bootstrap_replicates": 2,
                "number_of_modes": 20,
            },
        )

        folder = storage.step_folder(run_folder, step)
        storage.save_layer_spectrum(folder, 0, result["spectrum"])

        summary = {
            "step": step,
            "probe_loss": 1.0,
            "induction": {"induction_loss_drop": 0.0},
            "layers": {
                0: layer_summary(result),
            },
        }
        storage.save_summary(folder, summary)


def test_timeline_reads_a_fake_run(tmp_path):
    steps = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    write_fake_run(tmp_path, steps)

    table = scalar_table(tmp_path, layer=0)
    assert list(table["step"]) == steps
    assert "dse_t1" in table
    assert "mu5" in table

    flow = spectral_flow(tmp_path, layer=0, number_of_modes=10)
    assert len(flow) == len(steps) - 1
    assert all(np.all(transition["overlap"] <= 1.0 + 1e-6) for transition in flow)

    candidates = avoided_crossing_candidates(table, first_mode=1, last_mode=3, window=3)
    pairs = [(candidate["lower_mode"], candidate["step"]) for candidate in candidates]

    # the synthetic crossing between modes 1 and 2 sits mid-way (steps 16..64)
    assert any(mode == 1 and 8 <= step <= 128 for mode, step in pairs)


def test_coincidence_on_log_steps_and_noise_scale():
    from nds.timeline import bootstrap_noise_scale
    from nds.timeline import coincide

    assert coincide(1000, 1300)
    assert not coincide(1000, 2000)
    assert coincide(0, 0)

    table = {
        "low": np.array([0.9, 1.0, 1.1]),
        "high": np.array([1.1, 1.4, 1.3]),
    }
    assert abs(bootstrap_noise_scale(table, "low", "high") - 0.1) < 1e-12


def test_identity_drift_is_zero_for_a_rescaled_neuron_and_one_for_a_new_one():
    from nds.timeline import neuron_identity_drift

    generator = np.random.default_rng(6)

    before = generator.standard_normal((5000, 3))
    after = before.copy()
    after[:, 0] = 3.0 * before[:, 0] + 1.0
    after[:, 1] = generator.standard_normal(5000)
    after[:, 2] = 0.0

    drift = neuron_identity_drift(before, after)

    assert abs(drift[0]) < 1e-12
    assert abs(drift[1] - 1.0) < 0.05
    assert np.isnan(drift[2])
