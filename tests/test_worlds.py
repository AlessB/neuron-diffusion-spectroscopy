"""
Rotated worlds (D41 = E): the moments rotate correctly, world 0 is the real neurons, and on a small
planted run the real localisation beats every world while a rotated world does not beat the rest.
"""

import importlib.util
from pathlib import Path

import numpy as np


def test_world_moments_rotate_the_covariance_and_keep_the_mean_structure():
    from nds.moments import BlockedNeuronMoments
    from nds.worlds import world_moments
    from nds.worlds import world_rotation

    generator = np.random.default_rng(0)
    values = generator.standard_normal((400, 10)) @ generator.standard_normal((10, 10))
    blocked = BlockedNeuronMoments(10, number_of_blocks=4)
    for block in range(4):
        blocked.add_batch(values[block * 100:(block + 1) * 100], block)

    neurons = np.array([0, 2, 3, 5, 7, 9])
    rotation = world_rotation(len(neurons), 3, worlds_seed=1)
    rotated = world_moments(blocked, neurons, rotation)

    expected = rotation.T @ np.cov(values[:, neurons], rowvar=False) @ rotation
    assert np.allclose(rotated.merged().covariance(), expected, atol=1e-10)
    assert np.allclose(world_rotation(len(neurons), 0), np.eye(len(neurons)))
    # the same world is rebuilt identically
    assert np.allclose(world_rotation(len(neurons), 3, worlds_seed=1), rotation)


def test_rank_p_values():
    from nds.worlds import world_rank_p_values

    real = np.array([3.0, 1.0, np.nan])
    worlds = np.array([[1.0, 2.0, 1.0], [1.5, 0.5, 1.0], [2.0, 3.0, 1.0]])
    p = world_rank_p_values(real, worlds)
    assert abs(p[0] - 0.25) < 1e-12
    assert abs(p[1] - 0.75) < 1e-12
    assert np.isnan(p[2])


def test_worlds_on_a_small_planted_run(tmp_path, monkeypatch):
    monkeypatch.setenv("NDS_RUNS_DIR", str(tmp_path))

    from nds import storage
    from nds.analysis import analyse_layer
    from nds.analysis import layer_summary
    from nds.moments import BlockedNeuronMoments
    from nds.worlds import world_rank_table

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("reanalyse_for_test", root / "scripts" / "reanalyse.py")
    reanalyse = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reanalyse)

    generator = np.random.default_rng(5)
    neurons = 96
    labels = np.repeat(np.arange(4), 24)
    settings = {"minimum_firing_rate": 0.0, "number_of_bootstrap_replicates": 2, "number_of_rotations": 0, "fixed_bandwidth_dse": False}

    run = storage.run_folder_for("tiny")
    for index in range(3):
        latents = generator.standard_normal((1200, 4))
        values = np.sqrt(0.6) * generator.standard_normal((1200, neurons)) + np.sqrt(0.4) * latents[:, labels]
        blocked = BlockedNeuronMoments(neurons, number_of_blocks=4)
        for block in range(4):
            blocked.add_batch(values[block * 300:(block + 1) * 300], block)
        result = analyse_layer(blocked, settings=settings, effective_tokens=1200)
        folder = storage.step_folder(run, index * 1000)
        storage.save_layer_moments(folder, 0, blocked.merged())
        storage.save_block_moments(folder, 0, blocked)
        storage.save_layer_spectrum(folder, 0, result["spectrum"])
        storage.save_summary(folder, {"model_name": "tiny", "step": index * 1000, "tokens_seen": index, "probe_loss": float("nan"), "induction": {"induction_loss_drop": 0.0}, "layers": {0: layer_summary(result)}})

    for world in range(4):
        output = storage.run_folder_for(f"tiny_world{world:02d}")
        reanalyse.reanalyse_run(run, output, settings, "tiny", world=world, worlds_seed=0)

    table = world_rank_table("tiny", 0, number_of_worlds=3)
    # four equal modules: the real localisation beats all 3 worlds at every checkpoint (p = 1/4)
    assert np.all(table["module_aligned_localization"]["p"] <= 0.25 + 1e-12)
    assert np.all(table["module_aligned_localization"]["real"] > table["module_aligned_localization"]["world_maximum"])
