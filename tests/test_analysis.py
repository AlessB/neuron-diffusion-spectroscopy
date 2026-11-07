"""
analyse_layer end to end on fake data: modules found, rotation removes them, baselines unchanged.
"""

import numpy as np

from nds.analysis import analyse_layer
from nds.moments import BlockedNeuronMoments
from nds.synthetic import modular_activations


def test_analyse_layer_on_modular_data():
    activations, _ = modular_activations(
        number_of_tokens=4000,
        module_sizes=(30, 30, 30),
        module_strength=0.7,
    )

    blocked = BlockedNeuronMoments(number_of_neurons=90, number_of_blocks=8)

    for block_index in range(8):
        rows = activations[block_index * 500:(block_index + 1) * 500]
        blocked.add_batch(rows, block_index)

    result = analyse_layer(
        blocked,
        settings={
            "neighbour_rank": 8,
            "number_of_bootstrap_replicates": 3,
            "minimum_firing_rate": 0.0,
        },
    )

    assert result["spectrum"]["module_count"] == 3
    assert result["rotations"][0]["spectrum"]["module_count"] != 3

    for name, value in result["baselines"].items():
        assert np.isclose(value, result["rotations"][0]["baselines"][name], rtol=1e-6), name

    bootstrap_counts = [replicate["module_count"] for replicate in result["bootstrap"]]
    assert all(count == 3 for count in bootstrap_counts)

    assert len(result["spectrum"]["live_neuron_ids"]) == 90
