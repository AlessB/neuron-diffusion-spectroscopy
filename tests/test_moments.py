"""
moments.py must give exactly what numpy gives on the full activation table.
"""

import numpy as np

from nds.moments import NeuronMoments, BlockedNeuronMoments


def test_streaming_covariance_matches_numpy():
    random_generator = np.random.default_rng(0)

    # a large mean on purpose: naive sum-of-squares formulas lose precision here
    activations = 1000.0 + random_generator.standard_normal((1000, 7))

    moments = NeuronMoments(number_of_neurons=7)

    for start in range(0, 1000, 137):
        moments.add_batch(activations[start:start + 137])

    expected = np.cov(activations, rowvar=False)

    assert moments.count == 1000
    assert np.allclose(moments.mean, activations.mean(axis=0))
    assert np.allclose(moments.covariance(), expected, atol=1e-10)


def test_blocks_merge_to_the_whole():
    random_generator = np.random.default_rng(1)
    activations = random_generator.standard_normal((800, 5))

    blocked = BlockedNeuronMoments(number_of_neurons=5, number_of_blocks=4)

    for block_index in range(4):
        rows = activations[block_index * 200:(block_index + 1) * 200]
        blocked.add_batch(rows, block_index)

    merged = blocked.merged()

    assert np.allclose(merged.covariance(), np.cov(activations, rowvar=False))


def test_firing_rate_counts_positive_activations():
    activations = np.array(
        [
            [1.0, -1.0],
            [2.0, -0.5],
            [-1.0, -0.1],
            [3.0, 0.2],
        ]
    )

    moments = NeuronMoments(number_of_neurons=2)
    moments.add_batch(activations)

    assert np.allclose(moments.firing_rate(), [0.75, 0.25])
    assert np.allclose(moments.max_abs_activation, [3.0, 1.0])
