"""
neuron_stats.py on data with known moments.
"""

import numpy as np

from nds.neuron_stats import layer_neuron_statistics
from nds.neuron_stats import position_zero_spike_ratio
from nds.neuron_stats import skew_and_kurtosis
from nds.neuron_stats import vocabulary_alignment_moments


def test_skew_and_kurtosis_from_raw_moments_match_direct_ones():
    generator = np.random.default_rng(0)

    values = np.column_stack(
        [
            generator.standard_normal(200000),
            generator.exponential(1.0, 200000),
        ]
    )

    skew, kurtosis = skew_and_kurtosis(
        values.mean(axis=0),
        (values ** 2).mean(axis=0),
        (values ** 3).mean(axis=0),
        (values ** 4).mean(axis=0),
    )

    # Gaussian: skew 0, kurtosis 3. Exponential: skew 2, kurtosis 9.
    assert abs(skew[0]) < 0.05
    assert abs(kurtosis[0] - 3.0) < 0.1
    assert abs(skew[1] - 2.0) < 0.1
    assert abs(kurtosis[1] - 9.0) < 0.8


def test_vocabulary_moments_flag_a_prediction_neuron():
    generator = np.random.default_rng(1)

    vocabulary = 3000
    model_width = 256

    unembedding = generator.standard_normal((vocabulary, model_width))

    # twenty tokens share one direction; neuron 0 writes along it (it promotes those tokens)
    unembedding[:20, 0] += 10.0

    output_weights = generator.standard_normal((model_width, 3))
    output_weights[:, 0] = 0.0
    output_weights[0, 0] = 1.0

    skew, kurtosis = vocabulary_alignment_moments(unembedding, output_weights, chunk_size=2)

    assert kurtosis[0] > 10.0
    assert skew[0] > 0.0
    assert kurtosis[1] < 5.0


def test_position_zero_spike_and_layer_statistics():
    ratio = position_zero_spike_ratio([1.0, 500.0, 2.0], [1.0, 2.0, 3.0])
    assert abs(ratio - 250.0) < 1e-9

    arrays, scalars = layer_neuron_statistics(
        mean=np.zeros(4),
        variance=np.ones(4),
        raw_moments={
            "mean_cube": np.zeros(4),
            "mean_fourth_power": np.full(4, 3.0),
            "positive_fraction": np.array([0.5, 0.01, 0.5, 0.5]),
        },
        position_zero={"mean_absolute": np.ones(4), "max_absolute": np.array([1.0, 1.0, 1.0, 400.0])},
        max_absolute_elsewhere=np.full(4, 2.0),
    )

    assert np.allclose(arrays["activation_kurtosis"], 3.0)
    assert abs(scalars["sparse_neuron_fraction"] - 0.25) < 1e-12
    assert abs(scalars["position_zero_spike_ratio"] - 200.0) < 1e-9
