"""
Level-spacing ratio (spectra.spacing_ratio) and subspace distance (timeline.subspace_distance),
on random matrices and subspaces with known answers.
"""

import numpy as np

from nds.spectra import spacing_ratio
from nds.timeline import subspace_distance


def test_spacing_ratio_of_independent_levels_is_the_poisson_value():
    generator = np.random.default_rng(0)

    levels = np.sort(generator.uniform(0.0, 1.0, size=20000))

    value = spacing_ratio(levels)

    poisson_value = 2.0 * np.log(2.0) - 1.0
    assert abs(value - poisson_value) < 0.01


def test_spacing_ratio_of_a_random_symmetric_matrix_is_the_goe_value():
    generator = np.random.default_rng(1)

    size = 1500
    gaussian = generator.standard_normal((size, size))
    symmetric = (gaussian + gaussian.T) / 2.0

    eigenvalues = np.linalg.eigvalsh(symmetric)

    edge = size // 10
    value = spacing_ratio(eigenvalues, edge, size - edge)

    assert abs(value - 0.5307) < 0.015


def test_spacing_ratio_handles_too_few_levels():
    assert np.isnan(spacing_ratio([1.0, 2.0]))


def test_subspace_distance_extremes_and_mixing():
    generator = np.random.default_rng(2)

    basis, _ = np.linalg.qr(generator.standard_normal((50, 6)))

    first = basis[:, :3]
    second = basis[:, 3:]

    assert subspace_distance(first, first) < 1e-12
    assert abs(subspace_distance(first, second) - 1.0) < 1e-12

    # mixing the modes inside the subspace is not a change
    mixing, _ = np.linalg.qr(generator.standard_normal((3, 3)))
    assert subspace_distance(first, first @ mixing) < 1e-12

    # one of three directions replaced by an orthogonal one: distance 1/3
    half = np.column_stack([first[:, 0], first[:, 1], second[:, 0]])
    assert abs(subspace_distance(first, half) - 1.0 / 3.0) < 1e-12


def test_knowledge_entropy_extremes_and_rotation_changes_it():
    from nds.baselines import knowledge_entropy
    from nds.baselines import knowledge_entropy_of_sample

    assert abs(knowledge_entropy(np.ones(64)) - 1.0) < 1e-12

    one_neuron = np.zeros(64)
    one_neuron[3] = 5.0
    assert abs(knowledge_entropy(one_neuron)) < 1e-12

    # a few neurons carry all the activation: low entropy; a rotation spreads it out
    generator = np.random.default_rng(3)
    sample = np.zeros((400, 64))
    sample[:, :4] = np.abs(generator.standard_normal((400, 4))) * 5.0

    result = knowledge_entropy_of_sample(sample, number_of_rotations=3, random_generator=generator)

    assert result["original"] < 0.4
    for rotated in result["rotated"]:
        assert rotated > 0.9


def test_circular_shift_null_finds_a_planted_burst_and_not_noise():
    from nds.timeline import circular_shift_p_value

    generator = np.random.default_rng(4)

    changes = generator.normal(0.0, 1.0, size=60)
    window = np.zeros(60, dtype=bool)
    window[20:24] = True

    _, p_noise = circular_shift_p_value(changes, window, number_of_shifts=500, seed=1)
    assert p_noise > 0.05

    planted = changes.copy()
    planted[20:24] = 8.0
    enrichment, p_planted = circular_shift_p_value(planted, window, number_of_shifts=500, seed=1)
    assert enrichment > 3.0
    assert p_planted < 0.05


def test_known_events_and_their_windows():
    from nds.known_events import KNOWN_EVENTS
    from nds.known_events import events_for_model
    from nds.known_events import transitions_in_window

    for event in KNOWN_EVENTS:
        assert event["first_step"] <= event["last_step"]

    assert len(events_for_model("70m")) == 4

    steps = [0, 256, 512, 1000, 2000, 3000]
    flags = transitions_in_window(steps, 512, 1600)
    assert flags == [False, True, True, True, False]


def test_heat_capacity_peaks_higher_with_modules_than_without():
    from nds.graphs import neuron_graph_from_covariance
    from nds.spectra import diffusion_spectrum
    from nds.spectra import heat_capacity_curve

    generator = np.random.default_rng(5)

    # planted modules: a few shared factors, each driving its own block of neurons
    neurons = 240
    tokens = 4000
    factors = generator.standard_normal((tokens, 6))
    loadings = np.zeros((6, neurons))
    for module in range(6):
        loadings[module, module * 20:(module + 1) * 20] = 1.0
    with_modules = factors @ loadings + generator.standard_normal((tokens, neurons))
    without_modules = generator.standard_normal((tokens, neurons))

    peaks = []
    for activations in [with_modules, without_modules]:
        covariance = np.cov(activations, rowvar=False)
        graph = neuron_graph_from_covariance(covariance)
        spectrum = diffusion_spectrum(graph["kernel"])
        curve = heat_capacity_curve(spectrum["laplacian_eigenvalues"])
        peaks.append(curve["peak_heat_capacity"])

    assert peaks[0] > 1.5 * peaks[1]


def test_heat_capacity_of_two_levels_matches_the_schottky_formula():
    from nds.spectra import heat_capacity_curve

    # levels 0 (dropped), 1, 2: after dropping mu_0, a two-level system with gap 1
    taus = np.array([0.5, 1.0, 2.4, 5.0])
    curve = heat_capacity_curve([0.0, 1.0, 2.0], diffusion_scales=taus)

    expected = taus ** 2 * np.exp(-taus) / (1.0 + np.exp(-taus)) ** 2
    assert np.allclose(curve["heat_capacity"], expected)


def test_varimax_recovers_module_indicators_and_support_sees_large_modules():
    from nds.controls import random_orthogonal_matrix
    from nds.spectra import aligned_localization
    from nds.spectra import aligned_support
    from nds.spectra import subspace_localization

    generator = np.random.default_rng(10)

    # three modules covering all 120 neurons, presented in a mixed basis
    labels = np.repeat(np.arange(3), 40)
    indicators = np.zeros((120, 3))
    indicators[np.arange(120), labels] = 1.0 / np.sqrt(40)
    mixing, _ = np.linalg.qr(generator.standard_normal((3, 3)))
    mixed = indicators @ mixing

    # the leverage is flat, and IPR of a third-of-the-layer module equals the random value
    assert abs(subspace_localization(mixed) - 1.0) < 1e-9
    assert abs(aligned_localization(mixed) - 1.0) < 1e-3

    # L1 support sees it: 2 m / (pi s) = 240 / (40 pi)
    assert abs(aligned_support(mixed) - 240.0 / (40.0 * np.pi)) < 1e-3

    # and a rotation of the neurons takes it back to about 1
    rotation = random_orthogonal_matrix(120, generator)
    assert aligned_support(rotation.T @ mixed) < 1.2


def test_fidelity_susceptibility_of_a_two_level_crossing():
    from nds.timeline import fidelity_susceptibility

    # H(t) = [[v t / 2, g], [g, -v t / 2]]: the ground state turns fastest at t = 0, where
    # (d theta / dt)^2 = (v / (4 g))^2
    sweep = 1.0
    coupling = 0.05
    times = np.linspace(-0.5, 0.5, 201)

    ground_states = []
    for time in times:
        hamiltonian = np.array([[sweep * time / 2.0, coupling], [coupling, -sweep * time / 2.0]])
        _, vectors = np.linalg.eigh(hamiltonian)
        ground_states.append(vectors[:, 0])

    distances = []
    for index in range(1, len(times)):
        overlap = float(ground_states[index - 1] @ ground_states[index])
        distances.append(1.0 - overlap ** 2)

    chi = fidelity_susceptibility(distances, times)

    peak = int(np.argmax(chi))
    assert abs(0.5 * (times[peak] + times[peak + 1])) < 0.01
    assert abs(chi[peak] / (sweep / (4.0 * coupling)) ** 2 - 1.0) < 0.02


def test_davis_kahan_ratio_is_small_for_strong_modules_and_large_for_weak_ones():
    from nds.analysis import analyse_layer
    from nds.moments import BlockedNeuronMoments
    from nds.synthetic import modular_activations

    ratios = []
    for strength in [0.6, 0.06]:
        generator = np.random.default_rng(12)
        activations, _ = modular_activations(4000, (24,) * 6 + (1,) * 40, strength, generator)
        blocked = BlockedNeuronMoments(activations.shape[1], number_of_blocks=8)
        for block in range(8):
            blocked.add_batch(activations[block * 500:(block + 1) * 500], block)
        result = analyse_layer(
            blocked,
            settings={"minimum_firing_rate": 0.0, "number_of_rotations": 0, "number_of_bootstrap_replicates": 4},
        )
        ratios.append(result["spectrum"]["davis_kahan_ratio"])

    assert ratios[0] < 0.3
    assert ratios[1] > 2.0 * ratios[0]


def test_leading_eigenvectors_match_a_full_decomposition():
    from nds.spectra import chordal_subspace_distance
    from nds.spectra import leading_eigenvectors

    generator = np.random.default_rng(3)
    matrix = generator.standard_normal((80, 80))
    symmetric = matrix + matrix.T

    _, vectors = np.linalg.eigh(symmetric)
    expected = vectors[:, ::-1][:, :5]

    found = leading_eigenvectors(symmetric, 5)

    assert chordal_subspace_distance(expected, found) < 1e-8


def test_subspace_change_signal_to_noise_is_about_one_on_noise_and_large_on_a_jump():
    from nds.timeline import subspace_change_signal_to_noise

    noises = np.array([0.01, 0.01, 0.01, 0.01])
    distances = np.array([0.02, 0.30, 0.02])

    ratio = subspace_change_signal_to_noise(distances, noises)

    assert abs(ratio[0] - 1.0) < 1e-12
    assert ratio[1] > 10
    assert np.isnan(subspace_change_signal_to_noise(np.array([0.1]), np.array([np.nan, 0.01])))[0]


def test_paired_bootstrap_of_the_subspace_change_sees_a_one_neuron_move():
    from nds.analysis import paired_subspace_change
    from nds.moments import BlockedNeuronMoments

    generator = np.random.default_rng(11)
    neurons = 96
    tokens = 2400
    labels = np.repeat(np.arange(4), 24)
    moved = labels.copy()
    moved[0] = 1

    latents = generator.standard_normal((tokens, 4))
    private = generator.standard_normal((tokens, neurons))

    def blocked(current_labels):
        values = np.sqrt(0.7) * private + np.sqrt(0.3) * latents[:, current_labels]
        moments = BlockedNeuronMoments(neurons, number_of_blocks=8)
        for block in range(8):
            moments.add_batch(values[block * 300:(block + 1) * 300], block)
        return moments

    settings = {"minimum_firing_rate": 0.0}

    quiet = paired_subspace_change(blocked(labels), blocked(labels), settings, number_of_replicates=6)
    event = paired_subspace_change(blocked(labels), blocked(moved), settings, number_of_replicates=6)

    # same tokens, same network: no change at all, in every replicate
    assert quiet["distance"] < 1e-10
    assert max(quiet["replicates"]) < 1e-10

    # one neuron of 96 changes module: resolved many paired standard deviations above zero
    assert event["distance"] > 1e-3
    assert event["distance"] / event["replicate_sd"] > 5


def test_crossvalidated_distance_matches_projector_formula_and_removes_bias():
    from nds.spectra import crossvalidated_projector_distance

    generator = np.random.default_rng(2)
    bases = [np.linalg.qr(generator.standard_normal((40, 3)))[0] for _ in range(4)]
    d_cv, per_neuron = crossvalidated_projector_distance(*bases)

    projectors = [basis @ basis.T for basis in bases]
    expected = np.trace((projectors[0] - projectors[1]) @ (projectors[2] - projectors[3])) / 6.0
    assert abs(d_cv - expected) < 1e-10
    assert abs(per_neuron.sum() - d_cv) < 1e-10

    # same true subspace, independent noise in each of the four estimates: d_cv near 0, plain d not
    truth = np.linalg.qr(generator.standard_normal((200, 4)))[0]
    values = []
    for _ in range(40):
        noisy = [truth + 0.05 * generator.standard_normal(truth.shape) for _ in range(4)]
        values.append(crossvalidated_projector_distance(*noisy)[0])
    plain = np.sum((noisy[0] @ np.linalg.pinv(noisy[0]) - noisy[1] @ np.linalg.pinv(noisy[1])) ** 2) / 8.0
    assert abs(np.mean(values)) < 0.2 * plain


def test_crossvalidated_subspace_change_finds_the_moved_neuron():
    from nds.analysis import crossvalidated_subspace_change
    from nds.moments import BlockedNeuronMoments

    generator = np.random.default_rng(13)
    neurons = 96
    labels = np.repeat(np.arange(4), 24)
    moved = labels.copy()
    moved[5] = 2
    latents = generator.standard_normal((2400, 4))
    private = generator.standard_normal((2400, neurons))

    def blocked(current):
        values = np.sqrt(0.7) * private + np.sqrt(0.3) * latents[:, current]
        moments = BlockedNeuronMoments(neurons, number_of_blocks=8)
        for block in range(8):
            moments.add_batch(values[block * 300:(block + 1) * 300], block)
        return moments

    settings = {"minimum_firing_rate": 0.0}
    quiet = crossvalidated_subspace_change(blocked(labels), blocked(labels), settings)
    event = crossvalidated_subspace_change(blocked(labels), blocked(moved), settings)

    assert abs(quiet["d_cv"]) < 1e-10
    assert event["d_cv"] > 1e-3
    assert int(event["neuron_ids"][np.argmax(event["per_neuron"])]) == 5
