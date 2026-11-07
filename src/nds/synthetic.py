"""
synthetic.py
============

Fake "neuron activations" with structure we choose, so every part of the pipeline can be tested
on data where we know the right answer before it ever sees a real model.

The recipe: a few hidden "latent" signals per token; each module of neurons listens to one latent
plus private noise. Neurons in the same module are correlated, neurons in different modules are
not. `module_strength` (0..1) sets how much of each neuron's variance comes from its module.

These are the same toys as the deep-dive probes (probes/probe_neuron_graph.py, T1 and T3).
"""

import numpy as np


def modular_activations(
    number_of_tokens=2000,
    module_sizes=(32, 32, 32, 32),
    module_strength=0.7,
    random_generator=None,
):
    """
    Activations (tokens x neurons) with one module per entry of `module_sizes`.

    Returns (activations, module_labels).
    """
    if random_generator is None:
        random_generator = np.random.default_rng(0)

    number_of_modules = len(module_sizes)
    number_of_neurons = int(np.sum(module_sizes))

    latents = random_generator.standard_normal((number_of_tokens, number_of_modules))

    activations = np.zeros((number_of_tokens, number_of_neurons))
    module_labels = np.zeros(number_of_neurons, dtype=int)

    signal_weight = np.sqrt(module_strength)
    noise_weight = np.sqrt(1.0 - module_strength)

    neuron = 0
    for module, size in enumerate(module_sizes):
        for _ in range(size):
            loading = 0.7 + 0.6 * random_generator.random()

            noise = random_generator.standard_normal(number_of_tokens)

            activations[:, neuron] = signal_weight * loading * latents[:, module] + noise_weight * noise
            module_labels[neuron] = module

            neuron = neuron + 1

    return activations, module_labels


def three_module_path(
    position,
    number_of_tokens=3000,
    module_size=40,
    direct_coupling=0.1,
    random_generator=None,
):
    """
    The avoided-crossing toy (deep-dive probe T3). Three modules A, B, C.

    As `position` goes from 0 to 1 ("training time"), the correlation between A and B falls while
    the correlation between B and C rises. The two slowest diffusion modes (A-vs-rest and
    C-vs-rest) approach each other and, because of `direct_coupling` between A and C, avoid
    crossing: the slowest mode moves from being C-localised to A-localised.

    The same random seed is used for every position so the only thing that changes is `position`.

    Returns (activations, module_labels) with labels 0 = A, 1 = B, 2 = C.
    """
    if random_generator is None:
        random_generator = np.random.default_rng(1)

    latent_a = random_generator.standard_normal(number_of_tokens)
    latent_b = random_generator.standard_normal(number_of_tokens)
    latent_c = random_generator.standard_normal(number_of_tokens)

    shared_ab = random_generator.standard_normal(number_of_tokens)
    shared_bc = random_generator.standard_normal(number_of_tokens)
    shared_ac = random_generator.standard_normal(number_of_tokens)

    weight_ab = 0.6 * (1.0 - position)
    weight_bc = 0.6 * position
    weight_ac = direct_coupling

    signal_a = latent_a + weight_ab * shared_ab + weight_ac * shared_ac
    signal_b = latent_b + weight_ab * shared_ab + weight_bc * shared_bc
    signal_c = latent_c + weight_bc * shared_bc + weight_ac * shared_ac

    signals = [signal_a, signal_b, signal_c]

    columns = []
    labels = []

    for module, signal in enumerate(signals):
        for _ in range(module_size):
            noise = random_generator.standard_normal(number_of_tokens)
            columns.append(signal + 0.8 * noise)
            labels.append(module)

    activations = np.column_stack(columns)
    module_labels = np.array(labels)

    return activations, module_labels
