"""
neuron_attribution.py
=====================

Part 2, the dictionary-free route: which MLP neurons matter for subject-verb agreement (SVA) at a
given checkpoint, measured directly on the same 2,048 post-GELU neurons as our graph. Needs PyTorch.

Attribution patching (Marks et al. 2025, eq. 3) for neuron j of layer l:

    IE(l, j) = sum over positions t of  dm/da_lj(t) at the clean input  *  ( a_lj(t)_patch - a_lj(t)_clean )

m = logit(patch answer) - logit(clean answer) at the last position. One clean forward + backward
and one patch forward per batch give the effect of every neuron in every layer at once.

For early layers the linear approximation can be poor (why Marks et al. use integrated gradients);
`neuron_integrated_gradients` interpolates the input embeddings in `steps` stages (IG-inputs, as
in EAP-IG and ADAG) and is about `steps` times more expensive.

Run it at every checkpoint, in the same loop as activations.collect_neuron_moments, so each
checkpoint is downloaded once; the circuit at each step is then a set of neuron ids in the same
numbering as our graph's `live_neuron_ids`. Arora et al. (2026) report SVA neuron circuits about
100x smaller than MLP-output circuits at equal faithfulness.

Credit
------
- S. Marks, C. Rager, E. J. Michaud, Y. Belinkov, D. Bau, A. Mueller (2025). "Sparse feature
  circuits." ICLR 2025. SVA data and the metric: https://github.com/saprmarks/feature-circuits (MIT).
- A. Syed, C. Rager, A. Conmy (2024). "Attribution patching outperforms automated circuit
  discovery." BlackboxNLP 2024.
- M. Hanna, S. Pezzelle, Y. Belinkov (2024). EAP-IG, https://github.com/hannamw/EAP-IG (MIT).
- A. Arora, Z. Wu, J. Steinhardt, S. Schwettmann (2026). "Language model circuits are sparse in
  the neuron basis." arXiv:2601.22594. (re-implemented, not copied: their repo has no license)
- Sketch written in this project's code study (research/S3_code_circuits.md, 17 Oct 2026).
"""


def _register_activation_hooks(model, captured):
    """
    Forward hooks on every layer's mlp.act that store the post-GELU output in `captured`.
    """
    layers = model.gpt_neox.layers

    handles = []

    for layer_index in range(len(layers)):

        def make_hook(index):
            def hook(module, inputs, output):
                captured[index] = output
            return hook

        handle = layers[layer_index].mlp.act.register_forward_hook(make_hook(layer_index))
        handles.append(handle)

    return handles


def answer_logit_difference(logits, clean_answer_ids, patch_answer_ids):
    """
    m = logit(patch answer) - logit(clean answer) at the last position, one value per example.
    """
    import torch

    last_logits = logits[:, -1, :]

    rows = torch.arange(last_logits.shape[0], device=last_logits.device)

    return last_logits[rows, patch_answer_ids] - last_logits[rows, clean_answer_ids]


def neuron_attribution_patching(
    model,
    clean_token_ids,
    patch_token_ids,
    clean_answer_ids,
    patch_answer_ids,
):
    """
    IE for every (layer, neuron), averaged over the batch. Returns a tensor (layers, d_mlp).

    clean_token_ids, patch_token_ids: (batch, positions), same length (Marks et al. drop pairs whose
    prefixes differ in length). Answer ids: (batch,).
    """
    import torch

    captured = {}
    handles = _register_activation_hooks(model, captured)

    try:
        with torch.no_grad():
            model(patch_token_ids)
        patch_activations = dict(captured)

        output = model(clean_token_ids)

        metric = answer_logit_difference(output.logits, clean_answer_ids, patch_answer_ids)

        number_of_layers = len(model.gpt_neox.layers)

        clean_activations = []
        for layer_index in range(number_of_layers):
            clean_activations.append(captured[layer_index])

        gradients = torch.autograd.grad(metric.sum(), clean_activations)

    finally:
        for handle in handles:
            handle.remove()

    effects = []
    for layer_index in range(number_of_layers):
        difference = patch_activations[layer_index] - clean_activations[layer_index].detach()

        per_token_effect = gradients[layer_index] * difference

        summed_over_positions = per_token_effect.sum(dim=1)

        effects.append(summed_over_positions.mean(dim=0))

    return torch.stack(effects)


def neuron_integrated_gradients(
    model,
    clean_token_ids,
    patch_token_ids,
    clean_answer_ids,
    patch_answer_ids,
    steps=10,
):
    """
    IG-inputs: interpolate the input embeddings from clean to patch in `steps` stages, average the
    gradients at every mlp.act, multiply by (a_patch - a_clean). Returns (layers, d_mlp).
    """
    import torch

    embedding_layer = model.get_input_embeddings()

    with torch.no_grad():
        clean_embeddings = embedding_layer(clean_token_ids)
        patch_embeddings = embedding_layer(patch_token_ids)

    captured = {}
    handles = _register_activation_hooks(model, captured)

    number_of_layers = len(model.gpt_neox.layers)

    try:
        with torch.no_grad():
            model(patch_token_ids)
            patch_activations = dict(captured)

            model(clean_token_ids)
            clean_activations = dict(captured)

        summed_gradients = None

        for step in range(steps):
            alpha = step / steps

            mixed_embeddings = (1.0 - alpha) * clean_embeddings + alpha * patch_embeddings

            output = model(inputs_embeds=mixed_embeddings)

            metric = answer_logit_difference(output.logits, clean_answer_ids, patch_answer_ids)

            activations = []
            for layer_index in range(number_of_layers):
                activations.append(captured[layer_index])

            gradients = torch.autograd.grad(metric.sum(), activations)

            if summed_gradients is None:
                summed_gradients = list(gradients)
            else:
                for layer_index in range(number_of_layers):
                    summed_gradients[layer_index] = summed_gradients[layer_index] + gradients[layer_index]

    finally:
        for handle in handles:
            handle.remove()

    effects = []
    for layer_index in range(number_of_layers):
        mean_gradient = summed_gradients[layer_index] / steps

        difference = patch_activations[layer_index] - clean_activations[layer_index]

        per_token_effect = mean_gradient * difference

        effects.append(per_token_effect.sum(dim=1).mean(dim=0))

    return torch.stack(effects)
