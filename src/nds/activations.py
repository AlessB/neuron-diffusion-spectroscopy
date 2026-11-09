"""
activations.py
==============

Run one Pythia checkpoint over the probe text and keep running statistics of every MLP neuron.
This is the only file that needs PyTorch and the transformers library.

Where the neurons are
---------------------
In the Hugging Face GPT-NeoX code (transformers/models/gpt_neox/modeling_gpt_neox.py) every layer
has an MLP:

    class GPTNeoXMLP(nn.Module):
        def forward(self, hidden_states):
            hidden_states = self.dense_h_to_4h(hidden_states)   # 512 -> 2048 for Pythia-70M
            hidden_states = self.act(hidden_states)             # GELU  <- the neurons
            hidden_states = self.dense_4h_to_h(hidden_states)   # 2048 -> 512
            return hidden_states

`self.act` is its own module in every layer, so a forward hook on
model.gpt_neox.layers[l].mlp.act hands us the post-GELU activations of layer l:
a tensor of shape (batch, tokens, d_mlp). That is the same quantity TransformerLens calls
blocks.{l}.mlp.hook_post. We hook transformers directly (as Li et al. 2025 do in their code) so the
project does not depend on TransformerLens' version.

What happens to the activations
-------------------------------
They are never stored. For each batch, layer and bootstrap block we compute, on the GPU:
    token count, mean, centred scatter matrix (m x m), active count, largest |activation|
and merge them, still on the GPU and in float64, into a running total per layer and block (the
same Chan-Golub-LeVeque merge as moments.py). Only at the end of the probe text do those totals go
to the CPU, once, into a BlockedNeuronMoments. One batch of 16 sequences x 256 tokens x 2048
neurons in float32 is 33 MB; one scatter matrix is 16 MB in float32 and 32 MB in float64. The
running totals take layers x blocks x 32 MB of GPU memory (6 x 8 x 32 MB = 1.6 GB for Pythia-70M).

Until 8 Nov 2026 every batch's scatter matrices were copied to the CPU and merged there: 3,072
copies of 32 MB per checkpoint, which took about 1,000 of the 2,300 seconds of the first smoke test
on the Aero. Same formula and same order of merging; only where the merge happens changed, so
the numbers agree up to rounding in the last digits.

GPU notes for AB's machine (Pascal GTX 1060): FP32 only (DECISIONS / AGENTS.md), one run per
card with CUDA_VISIBLE_DEVICES.

Credit
------
- Hugging Face transformers, GPT-NeoX implementation: https://github.com/huggingface/transformers
- S. Black, S. Biderman, E. Hallahan, et al. (2022). "GPT-NeoX-20B: An open-source autoregressive
  language model." (the architecture Pythia uses)
- Li et al. (2025) code loads Pythia checkpoints the same way:
  GPTNeoXForCausalLM.from_pretrained("EleutherAI/<model>", revision=f"step{step}").
- O. Skean et al. (2025), prompt entropy: see baselines.py for the recipe and its credit.
"""

import time

import numpy as np

from nds.baselines import prompt_entropy_from_gram
from nds.moments import BlockedNeuronMoments


def load_pythia(
    model_name,
    step,
    device,
    dtype_name="float32",
    cache_dir=None,
    attention_implementation="eager",
):
    """
    Load one Pythia checkpoint and its tokenizer.

    model_name                 e.g. "pythia-70m-deduped" (the "EleutherAI/" prefix is added here)
    step                       training step, e.g. 1000 (loaded from git branch "step1000")
    attention_implementation   "eager" so attention patterns can be read (induction.py);
                               "sdpa" is faster when attention patterns are not needed
    """
    import torch
    from transformers import AutoTokenizer
    from transformers import GPTNeoXForCausalLM

    repository = f"EleutherAI/{model_name}"
    revision = f"step{int(step)}"

    dtype = getattr(torch, dtype_name)

    model = GPTNeoXForCausalLM.from_pretrained(
        repository,
        revision=revision,
        torch_dtype=dtype,
        cache_dir=cache_dir,
        attn_implementation=attention_implementation,
    )

    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        repository,
        revision=revision,
        cache_dir=cache_dir,
    )

    return model, tokenizer


def collect_neuron_moments(
    model,
    token_ids,
    batch_size=16,
    number_of_blocks=8,
    layers=None,
    active_threshold=0.0,
    skip_first_positions=0,
    compute_prompt_entropy=True,
    token_positions=None,
):
    """
    Run the probe text through the model and return one BlockedNeuronMoments per layer, plus the
    mean next-token loss on the probe text.

    token_ids               integer array (number_of_sequences, sequence_length). Sequence i goes
                            to block i % number_of_blocks, so blocks are interleaved.
    skip_first_positions    positions 0 .. skip-1 of every sequence still go through the model
                            (they are context) but are left out of the neuron statistics.
                            1 = leave out the <|endoftext|> position (proposed D17, probe_text.py).
    compute_prompt_entropy  also compute Skean et al.'s prompt entropy of every sequence and layer
                            (baselines.prompt_entropy); one 255 x 255 eigen-decomposition per
                            sequence and layer, about a second per checkpoint for 70M.
    token_positions         (sequence_index, position) arrays from token_sample.choose_token_positions:
                            the fixed token sample kept at every checkpoint (D31). None skips it.

    Also measured: every neuron's lag-one correlation between neighbouring tokens, whose median sets
    the effective number of tokens for the null (noise.py, D36).
    """
    import torch
    import torch.nn.functional as functional

    device = next(model.parameters()).device

    all_layers = model.gpt_neox.layers

    if layers is None:
        layers = list(range(len(all_layers)))

    number_of_neurons = model.config.intermediate_size

    moments_per_layer = {}
    for layer in layers:
        moments_per_layer[layer] = BlockedNeuronMoments(
            number_of_neurons=number_of_neurons,
            number_of_blocks=number_of_blocks,
        )
        for block in moments_per_layer[layer].blocks:
            block.active_threshold = active_threshold

    # the hook writes the current batch's activations here, keyed by layer
    captured = {}

    def make_hook(layer):
        def hook(module, inputs, output):
            captured[layer] = output.detach()
        return hook

    handles = []
    for layer in layers:
        handle = all_layers[layer].mlp.act.register_forward_hook(make_hook(layer))
        handles.append(handle)

    token_ids = np.asarray(token_ids)
    number_of_sequences = token_ids.shape[0]

    total_loss = 0.0
    total_predicted_tokens = 0

    # prompt entropy (raw, nats) of every sequence, per layer
    prompt_entropies = {}
    for layer in layers:
        prompt_entropies[layer] = []

    # lag-one sums per layer: sum of x_t * x_(t+1), sum of x_t, sum of x_t^2 over neighbouring
    # positions (float64 on the CPU; one vector of neurons each)
    lag_sums = {}
    for layer in layers:
        lag_sums[layer] = {
            "pairs": 0,
            "product": np.zeros(number_of_neurons),
            "first": np.zeros(number_of_neurons),
            "second": np.zeros(number_of_neurons),
            "first_squared": np.zeros(number_of_neurons),
            "second_squared": np.zeros(number_of_neurons),
        }

    # sum of |x| per neuron (Kim et al.'s knowledge entropy, baselines.knowledge_entropy), and the
    # higher moments and sparsity of Gurnee et al.'s neuron-family statistics (sums of x^3, x^4 and
    # the count of x > 0); all over the same tokens as the covariance (position 0 left out)
    absolute_sums = {}
    absolute_counts = {}
    cube_sums = {}
    fourth_power_sums = {}
    positive_counts = {}
    for layer in layers:
        absolute_sums[layer] = np.zeros(number_of_neurons)
        absolute_counts[layer] = 0
        cube_sums[layer] = np.zeros(number_of_neurons)
        fourth_power_sums[layer] = np.zeros(number_of_neurons)
        positive_counts[layer] = np.zeros(number_of_neurons)

    # position 0 on its own (D30 leaves it out of every statistic; research2/R4 pitfall P2): the
    # mean and max of |x| there, so the attention-sink spike is recorded rather than lost
    position_zero_absolute_sums = {}
    position_zero_max = {}
    position_zero_rows = {}
    for layer in layers:
        position_zero_absolute_sums[layer] = np.zeros(number_of_neurons)
        position_zero_max[layer] = np.zeros(number_of_neurons)
        position_zero_rows[layer] = 0

    # the token sample: which (row in the probe set, position) to keep, grouped by row
    sample_rows = {}
    sample_storage = {}
    if token_positions is not None:
        sequence_index, position = token_positions
        for order, (row, column) in enumerate(zip(np.asarray(sequence_index), np.asarray(position))):
            sample_rows.setdefault(int(row), []).append((order, int(column)))
        for layer in layers:
            sample_storage[layer] = np.zeros((len(sequence_index), number_of_neurons), dtype=np.float32)

    # running totals per layer and block, kept on the device (see the module docstring)
    device_totals = {}
    for layer in layers:
        device_totals[layer] = {}
        for block_index in range(number_of_blocks):
            device_totals[layer][block_index] = empty_device_totals()

    start_time = time.time()

    try:
        with torch.no_grad():
            for start in range(0, number_of_sequences, batch_size):
                stop = min(start + batch_size, number_of_sequences)

                batch_ids = torch.as_tensor(token_ids[start:stop], dtype=torch.long, device=device)

                output = model(batch_ids)

                # mean next-token loss on the probe text (a basic "how trained is it" number)
                logits = output.logits[:, :-1, :].float()
                targets = batch_ids[:, 1:]

                loss_sum = functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                    reduction="sum",
                )

                total_loss = total_loss + float(loss_sum)
                total_predicted_tokens = total_predicted_tokens + int(targets.numel())

                # summarise each layer's activations per block, on the device
                sequence_indices = np.arange(start, stop)
                block_of_sequence = sequence_indices % number_of_blocks

                for layer in layers:
                    activations = captured[layer].float()

                    # the fixed token sample (positions are in the full row, before skipping)
                    for row_in_batch, probe_row in enumerate(range(start, stop)):
                        for order, column in sample_rows.get(probe_row, []):
                            sample_storage[layer][order] = activations[row_in_batch, column].cpu().numpy()

                    # position 0 recorded separately before it is left out
                    first_position = activations[:, 0, :].abs().double()
                    position_zero_absolute_sums[layer] += first_position.sum(dim=0).cpu().numpy()
                    batch_maximum = first_position.max(dim=0).values.cpu().numpy()
                    position_zero_max[layer] = np.maximum(position_zero_max[layer], batch_maximum)
                    position_zero_rows[layer] += int(first_position.shape[0])

                    # leave the separator position(s) out of every statistic
                    activations = activations[:, skip_first_positions:, :]

                    # neighbouring-token sums for the lag-one correlation
                    earlier = activations[:, :-1, :].reshape(-1, number_of_neurons).double()
                    later = activations[:, 1:, :].reshape(-1, number_of_neurons).double()
                    sums = lag_sums[layer]
                    sums["pairs"] += int(earlier.shape[0])
                    sums["product"] += (earlier * later).sum(dim=0).cpu().numpy()
                    sums["first"] += earlier.sum(dim=0).cpu().numpy()
                    sums["second"] += later.sum(dim=0).cpu().numpy()
                    sums["first_squared"] += (earlier * earlier).sum(dim=0).cpu().numpy()
                    sums["second_squared"] += (later * later).sum(dim=0).cpu().numpy()

                    flat_activations = activations.reshape(-1, number_of_neurons)
                    absolute_sums[layer] += flat_activations.abs().double().sum(dim=0).cpu().numpy()
                    absolute_counts[layer] += int(flat_activations.shape[0])

                    flat_double = flat_activations.double()
                    squared = flat_double * flat_double
                    cube_sums[layer] += (squared * flat_double).sum(dim=0).cpu().numpy()
                    fourth_power_sums[layer] += (squared * squared).sum(dim=0).cpu().numpy()
                    positive_counts[layer] += (flat_activations > 0).sum(dim=0).double().cpu().numpy()

                    if compute_prompt_entropy:
                        for row in range(activations.shape[0]):
                            gram = token_gram_on_device(activations[row])
                            entropy = prompt_entropy_from_gram(gram)
                            prompt_entropies[layer].append(entropy)

                    for block_index in np.unique(block_of_sequence):
                        rows_in_batch = np.flatnonzero(block_of_sequence == block_index)

                        block_activations = activations[rows_in_batch]
                        block_activations = block_activations.reshape(-1, number_of_neurons)

                        merge_batch_into_device_totals(
                            device_totals[layer][int(block_index)],
                            block_activations,
                            active_threshold,
                        )

                captured.clear()

    finally:
        for handle in handles:
            handle.remove()

    # the running totals go to the CPU once per layer and block
    for layer in layers:
        for block_index in range(number_of_blocks):
            totals = device_totals[layer][block_index]
            if totals["count"] == 0:
                continue
            moments_per_layer[layer].add_batch_summary(
                block_index,
                totals["count"],
                totals["mean"].cpu().numpy(),
                totals["scatter"].cpu().numpy(),
                totals["active_count"].cpu().numpy(),
                totals["max_abs"].cpu().numpy(),
            )
    device_totals.clear()

    elapsed_seconds = time.time() - start_time

    prompt_entropy_per_layer = {}

    if compute_prompt_entropy:
        tokens_per_prompt = token_ids.shape[1] - skip_first_positions

        largest_possible = min(np.log(tokens_per_prompt), np.log(number_of_neurons))

        for layer in layers:
            values = np.array(prompt_entropies[layer], dtype=np.float64)

            mean_value = float(np.nanmean(values))

            prompt_entropy_per_layer[layer] = {
                "mean_raw": mean_value,
                "mean_normalised": mean_value / largest_possible,
                "standard_error": float(np.nanstd(values) / np.sqrt(max(len(values), 1))),
            }

    # lag-one correlation per neuron, then its median per layer
    lag_one = {}
    for layer in layers:
        lag_one[layer] = median_lag_one_correlation(lag_sums[layer])

    mean_absolute_activation = {}
    raw_moments = {}
    position_zero = {}
    for layer in layers:
        count = max(absolute_counts[layer], 1)

        mean_absolute_activation[layer] = absolute_sums[layer] / count

        raw_moments[layer] = {
            "count": absolute_counts[layer],
            "mean_cube": cube_sums[layer] / count,
            "mean_fourth_power": fourth_power_sums[layer] / count,
            "positive_fraction": positive_counts[layer] / count,
        }

        position_zero[layer] = {
            "mean_absolute": position_zero_absolute_sums[layer] / max(position_zero_rows[layer], 1),
            "max_absolute": position_zero_max[layer],
        }

    result = {
        "moments": moments_per_layer,
        "mean_absolute_activation": mean_absolute_activation,
        "raw_moments": raw_moments,
        "position_zero": position_zero,
        "prompt_entropy": prompt_entropy_per_layer,
        "lag_one_correlation": lag_one,
        "token_sample": sample_storage,
        "probe_loss": total_loss / max(total_predicted_tokens, 1),
        "elapsed_seconds": elapsed_seconds,
    }

    return result


def median_lag_one_correlation(sums):
    """
    Median over neurons of corr(x_t, x_(t+1)) from the accumulated sums (dead neurons skipped).
    """
    pairs = max(sums["pairs"], 1)

    mean_first = sums["first"] / pairs
    mean_second = sums["second"] / pairs

    covariance = sums["product"] / pairs - mean_first * mean_second

    variance_first = sums["first_squared"] / pairs - mean_first ** 2
    variance_second = sums["second_squared"] / pairs - mean_second ** 2

    scale = np.sqrt(np.maximum(variance_first * variance_second, 0.0))

    alive = scale > 1e-12

    if not np.any(alive):
        return float("nan")

    correlations = covariance[alive] / scale[alive]

    return float(np.median(correlations))


def token_gram_on_device(token_activations):
    """
    Skean et al.'s normalisation for ONE prompt, done where the tensor lives: centre the tokens,
    scale each token row to unit length, return the token Gram matrix (tokens x tokens) as a
    float64 numpy array. Same maths as baselines.normalised_token_rows, which the tests check.
    """
    centred = token_activations - token_activations.mean(dim=0)

    lengths = centred.norm(dim=1, keepdim=True)

    lengths = lengths.clamp_min(1e-12)

    rows = centred / lengths

    gram = rows @ rows.T

    return gram.double().cpu().numpy()


def empty_device_totals():
    """
    Running totals of one layer and one bootstrap block, before any batch has been added.
    The tensors are created by the first batch, on that batch's device.
    """
    totals = {
        "count": 0,
        "mean": None,
        "scatter": None,
        "active_count": None,
        "max_abs": None,
    }
    return totals


def merge_batch_into_device_totals(totals, activations, active_threshold):
    """
    Add one batch of activations (tokens, neurons), a tensor on the GPU or CPU, to the running
    totals of its layer and block, without leaving the device.

    The batch is centred on its own mean and its scatter computed in float32 (accurate enough once
    centred, method notes section 13); everything that is accumulated is float64. The merge is
    the Chan-Golub-LeVeque update, line for line the same as NeuronMoments._merge in moments.py:
        delta   = batch_mean - mean
        mean    = mean + delta * batch_count / total_count
        scatter = scatter + batch_scatter + outer(delta, delta) * count * batch_count / total_count
    """
    import torch

    batch_count = int(activations.shape[0])

    if batch_count == 0:
        return

    batch_mean = activations.mean(dim=0)
    batch_centred = activations - batch_mean
    batch_scatter = batch_centred.T @ batch_centred

    batch_mean_double = batch_mean.double()
    batch_scatter_double = batch_scatter.double()
    batch_active_count = (activations > active_threshold).sum(dim=0)
    batch_max_abs = activations.abs().max(dim=0).values.double()

    if totals["count"] == 0:
        totals["count"] = batch_count
        totals["mean"] = batch_mean_double
        totals["scatter"] = batch_scatter_double
        totals["active_count"] = batch_active_count
        totals["max_abs"] = batch_max_abs
        return

    previous_count = totals["count"]
    total_count = previous_count + batch_count

    delta = batch_mean_double - totals["mean"]

    new_mean = totals["mean"] + delta * (batch_count / total_count)

    cross_weight = previous_count * batch_count / total_count
    cross_term = torch.outer(delta, delta) * cross_weight
    new_scatter = totals["scatter"] + batch_scatter_double + cross_term

    totals["count"] = total_count
    totals["mean"] = new_mean
    totals["scatter"] = new_scatter
    totals["active_count"] = totals["active_count"] + batch_active_count
    totals["max_abs"] = torch.maximum(totals["max_abs"], batch_max_abs)


def summarise_on_device(activations, active_threshold):
    """
    Count, mean, centred scatter, active count and max |x| for a (tokens, neurons) tensor,
    computed where the tensor lives (GPU or CPU) and returned as numpy arrays.

    Centring on the batch mean before multiplying keeps float32 accurate enough; the merge across
    batches happens in float64 on the CPU (moments.py). No longer used by collect_neuron_moments
    (which merges on the device, merge_batch_into_device_totals); kept for notebooks and tests.
    """
    count = activations.shape[0]

    mean = activations.mean(dim=0)

    centred = activations - mean

    scatter = centred.T @ centred

    active_count = (activations > active_threshold).sum(dim=0)

    max_abs = activations.abs().max(dim=0).values

    summary = {
        "count": count,
        "mean": mean.double().cpu().numpy(),
        "scatter": scatter.double().cpu().numpy(),
        "active_count": active_count.cpu().numpy(),
        "max_abs": max_abs.double().cpu().numpy(),
    }

    return summary
