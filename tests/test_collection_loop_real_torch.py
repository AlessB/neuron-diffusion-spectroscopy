"""
The collection loop (nds.activations.collect_neuron_moments) with real PyTorch, on a tiny
GPT-NeoX-shaped model with random weights (no download). Runs on the CPU, and on the GPU too when
one is visible. Every statistic is compared with the same quantity computed directly from the full
activations.

Why: test_collection_loop_fake_torch checks the bookkeeping with a numpy stand-in and is skipped
where real torch is installed. Since 8 Nov 2026 the per-block merge happens on the device in
float64 (merge_batch_into_device_totals), so it needs a test that runs with real tensors.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nds.activations import collect_neuron_moments

NEURONS = 40
WIDTH = 12
VOCABULARY = 30
NUMBER_OF_LAYERS = 2


class TinyMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense_h_to_4h = torch.nn.Linear(WIDTH, NEURONS)
        self.act = torch.nn.GELU()
        self.dense_4h_to_h = torch.nn.Linear(NEURONS, WIDTH)

    def forward(self, hidden):
        return self.dense_4h_to_h(self.act(self.dense_h_to_4h(hidden)))


class TinyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = TinyMLP()

    def forward(self, hidden):
        return hidden + self.mlp(hidden)


class TinyOutput:
    def __init__(self, logits):
        self.logits = logits


class TinyNeoX(torch.nn.Module):
    """
    The attribute names collect_neuron_moments uses: gpt_neox.layers[l].mlp.act,
    config.intermediate_size, and a call that returns an object with .logits.
    """

    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(VOCABULARY, WIDTH)
        self.gpt_neox = torch.nn.Module()
        self.gpt_neox.layers = torch.nn.ModuleList()
        for _ in range(NUMBER_OF_LAYERS):
            self.gpt_neox.layers.append(TinyLayer())
        self.unembed = torch.nn.Linear(WIDTH, VOCABULARY)
        self.config = type("Config", (), {"intermediate_size": NEURONS})()

    def forward(self, input_ids):
        hidden = self.embed(input_ids)
        for layer in self.gpt_neox.layers:
            hidden = layer(hidden)
        logits = self.unembed(hidden)
        return TinyOutput(logits)


def direct_activations(model, ids):
    """
    Post-GELU activations of every layer for all sequences at once, as float64 numpy arrays.
    """
    captured = {}
    handles = []
    for layer_index in range(NUMBER_OF_LAYERS):
        def hook(module, inputs, output, layer_index=layer_index):
            captured[layer_index] = output.detach().double().cpu().numpy()
        handle = model.gpt_neox.layers[layer_index].mlp.act.register_forward_hook(hook)
        handles.append(handle)
    device = next(model.parameters()).device
    with torch.no_grad():
        model(torch.as_tensor(ids, dtype=torch.long, device=device))
    for handle in handles:
        handle.remove()
    return captured


def available_devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


@pytest.mark.parametrize("device", available_devices())
def test_device_merge_matches_direct_computation(device):
    torch.manual_seed(0)
    model = TinyNeoX().to(device)
    model.eval()

    random_generator = np.random.default_rng(1)
    number_of_sequences = 23
    sequence_length = 17
    skip_first = 1
    number_of_blocks = 4
    ids = random_generator.integers(1, VOCABULARY, size=(number_of_sequences, sequence_length))
    ids[:, 0] = 0

    result = collect_neuron_moments(
        model,
        ids,
        batch_size=5,
        number_of_blocks=number_of_blocks,
        skip_first_positions=skip_first,
        compute_prompt_entropy=False,
    )

    direct = direct_activations(model, ids)

    for layer in range(NUMBER_OF_LAYERS):
        kept = direct[layer][:, skip_first:, :]
        flat = kept.reshape(-1, NEURONS)

        blocked = result["moments"][layer]
        merged = blocked.merged()

        assert merged.count == flat.shape[0]
        assert np.allclose(merged.mean, flat.mean(axis=0), rtol=1e-5, atol=1e-6)
        assert np.allclose(merged.covariance(), np.cov(flat, rowvar=False), rtol=1e-4, atol=1e-6)
        assert np.allclose(merged.max_abs_activation, np.abs(flat).max(axis=0), rtol=1e-5, atol=1e-6)
        assert np.array_equal(merged.active_count, (flat > 0.0).sum(axis=0))

        # sequence i goes to block i % number_of_blocks, and each block's own statistics are right
        for block_index in range(number_of_blocks):
            block_rows = kept[block_index::number_of_blocks].reshape(-1, NEURONS)
            block = blocked.blocks[block_index]
            assert block.count == block_rows.shape[0]
            assert np.allclose(block.mean, block_rows.mean(axis=0), rtol=1e-5, atol=1e-6)
            assert np.allclose(block.covariance(), np.cov(block_rows, rowvar=False), rtol=1e-4, atol=1e-6)
