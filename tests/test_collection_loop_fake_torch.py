"""
The collection loop (nds.activations.collect_neuron_moments) on a fake GPT-NeoX-shaped model,
through a small numpy stand-in for the parts of torch it uses (tests/fake_torch). Every statistic
it returns is compared with the same quantity computed directly from the full activations.

Why: the build container has no PyTorch, so before 22 Oct 2026 this loop had never run. The stand-in
checks the bookkeeping (blocks, position 0, token sample, sums, prompt entropy, loss); it cannot
check anything about real torch (devices, dtypes on a GPU). Skipped when real torch is installed:
test_neuron_attribution and the P1.1 smoke test cover that case.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REAL_TORCH = importlib.util.find_spec("torch") is not None

FAKE_TORCH_FOLDER = str(Path(__file__).resolve().parent / "fake_torch")

pytestmark = pytest.mark.skipif(REAL_TORCH, reason="real torch installed; the stand-in would shadow nothing")

NEURONS = 48
LAYERS = 2
VOCAB = 50


class Handle:
    def __init__(self, hooks, fn): self.hooks, self.fn = hooks, fn
    def remove(self): self.hooks.remove(self.fn)


class Act:
    def __init__(self): self.hooks = []
    def register_forward_hook(self, fn):
        self.hooks.append(fn)
        return Handle(self.hooks, fn)


class Layer:
    def __init__(self): self.mlp = type("M", (), {})(); self.mlp.act = Act()


class Param:
    device = "cpu"


class FakeModel:
    """
    Activations are a fixed nonlinear function of the token ids (with a dependence on the previous
    token, so the lag-one correlation is not zero), and a huge spike at position 0.
    """
    def __init__(self):
        rng = np.random.default_rng(3)
        self.embed = rng.standard_normal((LAYERS, VOCAB, NEURONS))
        self.config = type("C", (), {"intermediate_size": NEURONS})()
        self.gpt_neox = type("G", (), {})()
        self.gpt_neox.layers = [Layer() for _ in range(LAYERS)]

    def parameters(self):
        yield Param()

    def activations(self, ids):
        result = {}
        for layer in range(LAYERS):
            current = self.embed[layer][ids]
            previous = np.roll(current, 1, axis=1)
            values = current + 0.5 * previous
            values = values * (values > -0.3)
            values[:, 0, :] = values[:, 0, :] + 40.0 * (layer + 1)
            result[layer] = values.astype(np.float32)
        return result

    def __call__(self, batch_ids):
        ids = batch_ids.a
        acts = self.activations(ids)
        for layer in range(LAYERS):
            for hook in list(self.gpt_neox.layers[layer].mlp.act.hooks):
                hook(None, None, sys.modules['torch'].Tensor(acts[layer]))
        logits = np.zeros(ids.shape + (VOCAB,), dtype=np.float32)
        return type("O", (), {"logits": sys.modules["torch"].Tensor(logits)})()



def test_collection_loop_statistics_match_direct_computation(monkeypatch):
    # the stand-in is visible only inside this test, so other tests never mistake it for torch
    monkeypatch.syspath_prepend(FAKE_TORCH_FOLDER)
    try:
        run_collection_check()
    finally:
        for name in list(sys.modules):
            if name == "torch" or name.startswith("torch."):
                del sys.modules[name]


def run_collection_check():
    import torch
    from nds.activations import collect_neuron_moments
    from nds.baselines import prompt_entropy
    from nds.noise import lag_one_correlations
    from nds.token_sample import choose_token_positions

    globals()["torch"] = torch

    rng = np.random.default_rng(0)
    sequences, length, skip = 24, 33, 1
    ids = rng.integers(1, VOCAB, size=(sequences, length))
    ids[:, 0] = 0

    positions = choose_token_positions(sequences, length, sample_size=40, skip_first_positions=skip, seed=1)
    model = FakeModel()
    out = collect_neuron_moments(model, ids, batch_size=5, number_of_blocks=4, skip_first_positions=skip, token_positions=positions)

    direct = model.activations(ids)

    for layer in range(LAYERS):
        full = direct[layer].astype(np.float64)
        kept = full[:, skip:, :]
        flat = kept.reshape(-1, NEURONS)
        merged = out["moments"][layer].merged()

        assert merged.count == flat.shape[0]
        assert np.allclose(merged.mean, flat.mean(axis=0), atol=1e-5)
        assert np.allclose(merged.covariance(), np.cov(flat, rowvar=False), rtol=1e-4, atol=1e-5)
        assert np.allclose(merged.max_abs_activation, np.abs(flat).max(axis=0), atol=1e-5)
        assert np.allclose(out["mean_absolute_activation"][layer], np.abs(flat).mean(axis=0), atol=1e-5)
        assert np.allclose(out["raw_moments"][layer]["mean_cube"], (flat ** 3).mean(axis=0), rtol=1e-4)
        assert np.allclose(out["raw_moments"][layer]["mean_fourth_power"], (flat ** 4).mean(axis=0), rtol=1e-4)
        assert np.allclose(out["raw_moments"][layer]["positive_fraction"], (flat > 0).mean(axis=0))

        # position 0 is recorded on its own and kept out of everything else
        assert np.allclose(out["position_zero"][layer]["max_absolute"], np.abs(full[:, 0, :]).max(axis=0), atol=1e-4)
        assert merged.max_abs_activation.max() < out["position_zero"][layer]["max_absolute"].max() / 5

        assert np.allclose(out["token_sample"][layer], full[positions[0], positions[1], :], atol=1e-5)

        # sequence i goes to block i % 4
        for block in range(4):
            assert out["moments"][layer].blocks[block].count == kept[block::4].reshape(-1, NEURONS).shape[0]

        entropies = []
        for row in range(sequences):
            entropies.append(prompt_entropy(kept[row])["raw"])
        assert abs(out["prompt_entropy"][layer]["mean_raw"] - np.mean(entropies)) < 1e-6

        # two slightly different estimators of the same lag-one correlation (separate vs pooled means)
        assert abs(out["lag_one_correlation"][layer] - np.median(lag_one_correlations(kept))) < 0.01

    # all-zero logits: the loss is log(vocabulary)
    assert abs(out["probe_loss"] - np.log(VOCAB)) < 1e-9
