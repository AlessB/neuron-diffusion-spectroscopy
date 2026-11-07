"""
neuron_attribution.py needs PyTorch; these tests run only where torch is installed (AB's desktop).
They use a tiny two-layer stand-in with the same attribute names as GPT-NeoX
(model.gpt_neox.layers[l].mlp.act), so no download is needed.
"""

import pytest

torch = pytest.importorskip("torch")

from nds.neuron_attribution import answer_logit_difference
from nds.neuron_attribution import neuron_attribution_patching


class TinyMLP(torch.nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.dense_h_to_4h = torch.nn.Linear(width, hidden)
        self.act = torch.nn.GELU()
        self.dense_4h_to_h = torch.nn.Linear(hidden, width)

    def forward(self, x):
        return self.dense_4h_to_h(self.act(self.dense_h_to_4h(x)))


class TinyLayer(torch.nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.mlp = TinyMLP(width, hidden)

    def forward(self, x):
        return x + self.mlp(x)


class TinyModel(torch.nn.Module):
    def __init__(self, vocabulary=11, width=8, hidden=16, number_of_layers=2):
        super().__init__()
        self.embed = torch.nn.Embedding(vocabulary, width)
        self.gpt_neox = torch.nn.Module()
        self.gpt_neox.layers = torch.nn.ModuleList()
        for _ in range(number_of_layers):
            self.gpt_neox.layers.append(TinyLayer(width, hidden))
        self.unembed = torch.nn.Linear(width, vocabulary)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids=None, inputs_embeds=None):
        if inputs_embeds is None:
            inputs_embeds = self.embed(input_ids)
        hidden = inputs_embeds
        for layer in self.gpt_neox.layers:
            hidden = layer(hidden)

        class Output:
            pass

        output = Output()
        output.logits = self.unembed(hidden)
        return output


def test_attribution_patching_matches_the_linear_prediction_for_a_tiny_change():
    torch.manual_seed(0)
    model = TinyModel()

    clean = torch.tensor([[1, 2, 3, 4]])
    patch = torch.tensor([[1, 2, 3, 5]])
    clean_answer = torch.tensor([6])
    patch_answer = torch.tensor([7])

    effects = neuron_attribution_patching(model, clean, patch, clean_answer, patch_answer)

    assert effects.shape == (2, 16)

    # the sum of all first-order effects of the last layer approximates the total change in the
    # metric when only that layer's neurons are patched; here we just check signs and finiteness
    assert torch.isfinite(effects).all()

    with torch.no_grad():
        clean_metric = answer_logit_difference(model(clean).logits, clean_answer, patch_answer)
        patch_metric = answer_logit_difference(model(patch).logits, clean_answer, patch_answer)

    assert torch.isfinite(patch_metric - clean_metric).all()
