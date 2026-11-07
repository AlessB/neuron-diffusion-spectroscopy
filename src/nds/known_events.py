"""
known_events.py
===============

What other papers report happening inside Pythia during training, in Pythia steps, so every
figure can show it next to our readouts (vertical bands). Pure data.

One Pythia step is 1,024 sequences x 2,048 tokens = 2,097,152 tokens, so 1e9 tokens is about step
477. Learning-rate warmup ends at step 1,430 (1% of 143,000).

Most of these were measured on 160m or larger models, not on 70m: `model` says which. A 70m
neuron-level timeline does not exist yet; that gap is part of what this project offers.

Collected 22 Oct 2026 (research2/R1, checked against the papers' pages; the He et al. venue is
claimed by their repository only).

Credit (one line per source; full citations in CREDITS.md)
------
- Tigges et al. 2024, "LLM circuit analyses are consistent across training and scale", NeurIPS 2024.
- Ge et al. 2026, "Evolution of concepts in language model pre-training", ICLR 2026.
- He et al. 2026, "Learning to read out", arXiv:2610.21025.
- Stecher et al. 2026, "Features have life history. And we should care.", arXiv:2605.18789.
- Lee, Smith, Adam, Hoogland 2025, "Influence dynamics and stagewise data attribution", arXiv:2510.12071.
- van der Wal et al. 2025, "PolyPythias", ICLR 2025.
- Duan et al. 2025, "How syntax specialization emerges in language models", arXiv:2505.19548.
- Gallego-Feliciano et al. 2025, "Hidden dynamics of massive activations in transformer training", arXiv:2508.03616.
"""

KNOWN_EVENTS = [
    {
        "first_step": 1,
        "last_step": 8,
        "label": "seeds diverge most (syntax)",
        "model": "160m seeds",
        "source": "Duan et al. 2025",
    },
    {
        "first_step": 128,
        "last_step": 128,
        "label": "initialisation features drop and recover",
        "model": "160m",
        "source": "Ge et al. 2026",
    },
    {
        "first_step": 512,
        "last_step": 1600,
        "label": "readout reorganises",
        "model": "160m, 1b, 6.9b",
        "source": "He et al. 2026",
    },
    {
        "first_step": 1000,
        "last_step": 1000,
        "label": "induction heads; new features start; SAE births slow",
        "model": "most sizes",
        "source": "Tigges et al. 2024; Ge et al. 2026; Stecher et al. 2026",
    },
    {
        "first_step": 1430,
        "last_step": 1430,
        "label": "learning-rate warmup ends",
        "model": "all",
        "source": "Pythia config",
    },
    {
        "first_step": 6000,
        "last_step": 6000,
        "label": "training-map transition",
        "model": "70m",
        "source": "PolyPythias",
    },
    {
        "first_step": 10000,
        "last_step": 10000,
        "label": "end of SAE assembly phase",
        "model": "160m, 410m",
        "source": "Stecher et al. 2026",
    },
    {
        "first_step": 28000,
        "last_step": 28000,
        "label": "training-map transition",
        "model": "70m",
        "source": "PolyPythias",
    },
    {
        "first_step": 30000,
        "last_step": 30000,
        "label": "induction influence peaks",
        "model": "Pythia",
        "source": "Lee et al. 2025",
    },
    {
        "first_step": 58000,
        "last_step": 58000,
        "label": "training-map transition",
        "model": "70m",
        "source": "PolyPythias",
    },
    {
        "first_step": 60000,
        "last_step": 60000,
        "label": "massive activations peak (residual stream)",
        "model": "all sizes",
        "source": "Gallego-Feliciano et al. 2025",
    },
    {
        "first_step": 94000,
        "last_step": 94000,
        "label": "training-map transition",
        "model": "70m",
        "source": "PolyPythias",
    },
]


def events_for_model(model_word=None):
    """
    The events, optionally only those whose `model` field contains model_word (e.g. "70m").
    """
    if model_word is None:
        return list(KNOWN_EVENTS)

    chosen = []
    for event in KNOWN_EVENTS:
        if model_word in event["model"]:
            chosen.append(event)

    return chosen


def transitions_in_window(steps, first_step, last_step):
    """
    For the transitions between neighbouring checkpoints (steps[i] -> steps[i + 1]), True when the
    transition overlaps [first_step, last_step]. This is the `in_window` that
    timeline.circular_shift_p_value needs.
    """
    flags = []

    for index in range(len(steps) - 1):
        start = steps[index]
        stop = steps[index + 1]

        overlaps = stop >= first_step and start <= last_step

        flags.append(overlaps)

    return flags
