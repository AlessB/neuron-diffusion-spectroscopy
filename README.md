# Neuron-Graph Spectra of LLM Pretraining

**Status (7 Nov 2026):** the library is written and tested on synthetic data (`pytest -q`: 87
passed, 1 skipped without PyTorch). **Nothing has run on a real Pythia checkpoint yet.** The first
real runs, on pythia-70m and pythia-70m-deduped, are planned for the week of 10 Nov 2026; their
figures will appear in this repository as they are made. Every number in `research/` comes from
synthetic data, not from a trained model.

## The question

Does the geometry of a language model's neurons reorganise during pretraining in ways that
covariance measures cannot see?

For each of Pythia's training checkpoints we build a graph over the MLP neurons of every layer:
nodes are neurons, and two neurons are joined strongly when their activations rise and fall
together over a fixed probe text (correlation distance, adaptive Gaussian kernel). We then follow
the spectrum of that graph's diffusion operator through training: how many modules it has, how
localised its slow modes are on specific neurons, its diffusion spectral entropy, and how modes
move, mix and avoid each other (the avoided crossings of coupled oscillators).

The baselines are the covariance measures used on the same checkpoints by Li et al. (2025),
RankMe and alpha-ReQ, plus Skean et al.'s (2025) prompt entropy. All of them are unchanged when
the neurons are mixed by a random rotation. The neuron graph is not, and the random-rotation
control on every figure is the test of whether an event is about specific neurons. A clean
negative result is a valid outcome.

## Quick start

    python -m venv .venv
    source .venv/bin/activate
    pip install -e ".[dev]"
    pytest -q                      # synthetic data only, no GPU, no downloads
    python scripts/dry_run.py      # the whole pipeline on fake neurons

Real runs need PyTorch (install it first, from the PyTorch index that matches your GPU) and then
`pip install -e ".[model,analysis]"`. A step-by-step guide for running on a GPU will be added
to `docs/`.

## Layout

| folder | what |
|---|---|
| `src/nds/` | the library |
| `scripts/` | collect activations, analyse, re-analyse, compare runs |
| `configs/` | one YAML file per run |
| `tests/` | unit tests on synthetic data |
| `research/` | simulation and check scripts with their outputs (synthetic data) |
| `docs/` | how to use and test the code (being written) |

Run outputs are not kept in git; figures are.

## Credit

This project stands on others' work: diffusion maps (Coifman and Lafon), PHATE, M-PHATE and
diffusion spectral entropy (Wolf, Krishnaswamy and colleagues), the Pythia suite (EleutherAI), the
representation-geometry work of Li, Agrawal, Ghosh, Lajoie, Richards and colleagues, sparse feature
circuits (Marks et al.), and the spectral graph theory behind all of it. `CREDITS.md` lists every
source with links, and every module in `src/nds/` names whose method it follows. No code from GPL,
non-commercial or unlicensed repositories is copied here.

By AB (BSc Physics, Concordia), as an MSc project proposal to Prof. Guy Wolf (Université de
Montréal / Mila), 2026. MIT licence.
