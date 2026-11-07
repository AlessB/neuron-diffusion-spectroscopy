"""
checkpoints.py
==============

Which Pythia models and training steps exist, and which subsets we use.

Pythia (Biderman et al. 2023) saved 154 checkpoints per model, the same steps for every size:

    step 0                                   (random initialisation)
    steps 1, 2, 4, 8, ..., 512               (10 log-spaced early steps)
    steps 1000, 2000, ..., 143000            (143 evenly spaced steps)

Each step is 1024 sequences x 2048 tokens = 2,097,152 tokens (about 2.1 million), so step 1000
is about 2.1 billion tokens and step 143000 (the final model) about 300 billion.

On the Hugging Face Hub every checkpoint is a git branch of the model repo, loaded with
    revision="step1000"
(see https://github.com/EleutherAI/pythia, "Models" section).

Credit
------
- S. Biderman, H. Schoelkopf, Q. Anthony, H. Bradley, K. O'Brien, E. Hallahan, M. A. Khan,
  S. Purohit, U. S. Prashanth, E. Raff, A. Skowron, L. Sutawika, O. van der Wal (2023).
  "Pythia: A suite for analyzing large language models across training and scaling." ICML 2023.
  Models: https://huggingface.co/EleutherAI  Code: https://github.com/EleutherAI/pythia
- O. van der Wal et al. (2025). "PolyPythias: Stability and outliers across fifty language model
  pre-training runs." ICLR 2025. (extra random seeds for the small sizes)
- Stanford CRFM Mistral GPT-2 small (2021): five seeds (alias, battlestar, caprica, darkmatter,
  expanse) trained for 400,000 steps on OpenWebText, 609 checkpoints each, published as Hugging
  Face tags `checkpoint-<step>` (load with revision="checkpoint-1000"). Schedule read from the tag
  list of stanford-crfm/alias-gpt2-small-x21 on 23 Oct 2026. https://crfm.stanford.edu/2021/08/26/mistral.html
"""

TOKENS_PER_STEP = 1024 * 2048

FINAL_STEP = 143000


def all_pythia_steps():
    """
    The 154 published checkpoint steps, in training order.
    """
    steps = [0]

    for power in range(10):
        steps.append(2 ** power)

    for step in range(1000, FINAL_STEP + 1, 1000):
        steps.append(step)

    return steps


def week_one_steps():
    """
    The 25-checkpoint subset for the first pass (deep-dive report, section 11): every early
    log-spaced step, then a roughly log-spaced selection of the later ones.
    """
    steps = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    later = [
        1000,
        2000,
        3000,
        4000,
        6000,
        8000,
        12000,
        16000,
        24000,
        32000,
        48000,
        64000,
        96000,
        143000,
    ]

    steps.extend(later)

    return steps


def tokens_seen(step):
    """
    Number of training tokens seen at a given step.
    """
    return step * TOKENS_PER_STEP


def revision_name(step):
    """
    Hugging Face branch name for a step, e.g. 1000 -> "step1000".
    """
    return f"step{int(step)}"


def crfm_gpt2_small_steps():
    """
    The 609 checkpoint steps of each Stanford CRFM GPT-2 small run (D45): every 10 steps to 100,
    every 50 to 2,000, every 100 to 20,000, every 1,000 to 400,000; 9 checkpoints strictly between
    steps 512 and 1,000, where Pythia has none.

    A CRFM step is not a Pythia step: compare the two models in tokens seen, not steps. Tokens per
    CRFM step are not yet checked (read the run's training config on the Hub before P2.7's figures).
    """
    steps = list(range(0, 101, 10))
    steps.extend(range(150, 2001, 50))
    steps.extend(range(2100, 20001, 100))
    steps.extend(range(21000, 400001, 1000))
    return steps
