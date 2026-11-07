"""
induction.py
============

When does the checkpoint have induction heads? Two standard measurements on repeated random tokens.

The test sequence
-----------------
Take r random tokens and repeat them once:   [t1 t2 ... tr  t1 t2 ... tr].
In the first half nothing can be predicted (the tokens are random). In the second half a model
that can "look back and copy" (an induction head: having seen "A B" before, after "A" predict "B")
predicts every token. So:

1. Behavioural score (no attention needed):
       induction_loss_drop = mean loss on the first repeat  -  mean loss on the second repeat
   Near 0 before induction heads form, several nats after.

2. Attention score per head (Olsson et al. 2022; TransformerLens' "induction score"):
   at position i in the second repeat, how much attention goes to position i - r + 1, the token
   right after the earlier copy of the current token. Averaged over positions; ranges 0..1.
   A head with score above ~0.4 is usually called an induction head.

Credit
------
- C. Olsson, N. Elhage, N. Nanda, N. Joseph, et al. (2022). "In-context learning and induction
  heads." Transformer Circuits Thread. https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/
- N. Elhage, N. Nanda, C. Olsson, et al. (2021). "A mathematical framework for transformer circuits."
- The repeated-random-token test and per-head induction score are also the standard
  TransformerLens demo (Nanda & Bloom, https://github.com/TransformerLensOrg/TransformerLens).
- Tigges et al. (2024) use the same kind of score to time induction-head formation across Pythia
  checkpoints.
"""

import numpy as np


def repeated_random_tokens(
    number_of_sequences=64,
    repeat_length=128,
    vocabulary_size=50000,
    first_allowed_token=1000,
    random_generator=None,
):
    """
    Integer array (number_of_sequences, 2 * repeat_length): random tokens, then the same again.

    Token ids below `first_allowed_token` are avoided; in the Pythia tokenizer they include special
    and very frequent tokens, which would make the "random" half partly predictable.
    """
    if random_generator is None:
        random_generator = np.random.default_rng(0)

    first_half = random_generator.integers(
        low=first_allowed_token,
        high=vocabulary_size,
        size=(number_of_sequences, repeat_length),
    )

    sequences = np.concatenate([first_half, first_half], axis=1)

    return sequences


def induction_scores(model, sequences, batch_size=16):
    """
    Behavioural and attention-based induction scores for one checkpoint.

    The model must be loaded with attention outputs available, e.g.
        GPTNeoXForCausalLM.from_pretrained(..., attn_implementation="eager")

    Returns a dict:
        loss_first_repeat, loss_second_repeat, induction_loss_drop
        head_scores    array (layers, heads) of mean attention to the "induction target"
    """
    import torch
    import torch.nn.functional as functional

    device = next(model.parameters()).device

    sequences = np.asarray(sequences)
    repeat_length = sequences.shape[1] // 2

    loss_first_total = 0.0
    loss_second_total = 0.0
    count_first = 0
    count_second = 0

    head_score_sum = None
    head_score_count = 0

    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch = torch.as_tensor(sequences[start:start + batch_size], dtype=torch.long, device=device)

            output = model(batch, output_attentions=True)

            # per-token next-token loss, shape (batch, positions - 1)
            logits = output.logits[:, :-1, :].float()
            targets = batch[:, 1:]

            token_losses = functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="none",
            )
            token_losses = token_losses.reshape(targets.shape)

            # predictions of tokens 1..r-1 (first repeat) and r+1..2r-1 (second repeat)
            first_repeat = token_losses[:, : repeat_length - 1]
            second_repeat = token_losses[:, repeat_length:]

            loss_first_total = loss_first_total + float(first_repeat.sum())
            loss_second_total = loss_second_total + float(second_repeat.sum())
            count_first = count_first + int(first_repeat.numel())
            count_second = count_second + int(second_repeat.numel())

            # attention: one tensor per layer, shape (batch, heads, query, key)
            layer_scores = []
            for attention in output.attentions:
                # query positions in the second repeat; their induction target is i - r + 1
                queries = torch.arange(repeat_length, 2 * repeat_length, device=device)
                keys = queries - repeat_length + 1

                attention_to_target = attention[:, :, queries, keys]

                layer_scores.append(attention_to_target.mean(dim=(0, 2)).float().cpu().numpy())

            batch_head_scores = np.stack(layer_scores)

            if head_score_sum is None:
                head_score_sum = batch_head_scores * 0.0

            head_score_sum = head_score_sum + batch_head_scores
            head_score_count = head_score_count + 1

    loss_first = loss_first_total / count_first
    loss_second = loss_second_total / count_second

    result = {
        "loss_first_repeat": loss_first,
        "loss_second_repeat": loss_second,
        "induction_loss_drop": loss_first - loss_second,
        "head_scores": head_score_sum / head_score_count,
    }

    return result
