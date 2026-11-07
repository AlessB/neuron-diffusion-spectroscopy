import numpy as np
from torch import Tensor


def cross_entropy(logits, targets, reduction="sum"):
    values = logits.a.astype(np.float64)
    values = values - values.max(axis=1, keepdims=True)
    log_probabilities = values - np.log(np.exp(values).sum(axis=1, keepdims=True))
    picked = log_probabilities[np.arange(len(targets.a)), targets.a]
    return Tensor(-picked.sum())
