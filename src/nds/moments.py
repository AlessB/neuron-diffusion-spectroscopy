"""
moments.py
==========

Running statistics of neuron activations, so we never have to keep the activations themselves.

Why this file exists
--------------------
For one Pythia-70M layer the probe text gives a table of 51,200 tokens x 2,048 neurons.
Kept for all 6 layers and 154 checkpoints that would be ~390 GB. Everything we compute later
(correlations, the neuron graph, RankMe, alpha-ReQ, the rotation control) only needs:

    - how many tokens we saw          n
    - the mean activation per neuron  mean        (vector, length m)
    - the centred scatter matrix      scatter     (m x m),  scatter = sum over tokens of (x - mean)(x - mean)^T

so we keep those and throw each batch of activations away as soon as it is added.

The covariance is then  scatter / (n - 1).

How batches are combined
------------------------
Adding raw sums (sum of x, sum of x x^T) and subtracting n * mean * mean^T at the end loses
precision when the mean is large compared with the spread (catastrophic cancellation).
Instead each batch is centred on its own mean and merged with the parallel update of
Chan, Golub & LeVeque (1979), which is exact and numerically stable:

    n      = n_a + n_b
    delta  = mean_b - mean_a
    mean   = mean_a + delta * n_b / n
    scatter = scatter_a + scatter_b + delta delta^T * n_a * n_b / n

Credit
------
- T. F. Chan, G. H. Golub, R. J. LeVeque (1983). "Algorithms for computing the sample variance:
  analysis and recommendations." The American Statistician 37(3), 242-247. (the journal paper;
  the update first appeared in their 1979 Stanford report STAN-CS-79-773, "Updating formulae and
  a pairwise algorithm for computing sample variances")
  The same update is used by numpy-free streaming tools everywhere (e.g. Welford-style online
  variance); we write it out explicitly so it can be read and checked.

Blocks (for the token bootstrap)
--------------------------------
The probe text is split into a fixed number of blocks (default 8). Each block keeps its own
moments. The full-data moments are the merge of all blocks, and a bootstrap replicate is a merge
of blocks drawn with replacement. This is how we estimate "how much would this spectrum wobble if
we had used different text" without storing the text's activations.
"""

import numpy as np


class NeuronMoments:
    """
    Running count, mean and centred scatter matrix for m neurons.

    Use it like this:

        moments = NeuronMoments(number_of_neurons=2048)

        for batch in batches:                 # batch has shape (tokens_in_batch, 2048)
            moments.add_batch(batch)

        covariance = moments.covariance()     # (2048, 2048)
    """

    def __init__(self, number_of_neurons):
        self.number_of_neurons = int(number_of_neurons)

        # Everything is accumulated in float64: the activations arrive as float32, but sums of
        # tens of thousands of products need the extra precision.
        self.count = 0
        self.mean = np.zeros(self.number_of_neurons, dtype=np.float64)
        self.scatter = np.zeros(
            (self.number_of_neurons, self.number_of_neurons),
            dtype=np.float64,
        )

        # How often each neuron is "on". Used to find dead neurons.
        # A token counts as "on" for a neuron when its activation is above `active_threshold`.
        self.active_count = np.zeros(self.number_of_neurons, dtype=np.int64)
        self.active_threshold = 0.0

        # Largest absolute activation seen per neuron. Used to find "massive activation" outliers
        # (Sun et al. 2024), which we want to be able to switch in and out of the analysis.
        self.max_abs_activation = np.zeros(self.number_of_neurons, dtype=np.float64)

    # ------------------------------------------------------------------------------------------
    # adding data
    # ------------------------------------------------------------------------------------------

    def add_batch(self, activations):
        """
        Add a batch of activations of shape (tokens, neurons).

        The batch is centred on its own mean first, then merged into the running totals with the
        Chan-Golub-LeVeque update (see the module docstring).
        """
        batch = np.asarray(activations, dtype=np.float64)

        if batch.ndim != 2:
            raise ValueError(f"expected a 2-D array (tokens, neurons), got shape {batch.shape}")

        if batch.shape[1] != self.number_of_neurons:
            raise ValueError(
                f"expected {self.number_of_neurons} neurons, got {batch.shape[1]}"
            )

        batch_count = batch.shape[0]

        if batch_count == 0:
            return

        batch_mean = batch.mean(axis=0)
        batch_centred = batch - batch_mean
        batch_scatter = batch_centred.T @ batch_centred

        self._merge(batch_count, batch_mean, batch_scatter)

        # bookkeeping for dead neurons and outliers
        is_active = batch > self.active_threshold
        self.active_count += is_active.sum(axis=0)

        batch_max_abs = np.abs(batch).max(axis=0)
        self.max_abs_activation = np.maximum(self.max_abs_activation, batch_max_abs)

    def add_batch_summary(
        self,
        batch_count,
        batch_mean,
        batch_scatter,
        batch_active_count,
        batch_max_abs,
    ):
        """
        Add a batch that was already summarised elsewhere (for example on the GPU by
        activations.py): its token count, mean, centred scatter matrix, per-neuron active count
        and per-neuron largest absolute value.

        This is the same as `add_batch`, without ever moving the full activation table to the CPU.
        """
        if batch_count == 0:
            return

        self._merge(
            int(batch_count),
            np.asarray(batch_mean, dtype=np.float64),
            np.asarray(batch_scatter, dtype=np.float64),
        )

        self.active_count += np.asarray(batch_active_count, dtype=np.int64)

        self.max_abs_activation = np.maximum(
            self.max_abs_activation,
            np.asarray(batch_max_abs, dtype=np.float64),
        )

    def merge_from(self, other):
        """
        Merge another NeuronMoments object into this one (used to combine blocks).
        """
        if other.number_of_neurons != self.number_of_neurons:
            raise ValueError("cannot merge moments with different numbers of neurons")

        if other.count == 0:
            return

        self._merge(other.count, other.mean, other.scatter)

        self.active_count += other.active_count
        self.max_abs_activation = np.maximum(self.max_abs_activation, other.max_abs_activation)

    def _merge(self, other_count, other_mean, other_scatter):
        """
        The Chan-Golub-LeVeque (1979) pairwise update. Written out one step per line.
        """
        if self.count == 0:
            self.count = int(other_count)
            self.mean = np.array(other_mean, dtype=np.float64)
            self.scatter = np.array(other_scatter, dtype=np.float64)
            return

        total_count = self.count + other_count

        delta = other_mean - self.mean

        new_mean = self.mean + delta * (other_count / total_count)

        cross_term = np.outer(delta, delta) * (self.count * other_count / total_count)
        new_scatter = self.scatter + other_scatter + cross_term

        self.count = int(total_count)
        self.mean = new_mean
        self.scatter = new_scatter

    # ------------------------------------------------------------------------------------------
    # reading results
    # ------------------------------------------------------------------------------------------

    def covariance(self):
        """
        Sample covariance  scatter / (n - 1).
        """
        if self.count < 2:
            raise ValueError("need at least two tokens to compute a covariance")

        return self.scatter / (self.count - 1)

    def firing_rate(self):
        """
        Fraction of tokens on which each neuron was above `active_threshold`.
        """
        if self.count == 0:
            return np.zeros(self.number_of_neurons)

        return self.active_count / self.count

    def copy(self):
        clone = NeuronMoments(self.number_of_neurons)
        clone.count = self.count
        clone.mean = self.mean.copy()
        clone.scatter = self.scatter.copy()
        clone.active_count = self.active_count.copy()
        clone.active_threshold = self.active_threshold
        clone.max_abs_activation = self.max_abs_activation.copy()
        return clone


class BlockedNeuronMoments:
    """
    One NeuronMoments per block of probe text, so we can bootstrap over blocks later.

    Tokens are assigned to blocks by the caller (usually: block = sequence_index % number_of_blocks),
    which keeps whole sequences together in one block.
    """

    def __init__(self, number_of_neurons, number_of_blocks=8):
        self.number_of_neurons = int(number_of_neurons)
        self.number_of_blocks = int(number_of_blocks)

        self.blocks = []
        for block_index in range(self.number_of_blocks):
            self.blocks.append(NeuronMoments(self.number_of_neurons))

    def add_batch(self, activations, block_index):
        self.blocks[block_index].add_batch(activations)

    def add_batch_summary(
        self,
        block_index,
        batch_count,
        batch_mean,
        batch_scatter,
        batch_active_count,
        batch_max_abs,
    ):
        self.blocks[block_index].add_batch_summary(
            batch_count,
            batch_mean,
            batch_scatter,
            batch_active_count,
            batch_max_abs,
        )

    def merged(self, block_indices=None):
        """
        Merge the chosen blocks (all blocks if None). Blocks may repeat, which is what a bootstrap
        replicate needs.
        """
        if block_indices is None:
            block_indices = range(self.number_of_blocks)

        total = NeuronMoments(self.number_of_neurons)

        for block_index in block_indices:
            total.merge_from(self.blocks[block_index])

        return total

    def bootstrap_replicate(self, random_generator):
        """
        One bootstrap replicate: draw `number_of_blocks` blocks with replacement and merge them.
        """
        chosen_blocks = random_generator.integers(
            low=0,
            high=self.number_of_blocks,
            size=self.number_of_blocks,
        )

        return self.merged(chosen_blocks)
