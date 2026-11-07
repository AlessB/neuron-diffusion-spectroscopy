"""
controls.py
===========

The controls that make a result (positive or negative) believable.

1. Random rotation of the neurons
---------------------------------
Replace the activation matrix X (tokens x neurons) by X Q, with Q a random orthogonal matrix.
Every new "neuron" is then a blend of all the old ones. The covariance becomes Q^T C Q:

    - its eigenvalues are unchanged      -> RankMe, alpha-ReQ, matrix entropy are IDENTICAL
    - its correlation structure is mixed -> the neuron graph loses its modules

So: if a neuron-graph event survives the rotation, it was never about specific neurons; it was
the covariance spectrum seen through a different lens. If it disappears under rotation while the
baselines stay put, the event is neuron-aligned, which is the project's central claim.

We rotate the COVARIANCE (Q^T C Q), which is mathematically the same as rotating the activations
and recomputing the covariance, and costs nothing extra.

A uniformly random orthogonal matrix ("Haar measure") is made by QR-factorising a Gaussian matrix
and fixing the signs of R's diagonal, as explained by Mezzadri (2007). Without the sign fix the
result is orthogonal but not uniformly random.

2. Token bootstrap
------------------
Recompute everything on resampled blocks of the probe text (moments.BlockedNeuronMoments). The
spread of a quantity over replicates is its noise band. An "event" must stand outside that band.

Credit
------
- F. Mezzadri (2007). "How to generate random matrices from the classical compact groups."
  Notices of the AMS 54(5):592-604.
- B. Efron (1979). "Bootstrap methods: another look at the jackknife." Annals of Statistics 7(1).
"""

import numpy as np


def random_orthogonal_matrix(size, random_generator):
    """
    A Haar-random orthogonal matrix of shape (size, size) (Mezzadri 2007).
    """
    gaussian = random_generator.standard_normal((size, size))

    q_matrix, r_matrix = np.linalg.qr(gaussian)

    # multiply each column of Q by the sign of the matching diagonal entry of R
    diagonal_signs = np.sign(np.diag(r_matrix))
    diagonal_signs[diagonal_signs == 0] = 1.0

    orthogonal = q_matrix * diagonal_signs[np.newaxis, :]

    return orthogonal


def rotate_covariance(covariance, orthogonal):
    """
    Covariance of the rotated activations X Q:   Q^T C Q.
    """
    rotated = orthogonal.T @ covariance @ orthogonal

    # exact symmetry after rounding
    rotated = (rotated + rotated.T) / 2.0

    return rotated


def rotate_live_covariance(covariance, is_dead, random_generator):
    """
    Rotate only the live neurons among themselves, leaving dead neurons out of the mix.

    Why: dead neurons are dropped before building the graph. If we rotated them in, every new
    "neuron" would pick up a little of a dead one and none would be dead any more, so the rotated
    graph would have more nodes than the original and the comparison would not be like for like.

    Returns the rotated covariance of the live neurons only (m_live x m_live).
    """
    live_indices = np.flatnonzero(~np.asarray(is_dead))

    live_covariance = covariance[np.ix_(live_indices, live_indices)]

    orthogonal = random_orthogonal_matrix(len(live_indices), random_generator)

    return rotate_covariance(live_covariance, orthogonal)
