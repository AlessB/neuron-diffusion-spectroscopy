"""
flow.py
=======

Following the neuron-graph modes from one checkpoint to the next ("spectral flow"), and fitting
avoided crossings with a two-level model.

Why modes can be followed at all
--------------------------------
A neuron keeps its identity across checkpoints: neuron 731 of layer 2 at step 1000 is the same
row of the same weight matrix at step 2000. So an eigenvector at one checkpoint (a list of weights
over neurons) can be compared directly with the eigenvectors at the next checkpoint. The mode that
"continues" mode i is the one it overlaps with most:

    overlap(i, j) = | < v_i(step a), v_j(step b) > |

Several modes can compete for the same successor, so we solve the assignment for all modes at
once (the Hungarian algorithm, Kuhn 1955; scipy.optimize.linear_sum_assignment) to maximise the
total overlap.

Two practical details:
- Dead neurons differ between checkpoints. We compare vectors on the neurons alive at BOTH steps,
  then renormalise. (A vector that loses a lot of weight that way is flagged by its low norm.)
- Eigenvectors have arbitrary sign. The absolute value in the overlap takes care of it.

Avoided crossings: the two-level model
--------------------------------------
When two eigenvalues approach each other as training goes on and then separate again without
crossing, the textbook model is a 2 x 2 Hamiltonian with diagonal energies that cross linearly and
a constant coupling g between them:

    H(s) = [[ e_a(s), g ], [ g, e_b(s) ]],      e_a - e_b = v (s - s0)

with eigenvalues

    mu_plus/minus(s) = c(s) +/- sqrt( (v (s - s0) / 2)^2 + g^2 ),    c(s) = c0 + c1 (s - s0)

The minimum gap is 2g, reached at s0. v is how fast the two bare modes sweep past each other.
In the Landau-Zener picture, g^2 / v measures how "adiabatic" the passage is: large means the
modes smoothly exchange character (each eigenvector swaps which module it lives on); small means
they pass almost as if uncoupled.

For us s is training time (log of step is usually the natural axis) and mu are Laplacian
eigenvalues. The fitted g is an EFFECTIVE coupling: it can include indirect paths through other
modules (the probe in the deep-dive found the minimum gap was not monotone in the direct coupling).

Credit
------
- H. W. Kuhn (1955). "The Hungarian method for the assignment problem." Naval Research Logistics
  Quarterly 2(1-2):83-97.
- J. von Neumann, E. Wigner (1929). "Uber das Verhalten von Eigenwerten bei adiabatischen
  Prozessen." Physikalische Zeitschrift 30, 467-470. (the non-crossing rule; corrected 18 Oct
  2026, an earlier version cited their other 1929 paper by mistake)
- C. Zener (1932). "Non-adiabatic crossing of energy levels." Proc. Royal Society A 137.
- S. N. Shevchenko, S. Ashhab, F. Nori (2010). "Landau-Zener-Stuckelberg interferometry." Physics
  Reports 492(1). (the adiabaticity parameter)
- A. Bjorck, G. H. Golub (1973). "Numerical methods for computing angles between linear
  subspaces." Mathematics of Computation 27(123). (match_mode_groups)
- AB's undergraduate work on coupled optical resonators (mode hybridisation) is the intuition here.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.optimize import least_squares


# ----------------------------------------------------------------------------------------------
# matching modes between two checkpoints
# ----------------------------------------------------------------------------------------------

def restrict_to_common_neurons(
    vectors_a,
    live_indices_a,
    vectors_b,
    live_indices_b,
):
    """
    Keep only the rows of each eigenvector matrix that belong to neurons alive at both steps.

    vectors_a has one row per live neuron of step a (in the order of live_indices_a), one column
    per mode. Same for b. Returns the two restricted matrices (rows in the same neuron order) and
    the norm each column kept (1.0 = nothing lost).
    """
    common = np.intersect1d(live_indices_a, live_indices_b)

    position_in_a = {neuron: row for row, neuron in enumerate(live_indices_a)}
    position_in_b = {neuron: row for row, neuron in enumerate(live_indices_b)}

    rows_a = [position_in_a[neuron] for neuron in common]
    rows_b = [position_in_b[neuron] for neuron in common]

    restricted_a = vectors_a[rows_a, :]
    restricted_b = vectors_b[rows_b, :]

    kept_norm_a = np.linalg.norm(restricted_a, axis=0)
    kept_norm_b = np.linalg.norm(restricted_b, axis=0)

    restricted_a = restricted_a / np.maximum(kept_norm_a, 1e-300)
    restricted_b = restricted_b / np.maximum(kept_norm_b, 1e-300)

    return restricted_a, restricted_b, kept_norm_a, kept_norm_b


def match_modes(vectors_a, vectors_b):
    """
    Best one-to-one assignment of modes at step a to modes at step b.

    vectors_a, vectors_b: (neurons, modes) with the SAME neuron rows (see restrict_to_common_neurons).

    Returns a dict:
        successor[i]     index of the mode at step b that continues mode i of step a
        overlap[i]       |< v_i(a), v_successor(b) >|   (1 = identical, 0 = unrelated)
        overlap_matrix   all pairwise |overlaps|
    """
    overlap_matrix = np.abs(vectors_a.T @ vectors_b)

    # linear_sum_assignment minimises cost, so use the negative overlap
    rows, columns = linear_sum_assignment(-overlap_matrix)

    number_of_modes = vectors_a.shape[1]

    successor = np.full(number_of_modes, -1, dtype=int)
    overlap = np.zeros(number_of_modes)

    for row, column in zip(rows, columns):
        successor[row] = column
        overlap[row] = overlap_matrix[row, column]

    result = {
        "successor": successor,
        "overlap": overlap,
        "overlap_matrix": overlap_matrix,
    }

    return result


def group_near_degenerate_modes(eigenvalues, tolerance):
    """
    Split modes into groups of neighbours closer than `tolerance` (research2/R4 pitfall P4).

    Inside a group of nearly equal eigenvalues the individual eigenvectors are arbitrary mixtures:
    any rotation inside the group is an equally good answer, and the solver picks one at random.
    Matching such modes one by one manufactures fake swaps that look like avoided crossings. Only
    the group's subspace is well defined, so we match groups.

    eigenvalues   sorted Laplacian eigenvalues of the modes being matched
    tolerance     a spacing below this joins two modes; use the bootstrap spread of the eigenvalues
                  (e.g. the median over modes of the bootstrap standard deviation), so "degenerate"
                  means "closer than the probe text can resolve"
    Returns a list of lists of mode indices.
    """
    values = np.asarray(eigenvalues, dtype=np.float64)

    groups = []

    if len(values) == 0:
        return groups

    current = [0]

    for index in range(1, len(values)):
        if values[index] - values[index - 1] < tolerance:
            current.append(index)
        else:
            groups.append(current)
            current = [index]

    groups.append(current)

    return groups


def match_mode_groups(vectors_a, groups_a, vectors_b, groups_b):
    """
    Match groups of modes (group_near_degenerate_modes) between two checkpoints by how much of
    one group's subspace the other contains.

    For groups G (step a, k_G modes) and H (step b, k_H modes), with orthonormal columns:
        overlap(G, H) = ||V_G^T V_H||_F^2 / min(k_G, k_H)
    the mean squared cosine of the principal angles (Bjorck & Golub 1973): 1 when one subspace
    contains the other, 0 when they are orthogonal. Groups are then assigned one to one by the
    Hungarian method, as single modes are in match_modes.

    Returns a dict:
        successor[g]     index of the group at step b that continues group g of step a (-1: none)
        overlap[g]       its overlap
        overlap_matrix   all pairwise group overlaps
    """
    overlap_matrix = np.zeros((len(groups_a), len(groups_b)))

    for row, group_a in enumerate(groups_a):
        block_a, _ = np.linalg.qr(vectors_a[:, group_a])

        for column, group_b in enumerate(groups_b):
            block_b, _ = np.linalg.qr(vectors_b[:, group_b])

            cosines = block_a.T @ block_b

            smaller = min(len(group_a), len(group_b))

            overlap_matrix[row, column] = np.sum(cosines ** 2) / smaller

    rows, columns = linear_sum_assignment(-overlap_matrix)

    successor = np.full(len(groups_a), -1, dtype=int)
    overlap = np.zeros(len(groups_a))

    for row, column in zip(rows, columns):
        successor[row] = column
        overlap[row] = overlap_matrix[row, column]

    result = {
        "successor": successor,
        "overlap": overlap,
        "overlap_matrix": overlap_matrix,
    }

    return result


# ----------------------------------------------------------------------------------------------
# avoided-crossing fit
# ----------------------------------------------------------------------------------------------

def two_level_eigenvalues(times, centre_intercept, centre_slope, sweep_rate, crossing_time, coupling):
    """
    The two branches mu_minus(s), mu_plus(s) of the two-level model (see the module docstring).
    """
    shifted = times - crossing_time

    centre = centre_intercept + centre_slope * shifted

    half_splitting = np.sqrt(
        (0.5 * sweep_rate * shifted) ** 2 + coupling ** 2
    )

    lower = centre - half_splitting
    upper = centre + half_splitting

    return lower, upper


def fit_avoided_crossing(times, lower_branch, upper_branch):
    """
    Fit the two-level model to two eigenvalue tracks that approach each other.

    times          training time for each point (e.g. log10 of the step)
    lower_branch   the smaller of the two eigenvalues at each time
    upper_branch   the larger one

    Returns a dict with the fitted coupling g (half the minimum gap), sweep rate v, crossing time
    s0, the adiabaticity g^2 / |v|, and the root-mean-square residual of the fit.
    """
    times = np.asarray(times, dtype=np.float64)
    lower_branch = np.asarray(lower_branch, dtype=np.float64)
    upper_branch = np.asarray(upper_branch, dtype=np.float64)

    gaps = upper_branch - lower_branch
    closest = int(np.argmin(gaps))

    # starting guesses read straight off the data
    initial_crossing_time = times[closest]
    initial_coupling = max(gaps[closest] / 2.0, 1e-6)
    initial_centre = (upper_branch[closest] + lower_branch[closest]) / 2.0

    time_span = times.max() - times.min()
    if time_span > 0:
        initial_sweep_rate = (gaps.max() - gaps.min()) / (time_span / 2.0)
    else:
        initial_sweep_rate = 1.0

    initial_parameters = np.array(
        [
            initial_centre,
            0.0,
            max(initial_sweep_rate, 1e-6),
            initial_crossing_time,
            initial_coupling,
        ]
    )

    def residuals(parameters):
        centre_intercept, centre_slope, sweep_rate, crossing_time, coupling = parameters

        lower, upper = two_level_eigenvalues(
            times,
            centre_intercept,
            centre_slope,
            sweep_rate,
            crossing_time,
            coupling,
        )

        return np.concatenate([lower - lower_branch, upper - upper_branch])

    fit = least_squares(residuals, initial_parameters)

    centre_intercept, centre_slope, sweep_rate, crossing_time, coupling = fit.x

    coupling = abs(coupling)
    sweep_rate = abs(sweep_rate)

    root_mean_square = float(np.sqrt(np.mean(fit.fun ** 2)))

    if sweep_rate > 0:
        adiabaticity = coupling ** 2 / sweep_rate
    else:
        adiabaticity = float("inf")

    result = {
        "coupling": float(coupling),
        "minimum_gap": float(2.0 * coupling),
        "sweep_rate": float(sweep_rate),
        "crossing_time": float(crossing_time),
        "centre_intercept": float(centre_intercept),
        "centre_slope": float(centre_slope),
        "adiabaticity": float(adiabaticity),
        "rms_residual": root_mean_square,
    }

    return result


# ----------------------------------------------------------------------------------------------
# is a fitted crossing resolved, and do the eigenvectors swap the way the model says?
# (research2/R2 section 3)
# ----------------------------------------------------------------------------------------------

def crossing_diagnostics(fit, checkpoint_spacing):
    """
    Two numbers that say whether a fitted avoided crossing can be believed.

    fit                  output of fit_avoided_crossing
    checkpoint_spacing   the spacing of the checkpoints near the crossing, in the same time units
                         as the fit (e.g. the difference of log10(step + 1) between neighbours)

    width = 2 g / v: how long, in training time, the two modes spend mixed (the gap is within a
    factor sqrt 2 of its minimum). If fewer than two checkpoints fall inside it the passage is
    "unresolved": a sharp crossing between two checkpoints and a true crossing look the same.

    Landau-Zener parameter delta = g^2 / v (with our gap 2g and sweep v; Shevchenko, Ashhab & Nori
    2010 write it Delta^2 / 4v with Delta = 2g). Large delta: the modes smoothly exchange
    character. Small delta: they pass each other almost as if uncoupled.
    """
    coupling = fit["coupling"]
    sweep_rate = fit["sweep_rate"]

    if sweep_rate > 0:
        width = 2.0 * coupling / sweep_rate
        landau_zener_parameter = coupling ** 2 / sweep_rate
    else:
        width = float("inf")
        landau_zener_parameter = float("inf")

    resolved = width >= 2.0 * checkpoint_spacing

    diagnostics = {
        "width": float(width),
        "checkpoints_inside": float(width / checkpoint_spacing),
        "resolved": bool(resolved),
        "landau_zener_parameter": float(landau_zener_parameter),
    }

    return diagnostics


def predicted_mixing(times, fit):
    """
    What the two-level model predicts for the eigenvectors: at each time, the squared overlap of
    the LOWER mode with the lower mode at the first time. Starts at 1 and falls to 0 through a
    resolved avoided crossing (the two modes have swapped character), following the mixing angle.

    Compare with the measured |< v_lower(first time), v_lower(s) >|^2 from the saved eigenvectors:
    eigenvalues alone cannot tell an avoided crossing from two modes that only come close; this
    test can. In the neuron basis it also says WHICH neurons the two modes exchange, which a
    rotation destroys.
    """
    times = np.asarray(times, dtype=np.float64)

    coupling = fit["coupling"]
    sweep_rate = fit["sweep_rate"]
    crossing_time = fit["crossing_time"]

    lower_vectors = []

    for time in times:
        detuning = sweep_rate * (time - crossing_time)

        # the 2 x 2 model with diagonal energies +/- detuning / 2 (the centre drops out)
        hamiltonian = np.array(
            [
                [detuning / 2.0, coupling],
                [coupling, - detuning / 2.0],
            ]
        )

        _, vectors = np.linalg.eigh(hamiltonian)

        lower_vectors.append(vectors[:, 0])

    first = lower_vectors[0]

    overlaps = []
    for vector in lower_vectors:
        overlaps.append(float(np.dot(first, vector) ** 2))

    return np.asarray(overlaps)
