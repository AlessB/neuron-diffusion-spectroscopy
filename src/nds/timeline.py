"""
timeline.py
===========

Putting the checkpoints in a row: tables over training time, spectral flow between neighbouring
checkpoints, avoided-crossing candidates, and change points.

Everything here reads what scripts/collect.py saved (storage.py layout) and is pure numpy/scipy.

Training-time axis
------------------
Pythia's checkpoints are log-spaced early and linear later. Most of the interesting events happen
early, so the natural axis for plots and fits is log10(step + 1) (the +1 keeps step 0 finite).

Change points
-------------
"Something happened here" is decided by a change-point method, not by eye. We use PELT
(Killick, Fearnhead & Eckley 2012) through the `ruptures` library (Truong, Oudre & Vayatis 2020)
when it is installed. The penalty is set from the bootstrap noise, so a change has to be larger
than what resampling the probe text produces.

Credit
------
- R. Killick, P. Fearnhead, I. A. Eckley (2012). "Optimal detection of changepoints with a linear
  computational cost." Journal of the American Statistical Association 107(500).
- C. Truong, L. Oudre, N. Vayatis (2020). "Selective review of offline change point detection
  methods." Signal Processing 167. Library: https://github.com/deepcharles/ruptures (BSD-2).
- A. Bjorck, G. H. Golub (1973). "Numerical methods for computing angles between linear
  subspaces." Mathematics of Computation 27(123). (principal angles, subspace_distance)
- L. Hubert, P. Arabie (1985). "Comparing partitions." Journal of Classification 2. (ARI)
- P. Zanardi, N. Paunkovic (2006). "Ground state overlap and quantum phase transitions." Physical
  Review E 74, 031123. S.-J. Gu (2010). "Fidelity approach to quantum phase transitions."
  International Journal of Modern Physics B 24; arXiv:0811.3127. (fidelity_susceptibility)
- N. Marzari, D. Vanderbilt (1997). "Maximally localized generalized Wannier functions for
  composite energy bands." Physical Review B 56, 12847. (the idea behind aligned_module_vectors)
- M. Gigante, A. S. Charles, S. Krishnaswamy, G. Mishne (2019). "Visualizing the PHATE of neural
  networks." NeurIPS 2019. (multislice graphs over units x epochs; identity drift is the per-unit
  quantity behind their interslice weights)
- M. He, W. F. Shen, A. Iacob, A. Jovanovic, X. Qiu, N. D. Lane (2026). "Learning to read out:
  unembedding dynamics in language model pretraining." arXiv:2610.21025. Code (MIT):
  https://github.com/hematteo/learning-to-read-out, reproduce/lifecycle/find_reorganization_steps.py
  (the circular-shift null, np.roll over checkpoint index). Idea re-implemented, not copied.
"""

from pathlib import Path

import numpy as np

from nds import storage
from nds.flow import fit_avoided_crossing
from nds.flow import match_modes
from nds.flow import restrict_to_common_neurons


NEURON_STATISTIC_COLUMNS = [
    "median_activation_kurtosis",
    "sparse_neuron_fraction",
    "prediction_neuron_fraction",
    "suppression_neuron_fraction",
    "position_zero_spike_ratio",
    "htsr_alpha_w_in",
    "htsr_alpha_w_out",
]


def append_with_controls(columns, name, spectrum, layer_summary, rotated_prefix, bootstrap_prefix, null_name):
    """
    Append one readout and the range of its controls to the table columns: the value, the min and
    max over the rotations, over the bootstrap replicates, and the first null draw. Missing values
    (older runs) become NaN.
    """
    value = spectrum.get(name)
    if value is None:
        value = np.nan
    columns[name].append(value)

    groups = [
        (rotated_prefix, [rotation["spectrum"] for rotation in layer_summary["rotations"]]),
        (bootstrap_prefix, layer_summary["bootstrap"]),
    ]
    for prefix, summaries in groups:
        values = []
        for summary in summaries:
            entry = summary.get(name)
            if entry is not None:
                values.append(entry)
        if values:
            columns[prefix + "_min"].append(float(np.min(values)))
            columns[prefix + "_max"].append(float(np.max(values)))
        else:
            columns[prefix + "_min"].append(np.nan)
            columns[prefix + "_max"].append(np.nan)

    null = layer_summary.get("null") or []
    null_value = np.nan
    if null:
        entry = null[0].get(name)
        if entry is not None:
            null_value = entry
    columns[null_name].append(null_value)


def training_time(step):
    """
    log10(step + 1), the x-axis we use for training time.
    """
    return np.log10(np.asarray(step, dtype=np.float64) + 1.0)


# ----------------------------------------------------------------------------------------------
# tables over checkpoints
# ----------------------------------------------------------------------------------------------

def empty_safe_mean(values):
    """
    Mean of a list, NaN when it is empty (a run analysed with no rotations, such as a rotated
    world, has empty rotated columns).
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def empty_safe_min(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    return float(np.min(values))


def empty_safe_max(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    return float(np.max(values))


def scalar_table(run_folder, layer):
    """
    One row per finished checkpoint, the scalar measurements of one layer.

    Returns a dict of numpy arrays (column name -> values), sorted by step.
    """
    run_folder = Path(run_folder)

    steps = storage.finished_steps(run_folder)

    columns = {
        "step": [],
        "probe_loss": [],
        "induction_loss_drop": [],
        "module_count": [],
        "dead_fraction": [],
        "rankme_li": [],
        "alpha_req": [],
        "rotated_module_count": [],
        "slow_subspace_localization": [],
        "rotated_slow_subspace_localization": [],
        "first_ten_subspace_localization": [],
        "rotated_first_ten_subspace_localization": [],
        "rotated_rankme_li": [],
        "rotated_first_ten_localization_min": [],
        "rotated_first_ten_localization_max": [],
        "null_first_ten_subspace_localization": [],
        "null_module_count": [],
        "bootstrap_module_count_min": [],
        "bootstrap_module_count_max": [],
        "bootstrap_first_ten_localization_min": [],
        "bootstrap_first_ten_localization_max": [],
        "bootstrap_first_ten_localization_std": [],
        "first_ten_aligned_localization": [],
        "rotated_first_ten_aligned_localization": [],
        "rotated_first_ten_aligned_min": [],
        "rotated_first_ten_aligned_max": [],
        "null_first_ten_aligned_localization": [],
        "module_aligned_localization": [],
        "rotated_module_aligned_min": [],
        "rotated_module_aligned_max": [],
        "bootstrap_module_aligned_min": [],
        "bootstrap_module_aligned_max": [],
        "null_module_aligned_localization": [],
        "module_aligned_support": [],
        "rotated_module_support_min": [],
        "rotated_module_support_max": [],
        "bootstrap_module_support_min": [],
        "bootstrap_module_support_max": [],
        "null_module_aligned_support": [],
        "bootstrap_first_ten_aligned_min": [],
        "bootstrap_first_ten_aligned_max": [],
        "prompt_entropy": [],
        "token_graph_dse_t1": [],
        "twonn_dimension": [],
        "embedding_condensation": [],
        "lag_one_correlation": [],
        "slow_spacing_ratio": [],
        "bulk_spacing_ratio": [],
        "rotated_bulk_spacing_ratio": [],
        "knowledge_entropy": [],
        "knowledge_entropy_sample": [],
        "knowledge_entropy_sample_rotated": [],
        "degree_floor_localization": [],
        "davis_kahan_ratio": [],
        "module_subspace_noise": [],
        "slow_mode_degree_ratio": [],
        "median_localization_whole_spectrum": [],
        "heat_capacity_peak": [],
        "heat_capacity_peak_scale": [],
        "rotated_heat_capacity_peak": [],
        "median_activation_kurtosis": [],
        "sparse_neuron_fraction": [],
        "prediction_neuron_fraction": [],
        "suppression_neuron_fraction": [],
        "position_zero_spike_ratio": [],
        "htsr_alpha_w_in": [],
        "htsr_alpha_w_out": [],
    }

    dse_times = None
    number_of_eigenvalues = None

    for step in steps:
        summary = storage.load_summary(storage.step_folder(run_folder, step))

        layer_summary = summary["layers"][str(layer)]
        spectrum = layer_summary["spectrum"]

        columns["step"].append(step)
        columns["probe_loss"].append(summary["probe_loss"])
        columns["induction_loss_drop"].append(summary["induction"]["induction_loss_drop"])
        columns["module_count"].append(spectrum["module_count"])
        columns["dead_fraction"].append(spectrum["dead_fraction"])
        columns["rankme_li"].append(layer_summary["baselines"]["rankme_li"])
        columns["alpha_req"].append(layer_summary["baselines"]["alpha_req"])

        # rotation control: the mean over the rotations, and their range (D36: 5 rotations)
        rotated_counts = []
        rotated_slow = []
        rotated_ten = []
        rotated_rankme = []
        for rotation in layer_summary["rotations"]:
            rotated_counts.append(rotation["spectrum"]["module_count"])
            rotated_slow.append(rotation["spectrum"].get("slow_subspace_localization", np.nan))
            rotated_ten.append(rotation["spectrum"].get("first_ten_subspace_localization", np.nan))
            rotated_rankme.append(rotation["baselines"]["rankme_li"])

        columns["rotated_module_count"].append(empty_safe_mean(rotated_counts))
        columns["rotated_slow_subspace_localization"].append(empty_safe_mean(rotated_slow))
        columns["rotated_first_ten_subspace_localization"].append(empty_safe_mean(rotated_ten))
        columns["rotated_first_ten_localization_min"].append(empty_safe_min(rotated_ten))
        columns["rotated_first_ten_localization_max"].append(empty_safe_max(rotated_ten))
        columns["rotated_rankme_li"].append(empty_safe_mean(rotated_rankme))

        columns["slow_subspace_localization"].append(spectrum.get("slow_subspace_localization", np.nan))
        columns["first_ten_subspace_localization"].append(spectrum.get("first_ten_subspace_localization", np.nan))

        # aligned localisation (varimax basis; added 22 Oct 2026, missing in older runs)
        aligned = spectrum.get("first_ten_aligned_localization")
        if aligned is None:
            aligned = np.nan
        columns["first_ten_aligned_localization"].append(aligned)

        rotated_aligned = []
        for rotation in layer_summary["rotations"]:
            value = rotation["spectrum"].get("first_ten_aligned_localization")
            if value is not None:
                rotated_aligned.append(value)
        if rotated_aligned:
            columns["rotated_first_ten_aligned_localization"].append(empty_safe_mean(rotated_aligned))
            columns["rotated_first_ten_aligned_min"].append(empty_safe_min(rotated_aligned))
            columns["rotated_first_ten_aligned_max"].append(empty_safe_max(rotated_aligned))
        else:
            columns["rotated_first_ten_aligned_localization"].append(np.nan)
            columns["rotated_first_ten_aligned_min"].append(np.nan)
            columns["rotated_first_ten_aligned_max"].append(np.nan)

        bootstrap_aligned = []
        for replicate in layer_summary["bootstrap"]:
            value = replicate.get("first_ten_aligned_localization")
            if value is not None:
                bootstrap_aligned.append(value)
        if bootstrap_aligned:
            columns["bootstrap_first_ten_aligned_min"].append(empty_safe_min(bootstrap_aligned))
            columns["bootstrap_first_ten_aligned_max"].append(empty_safe_max(bootstrap_aligned))
        else:
            columns["bootstrap_first_ten_aligned_min"].append(np.nan)
            columns["bootstrap_first_ten_aligned_max"].append(np.nan)

        # module-subspace aligned localisation (modes 0 .. k*-1; added 22 Oct 2026)
        append_with_controls(
            columns,
            "module_aligned_localization",
            spectrum,
            layer_summary,
            "rotated_module_aligned",
            "bootstrap_module_aligned",
            "null_module_aligned_localization",
        )

        append_with_controls(
            columns,
            "module_aligned_support",
            spectrum,
            layer_summary,
            "rotated_module_support",
            "bootstrap_module_support",
            "null_module_aligned_support",
        )

        # null: independent neurons at the same noise level (empty for older runs)
        null = layer_summary.get("null") or []
        if null:
            columns["null_first_ten_subspace_localization"].append(null[0].get("first_ten_subspace_localization", np.nan))
            columns["null_module_count"].append(null[0]["module_count"])
            null_aligned = null[0].get("first_ten_aligned_localization")
            if null_aligned is None:
                null_aligned = np.nan
            columns["null_first_ten_aligned_localization"].append(null_aligned)
        else:
            columns["null_first_ten_subspace_localization"].append(np.nan)
            columns["null_module_count"].append(np.nan)
            columns["null_first_ten_aligned_localization"].append(np.nan)

        bootstrap_counts = []
        bootstrap_ten = []
        for replicate in layer_summary["bootstrap"]:
            bootstrap_counts.append(replicate["module_count"])
            bootstrap_ten.append(replicate.get("first_ten_subspace_localization", np.nan))

        columns["bootstrap_module_count_min"].append(min(bootstrap_counts))
        columns["bootstrap_module_count_max"].append(max(bootstrap_counts))
        finite_ten = np.asarray(bootstrap_ten, dtype=np.float64)
        finite_ten = finite_ten[np.isfinite(finite_ten)]
        if len(finite_ten) > 0:
            columns["bootstrap_first_ten_localization_min"].append(float(finite_ten.min()))
            columns["bootstrap_first_ten_localization_max"].append(float(finite_ten.max()))
            # spread across replicates: a "susceptibility" that can peak near a transition
            # (research2/R2), since a system near a change responds most to small perturbations
            columns["bootstrap_first_ten_localization_std"].append(float(finite_ten.std()))
        else:
            columns["bootstrap_first_ten_localization_min"].append(np.nan)
            columns["bootstrap_first_ten_localization_max"].append(np.nan)
            columns["bootstrap_first_ten_localization_std"].append(np.nan)

        # token-side baselines (empty for older runs)
        prompt = layer_summary.get("baselines_prompt_entropy") or {}
        columns["prompt_entropy"].append(prompt.get("mean_normalised", np.nan))

        token_side = layer_summary.get("baselines_token_sample") or {}
        token_dse = token_side.get("token_graph_dse") or {}
        columns["token_graph_dse_t1"].append(token_dse.get("1", token_dse.get(1, np.nan)))
        columns["twonn_dimension"].append(token_side.get("twonn_dimension", np.nan))
        columns["embedding_condensation"].append(token_side.get("mean_token_cosine_centred", np.nan))

        lag_one = layer_summary.get("lag_one_correlation")
        if lag_one is None:
            lag_one = np.nan
        columns["lag_one_correlation"].append(lag_one)

        # knowledge entropy (Kim et al.): full probe, and on the token sample with its rotated control
        knowledge = layer_summary.get("baselines_knowledge_entropy")
        if knowledge is None:
            knowledge = np.nan
        columns["knowledge_entropy"].append(knowledge)

        columns["knowledge_entropy_sample"].append(token_side.get("knowledge_entropy_sample", np.nan))
        rotated_knowledge = token_side.get("knowledge_entropy_sample_rotated") or []
        if rotated_knowledge:
            columns["knowledge_entropy_sample_rotated"].append(empty_safe_mean(rotated_knowledge))
        else:
            columns["knowledge_entropy_sample_rotated"].append(np.nan)

        # degree floor and heat capacity (missing in older runs)
        for name in ["davis_kahan_ratio", "module_subspace_noise", "degree_floor_localization", "slow_mode_degree_ratio", "median_localization_whole_spectrum", "heat_capacity_peak", "heat_capacity_peak_scale"]:
            value = spectrum.get(name)
            if value is None:
                value = np.nan
            columns[name].append(value)

        rotated_heat = []
        for rotation in layer_summary["rotations"]:
            value = rotation["spectrum"].get("heat_capacity_peak")
            if value is not None:
                rotated_heat.append(value)
        if rotated_heat:
            columns["rotated_heat_capacity_peak"].append(empty_safe_mean(rotated_heat))
        else:
            columns["rotated_heat_capacity_peak"].append(np.nan)

        # neuron-family statistics and weight-side alpha (neuron_stats.py; missing in older runs)
        neuron_statistics = layer_summary.get("neuron_statistics") or {}
        for name in NEURON_STATISTIC_COLUMNS:
            value = neuron_statistics.get(name)
            if value is None:
                value = np.nan
            columns[name].append(value)

        # level statistics (missing in older runs)
        slow_ratio = spectrum.get("slow_spacing_ratio")
        if slow_ratio is None:
            slow_ratio = np.nan
        columns["slow_spacing_ratio"].append(slow_ratio)

        bulk_ratio = spectrum.get("bulk_spacing_ratio")
        if bulk_ratio is None:
            bulk_ratio = np.nan
        columns["bulk_spacing_ratio"].append(bulk_ratio)

        rotated_bulk = []
        for rotation in layer_summary["rotations"]:
            value = rotation["spectrum"].get("bulk_spacing_ratio")
            if value is not None:
                rotated_bulk.append(value)
        if rotated_bulk:
            columns["rotated_bulk_spacing_ratio"].append(empty_safe_mean(rotated_bulk))
        else:
            columns["rotated_bulk_spacing_ratio"].append(np.nan)

        # DSE at each diffusion time gets its own column
        if dse_times is None:
            dse_times = sorted(spectrum["dse"].keys(), key=int)
            for time_key in dse_times:
                columns[f"dse_t{time_key}"] = []

        for time_key in dse_times:
            columns[f"dse_t{time_key}"].append(spectrum["dse"][time_key])

        # the first Laplacian eigenvalues, one column each
        if number_of_eigenvalues is None:
            number_of_eigenvalues = len(spectrum["laplacian_eigenvalues"])
            for index in range(number_of_eigenvalues):
                columns[f"mu{index}"] = []

        for index in range(number_of_eigenvalues):
            columns[f"mu{index}"].append(spectrum["laplacian_eigenvalues"][index])

    table = {}
    for name, values in columns.items():
        table[name] = np.asarray(values)

    return table


# ----------------------------------------------------------------------------------------------
# spectral flow between neighbouring checkpoints
# ----------------------------------------------------------------------------------------------

def spectral_flow(run_folder, layer, number_of_modes=20):
    """
    For each pair of neighbouring checkpoints, match the first `number_of_modes` eigenvectors.

    Returns a list of dicts (one per step transition):
        step_from, step_to, successor (array), overlap (array), kept_norm_from, kept_norm_to
    A low overlap for a mode means that mode changed character between the two checkpoints.
    """
    run_folder = Path(run_folder)
    steps = storage.finished_steps(run_folder)

    transitions = []

    previous = None
    previous_step = None

    for step in steps:
        current = storage.load_layer_spectrum(storage.step_folder(run_folder, step), layer)

        if previous is not None:
            vectors_from, vectors_to, kept_from, kept_to = restrict_to_common_neurons(
                previous["symmetric_vectors"][:, :number_of_modes].astype(np.float64),
                previous["live_neuron_ids"],
                current["symmetric_vectors"][:, :number_of_modes].astype(np.float64),
                current["live_neuron_ids"],
            )

            matched = match_modes(vectors_from, vectors_to)

            transitions.append(
                {
                    "step_from": previous_step,
                    "step_to": step,
                    "successor": matched["successor"],
                    "overlap": matched["overlap"],
                    "kept_norm_from": kept_from,
                    "kept_norm_to": kept_to,
                }
            )

        previous = current
        previous_step = step

    return transitions


# ----------------------------------------------------------------------------------------------
# avoided-crossing candidates
# ----------------------------------------------------------------------------------------------

def avoided_crossing_candidates(table, first_mode=1, last_mode=10, window=3, gap_threshold=None):
    """
    Look for neighbouring Laplacian eigenvalues mu_k, mu_(k+1) whose gap has a local minimum over
    training, and fit the two-level model in a window around it.

    table: output of scalar_table (needs columns "step", "mu0", "mu1", ...).

    gap_threshold: only keep minima below this gap. If None, keep minima below the 10th percentile
    of all gaps of that pair.

    Returns a list of dicts with the mode pair, the step of the minimum and the fit.
    Candidates are only candidates: an event also needs (a) a mode swap in spectral_flow,
    (b) a gap below the bootstrap noise band, (c) no matching event in the rotation control.
    """
    times = training_time(table["step"])

    candidates = []

    for mode in range(first_mode, last_mode):
        lower = table[f"mu{mode}"]
        upper = table[f"mu{mode + 1}"]

        gaps = upper - lower

        if gap_threshold is None:
            threshold = np.percentile(gaps, 10)
        else:
            threshold = gap_threshold

        for index in range(1, len(gaps) - 1):
            is_local_minimum = gaps[index] <= gaps[index - 1] and gaps[index] <= gaps[index + 1]

            if not is_local_minimum:
                continue

            if gaps[index] > threshold:
                continue

            start = max(0, index - window)
            stop = min(len(gaps), index + window + 1)

            if stop - start < 5:
                continue

            fit = fit_avoided_crossing(times[start:stop], lower[start:stop], upper[start:stop])

            candidates.append(
                {
                    "lower_mode": mode,
                    "upper_mode": mode + 1,
                    "step": int(table["step"][index]),
                    "gap": float(gaps[index]),
                    "fit": fit,
                }
            )

    return candidates


# ----------------------------------------------------------------------------------------------
# change points
# ----------------------------------------------------------------------------------------------

def change_points(series, penalty, minimum_size=2, noise_scale=None):
    """
    PELT change points of a 1-D series (indices where a new segment starts).

    The series is divided by a scale first, so `penalty` is in units of (scale)^2:
    - noise_scale given (recommended; research2/R4 pitfall P6): the typical bootstrap spread of
      this readout, e.g. the median over checkpoints of the bootstrap standard deviation. Then a
      change must be large compared with what resampling the probe text does, and a penalty of
      about 2 log(n) is a BIC-like rule in noise units.
    - noise_scale None: the series' own standard deviation (the older behaviour). One big jump
      inflates that spread and hides the smaller changes, so use it only without a bootstrap.

    Uses the `ruptures` library if installed (pip install ruptures). Returns None if it is not,
    so callers can say "not run" instead of silently returning nothing.
    """
    try:
        import ruptures
    except ImportError:
        return None

    signal = np.asarray(series, dtype=np.float64)

    if noise_scale is not None and noise_scale > 0:
        spread = float(noise_scale)
    else:
        spread = float(np.std(signal))

    if spread > 0:
        signal = (signal - np.mean(signal)) / spread

    signal = signal.reshape(-1, 1)

    # jump=1: consider a change at every checkpoint (ruptures' default jump=5 would only allow
    # every 5th, too coarse for 25 or 154 checkpoints)
    algorithm = ruptures.Pelt(model="l2", min_size=minimum_size, jump=1)
    algorithm.fit(signal)

    predicted = algorithm.predict(pen=penalty)

    # ruptures always reports the end of the series as the last breakpoint; drop it
    breakpoints = []
    for point in predicted:
        if point < len(series):
            breakpoints.append(point)

    return breakpoints


def coincide(step_a, step_b, decades=0.15):
    """
    Do two events coincide? Within `decades` of each other on the log10(step + 1) axis.

    "Within one checkpoint" (D37) means one step early in training but up to 47k steps late in the
    week-1 grid, so coincidence is measured in training time instead (research2/R4 pitfall P6).
    0.15 decades is a factor of about 1.4 in steps. To be fixed in DECISIONS before real data.
    """
    distance = abs(float(training_time(step_a)) - float(training_time(step_b)))

    return distance <= decades


def bootstrap_noise_scale(table, column_minimum, column_maximum):
    """
    A noise scale for change_points from the bootstrap range columns of scalar_table: the median
    over checkpoints of (max - min) / 2. With 5 replicates the half-range is close to one standard
    deviation; with 20 or more it is closer to two, so say which was used.
    """
    half_ranges = (np.asarray(table[column_maximum]) - np.asarray(table[column_minimum])) / 2.0

    finite = half_ranges[np.isfinite(half_ranges)]

    if len(finite) == 0:
        return None

    return float(np.median(finite))


def partition_stability(labels_a, neuron_ids_a, labels_b, neuron_ids_b):
    """
    Adjusted Rand index between the module partitions of two checkpoints, on the neurons alive at
    both (DECISIONS D33, the secondary readout). Labels are per live neuron, in the order of the
    neuron ids saved with each spectrum.
    """
    from nds.modules import adjusted_rand_index

    position_a = {}
    for index, neuron in enumerate(np.asarray(neuron_ids_a)):
        position_a[int(neuron)] = index

    rows_a = []
    rows_b = []
    for index, neuron in enumerate(np.asarray(neuron_ids_b)):
        if int(neuron) in position_a:
            rows_a.append(position_a[int(neuron)])
            rows_b.append(index)

    if len(rows_a) < 2:
        return float("nan")

    common_a = np.asarray(labels_a)[rows_a]
    common_b = np.asarray(labels_b)[rows_b]

    return adjusted_rand_index(common_a, common_b)


def classify_events(
    steps,
    readout,
    band_low,
    band_high,
    rotated_readout,
    readout_change_indices,
    baseline_change_indices,
    tolerance=1,
):
    """
    Sort the change points of our readout into the two kinds of DECISIONS D37.

    steps                    checkpoint steps, in order
    readout                  our neuron-level readout (e.g. slow-subspace localisation), one per step
    band_low, band_high      its bootstrap band, one pair per step
    rotated_readout          the same readout under the rotation control (mean of the rotations)
    readout_change_indices   change points of `readout` (indices into steps; from change_points)
    baseline_change_indices  change points of any baseline (RankMe, alpha-ReQ, ...), merged
    tolerance                how many checkpoints apart still counts as "the same time"

    An index i qualifies if the jump readout[i] - readout[i - 1] is bigger than the bootstrap
    band's width there, and bigger than the rotated readout's own jump. It is "explained" if a
    baseline also changes within `tolerance` checkpoints, "hidden" otherwise. Replication across
    seeds is checked by the caller, which runs this per seed.

    Returns a list of dicts {index, step, kind, jump, band_width, rotated_jump}.
    """
    readout = np.asarray(readout, dtype=np.float64)
    band_low = np.asarray(band_low, dtype=np.float64)
    band_high = np.asarray(band_high, dtype=np.float64)
    rotated_readout = np.asarray(rotated_readout, dtype=np.float64)

    baseline_changes = sorted(set(int(index) for index in baseline_change_indices))

    events = []

    for index in sorted(set(int(i) for i in readout_change_indices)):
        if index <= 0 or index >= len(readout):
            continue

        jump = abs(readout[index] - readout[index - 1])

        band_width = max(band_high[index] - band_low[index], band_high[index - 1] - band_low[index - 1])

        rotated_jump = abs(rotated_readout[index] - rotated_readout[index - 1])

        if jump <= band_width or jump <= rotated_jump:
            continue

        near_baseline = False
        for baseline_index in baseline_changes:
            if abs(baseline_index - index) <= tolerance:
                near_baseline = True

        if near_baseline:
            kind = "explained"
        else:
            kind = "hidden"

        events.append(
            {
                "index": index,
                "step": int(steps[index]),
                "kind": kind,
                "jump": float(jump),
                "band_width": float(band_width),
                "rotated_jump": float(rotated_jump),
            }
        )

    return events


ATTACHMENT_THRESHOLD = 0.5


def partition_stability_series(run_folder, layer, attached_only=True):
    """
    Adjusted Rand index between each checkpoint's module partition and the previous one's
    (first entry NaN). Reads module_labels and live_neuron_ids from the saved spectra. By default
    only neurons attached to a module at both checkpoints count (module_attachment >= 0.5): the
    labels k-means gives to neurons in no module are arbitrary and flicker (research/round5).
    """
    run_folder = Path(run_folder)

    steps = storage.finished_steps(run_folder)

    values = []
    previous = None

    for step in steps:
        saved = storage.load_layer_spectrum(storage.step_folder(run_folder, step), layer)

        if "module_labels" not in saved:
            values.append(np.nan)
            previous = None
            continue

        labels = np.asarray(saved["module_labels"])
        ids = np.asarray(saved["live_neuron_ids"])

        # only neurons that belong to a module (attachment >= 0.5, analysis.py); older runs without
        # the attachment use every neuron
        if attached_only and "module_attachment" in saved:
            keep = np.asarray(saved["module_attachment"]) >= ATTACHMENT_THRESHOLD
            labels = labels[keep]
            ids = ids[keep]

        current = (labels, ids)

        if previous is None:
            values.append(np.nan)
        else:
            values.append(partition_stability(previous[0], previous[1], current[0], current[1]))

        previous = current

    return np.asarray(values, dtype=np.float64)


# ----------------------------------------------------------------------------------------------
# how fast the slow subspace turns between checkpoints
# ----------------------------------------------------------------------------------------------

def subspace_distance(vectors_a, vectors_b):
    """
    How far apart two subspaces are: the mean of sin^2 over their principal angles.

    vectors_a, vectors_b   (neurons, k) matrices with orthonormal columns, rows = the same neurons
    The singular values of A^T B are the cosines of the principal angles (Bjorck & Golub 1973), so
        distance = 1 - mean(cos^2) = 1 - ||A^T B||_F^2 / k.
    0 = the same subspace; 1 = orthogonal subspaces. With different dimensions the mean is over the
    smaller one, so 0 means "the smaller subspace sits inside the larger": a module that is added
    without disturbing the others is not counted as turning (the module count shows it instead). It is the squared projection (chordal)
    distance divided by k, and unlike matching single modes it does not care how the modes are
    mixed inside the subspace, so near-degenerate modes swapping places do not count as change.
    """
    a = np.asarray(vectors_a, dtype=np.float64)
    b = np.asarray(vectors_b, dtype=np.float64)

    # restrict_to_common_neurons renormalises each column; re-orthonormalise the whole block
    a, _ = np.linalg.qr(a)
    b, _ = np.linalg.qr(b)

    overlap = a.T @ b

    number_of_modes = min(a.shape[1], b.shape[1])

    mean_cos_squared = np.sum(overlap ** 2) / number_of_modes

    return float(1.0 - mean_cos_squared)


def subspace_change_series(run_folder, layer, first_mode=None, last_mode=None):
    """
    subspace_distance between the slow modes of each checkpoint and those of the previous one, on
    the neurons alive at both. First entry NaN.

    Default (first_mode and last_mode None): the module subspace of each checkpoint, modes
    0 .. k*-1 with k* that checkpoint's module count. A fixed set such as modes 1-10 mixes bulk
    noise modes in whenever k* < 11, and then the "turning" is mostly noise: on a planted run with
    4 modules it read about 0.66 between identical structures (tests/test_planted_events.py).
    Give first_mode and last_mode to use a fixed set instead.

    Divide by the change in log10(step + 1) to get a rate per unit of training time; the raw
    distance between Pythia's unevenly spaced checkpoints is not comparable across the run.
    """
    run_folder = Path(run_folder)

    steps = storage.finished_steps(run_folder)

    values = []
    previous = None
    previous_columns = None

    for step in steps:
        folder = storage.step_folder(run_folder, step)
        current = storage.load_layer_spectrum(folder, layer)

        if first_mode is None or last_mode is None:
            module_count = storage.load_summary(folder)["layers"][str(layer)]["spectrum"]["module_count"]
            current_columns = slice(0, max(int(module_count), 2))
        else:
            current_columns = slice(first_mode, last_mode)

        if previous is None:
            values.append(np.nan)
        else:
            vectors_from, vectors_to, _, _ = restrict_to_common_neurons(
                previous["symmetric_vectors"][:, previous_columns].astype(np.float64),
                previous["live_neuron_ids"],
                current["symmetric_vectors"][:, current_columns].astype(np.float64),
                current["live_neuron_ids"],
            )
            values.append(subspace_distance(vectors_from, vectors_to))

        previous = current
        previous_columns = current_columns

    return np.asarray(values, dtype=np.float64)


# ----------------------------------------------------------------------------------------------
# is a readout's change concentrated in a known-event window? (circular-shift null)
# ----------------------------------------------------------------------------------------------

def window_enrichment(changes, in_window):
    """
    Mean |change| over the transitions inside the window divided by the mean over all transitions.
    1 = the window is ordinary; 3 = changes there are three times the average.

    changes    (n,) change of a readout between neighbouring checkpoints (e.g. np.diff of a column)
    in_window  (n,) True for the transitions that fall inside the event window
    """
    magnitudes = np.abs(np.asarray(changes, dtype=np.float64))
    window = np.asarray(in_window, dtype=bool)

    finite = np.isfinite(magnitudes)

    overall = magnitudes[finite].mean()

    if overall <= 0 or not np.any(window & finite):
        return float("nan")

    inside = magnitudes[window & finite].mean()

    return float(inside / overall)


def circular_shift_p_value(changes, in_window, number_of_shifts=1000, seed=0):
    """
    The circular-shift null of He et al. (2026): slide the series of changes around the circle by a
    random offset (1 to n - 1 places), which keeps how bursty and how autocorrelated the readout is
    but breaks its alignment with the window, and recompute the enrichment.

    p = (1 + number of shifts with enrichment >= observed) / (1 + number_of_shifts).
    A small p says the readout changes in the window more than its own rhythm explains.

    Returns (observed enrichment, p).

    Caution: Pythia's checkpoints are unevenly spaced, so use changes per unit of log training time
    (np.diff(column) / np.diff(training_time(step))) or changes between the evenly log-spaced
    early checkpoints only; otherwise the window is compared with transitions of different length.
    """
    changes = np.asarray(changes, dtype=np.float64)
    window = np.asarray(in_window, dtype=bool)

    observed = window_enrichment(changes, window)

    number_of_transitions = len(changes)

    random_generator = np.random.default_rng(seed)

    at_least_as_large = 0

    for _ in range(number_of_shifts):
        shift = int(random_generator.integers(1, number_of_transitions))

        shifted = np.roll(changes, shift)

        value = window_enrichment(shifted, window)

        if np.isfinite(value) and value >= observed:
            at_least_as_large += 1

    p_value = (1.0 + at_least_as_large) / (1.0 + number_of_shifts)

    return observed, float(p_value)


# ----------------------------------------------------------------------------------------------
# neuron identity drift (research2/R4 idea I3)
# ----------------------------------------------------------------------------------------------

def neuron_identity_drift(sample_a, sample_b):
    """
    How much each neuron changed what it responds to between two checkpoints:
        drift_v = 1 - r_v(a, b)
    where r_v is the correlation, over the SAME sampled tokens, of neuron v's activation at
    checkpoint a with its activation at checkpoint b. 0 = it responds exactly as before, 1 = no
    relation, up to 2 = the opposite. Neurons constant at either checkpoint get NaN.

    sample_a, sample_b   (tokens, neurons) token samples (D31) at two checkpoints, same positions
    This is the per-neuron quantity behind M-PHATE's interslice weights (Gigante et al. 2019):
    neurons with small drift would be strongly tied across checkpoints in a multislice graph.
    """
    a = np.asarray(sample_a, dtype=np.float64)
    b = np.asarray(sample_b, dtype=np.float64)

    centred_a = a - a.mean(axis=0)
    centred_b = b - b.mean(axis=0)

    norm_a = np.sqrt((centred_a ** 2).sum(axis=0))
    norm_b = np.sqrt((centred_b ** 2).sum(axis=0))

    drift = np.full(a.shape[1], np.nan)

    alive = (norm_a > 1e-12) & (norm_b > 1e-12)

    correlation = (centred_a[:, alive] * centred_b[:, alive]).sum(axis=0) / (norm_a[alive] * norm_b[alive])

    drift[alive] = 1.0 - correlation

    return drift


def identity_drift_series(run_folder, layer):
    """
    neuron_identity_drift between each checkpoint and the previous one, from the saved token
    samples. Returns (steps, drift) with drift shaped (checkpoints, neurons); the first row is NaN.
    """
    run_folder = Path(run_folder)

    steps = storage.finished_steps(run_folder)

    rows = []
    previous = None

    for step in steps:
        current = storage.load_token_sample(storage.step_folder(run_folder, step), layer)["activations"]

        if previous is None:
            rows.append(np.full(current.shape[1], np.nan))
        else:
            rows.append(neuron_identity_drift(previous, current))

        previous = current

    return np.asarray(steps), np.vstack(rows)


# ----------------------------------------------------------------------------------------------
# module lineage: follow the varimax-aligned module vectors across checkpoints (22 Oct 2026)
# ----------------------------------------------------------------------------------------------

def aligned_module_vectors(saved_spectrum, module_count):
    """
    The sparsest basis of a checkpoint's module subspace (modes 0 .. k*-1, spectra.varimax_rotation),
    each column with its sign fixed so its largest entry is positive.

    In physics terms these are the "maximally localised Wannier functions" of the slowest band of
    the graph (Marzari & Vanderbilt 1997 choose the unitary mixing inside a group of bands that
    makes each function as compact as possible; varimax does the same with a fourth-moment
    criterion). Unlike raw eigenvectors they do not depend on how near-equal eigenvalues happen to
    mix, so they can be followed from one checkpoint to the next.
    """
    from nds.spectra import varimax_rotation

    columns = max(int(module_count), 2)

    basis = saved_spectrum["symmetric_vectors"][:, 0:columns].astype(np.float64)

    rotated = varimax_rotation(basis)

    for column in range(rotated.shape[1]):
        largest = np.argmax(np.abs(rotated[:, column]))
        if rotated[largest, column] < 0:
            rotated[:, column] = -rotated[:, column]

    return rotated


def module_lineage(run_folder, layer, continuation_overlap=0.5):
    """
    For each pair of neighbouring checkpoints, match the aligned module vectors (Hungarian on
    |overlap| over the neurons alive at both) and sort the modules of the later checkpoint into
    continued (overlap >= continuation_overlap with a predecessor) and new.

    Returns a list of dicts, one per transition:
        step_from, step_to, modules_from, modules_to,
        overlap        |<u_i(a), u_successor(b)>| for each module i of the earlier checkpoint
        continued      how many later modules have a predecessor above the threshold
        born           how many later modules have none (new or rebuilt modules)
    A module that keeps its neurons has overlap near 1; one whose membership was reshuffled drops
    towards 0, even when the spectrum and the module count do not move.
    """
    from nds.flow import match_modes

    run_folder = Path(run_folder)

    steps = storage.finished_steps(run_folder)

    transitions = []
    previous = None

    for step in steps:
        folder = storage.step_folder(run_folder, step)
        saved = storage.load_layer_spectrum(folder, layer)
        module_count = storage.load_summary(folder)["layers"][str(layer)]["spectrum"]["module_count"]

        current = {
            "step": step,
            "vectors": aligned_module_vectors(saved, module_count),
            "ids": saved["live_neuron_ids"],
        }

        if previous is not None:
            vectors_from, vectors_to, _, _ = restrict_to_common_neurons(
                previous["vectors"],
                previous["ids"],
                current["vectors"],
                current["ids"],
            )

            matched = match_modes(vectors_from, vectors_to)

            best_predecessor = np.max(matched["overlap_matrix"], axis=0)

            continued = int(np.sum(best_predecessor >= continuation_overlap))

            transitions.append(
                {
                    "step_from": previous["step"],
                    "step_to": step,
                    "modules_from": int(vectors_from.shape[1]),
                    "modules_to": int(vectors_to.shape[1]),
                    "overlap": matched["overlap"],
                    "continued": continued,
                    "born": int(vectors_to.shape[1]) - continued,
                }
            )

        previous = current

    return transitions


# ----------------------------------------------------------------------------------------------
# fidelity susceptibility (22 Oct 2026, research/round4 S2)
# ----------------------------------------------------------------------------------------------

def fidelity_susceptibility(distances, times):
    """
    chi = d / (dt)^2 for each pair of neighbouring checkpoints.

    d is the subspace distance between their module subspaces (1 - mean cos^2 of the principal
    angles, the many-state version of 1 - F^2 for the ground-state fidelity F) and t is training
    time, log10(step + 1). In quantum many-body physics the fidelity susceptibility peaks at a
    phase transition without any order parameter being chosen (Zanardi & Paunkovic 2006; Gu 2010);
    perturbatively it is sum_n |<n|dH|0>|^2 / (E_n - E_0)^2, so it peaks where a gap closes, which
    is why an avoided crossing shows up as a peak of height proportional to 1 / gap^2 (checked on
    our pipeline: peak x gap^2 constant across couplings, research/round4/sim2).

    Only for evenly spaced checkpoints with little noise. Dividing by (dt)^2 does not make
    Pythia's unevenly spaced checkpoints comparable: token noise does not shrink with dt, so on
    Pythia's schedule the raw chi peaks on the last checkpoints whatever happens
    (research/round4/sim6). Use subspace_change_signal_to_noise there. Caution: like the
    subspace distance it is unchanged by a fixed rotation of the neurons (angles between
    subspaces are preserved), so it says "the structure changed here", not "specific neurons
    changed"; ARI and module lineage carry that (research/round4/sim1).

    distances   (n - 1,) subspace distances between neighbouring checkpoints (NaN allowed)
    times       (n,) training times of the checkpoints
    Returns (n - 1,) values.
    """
    distances = np.asarray(distances, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)

    steps = np.diff(times)

    result = np.full(len(distances), np.nan)

    usable = steps > 0
    result[usable] = distances[usable] / steps[usable] ** 2

    return result


def subspace_change_signal_to_noise(distances, noises):
    """
    How far the module subspace moved between neighbouring checkpoints, in units of how far token
    noise alone moves it:

        signal_to_noise = d / (n_from + n_to)

    n is each checkpoint's module_subspace_noise, the mean distance between the full module
    subspace and a bootstrap replicate's; two independent estimates of the same subspace are
    about n_from + n_to apart (on quiet synthetic runs d / (n_from + n_to) was 1.1 to 1.2, so the
    floor is slightly low: read values below about 2 as noise).

    Why not the fidelity susceptibility chi = d / dt^2: on Pythia's schedule (log-spaced to step
    512, then every 1,000 steps) dt in log10(step + 1) falls from 0.3 to 0.003, so the constant
    noise in d becomes a raw chi 10,000 times larger at the end of training than at the start. On
    synthetic runs with Pythia's spacing the raw chi peaked on the last checkpoints in every run,
    with or without a planted reshuffle, and subtracting the noise floor before dividing by dt^2
    did not rescue it; this ratio found the reshuffle in 3 of 3 seeds (16 to 24 times the median,
    largest quiet value 1.8; research/round4/sim6). A jump in structure does not shrink with dt at
    all, so dividing it by dt^2 only measures the spacing.

    distances   (n - 1,) subspace distances between neighbouring checkpoints
    noises      (n,) module_subspace_noise of each checkpoint (NaN allowed)
    Returns (n - 1,) values.
    """
    distances = np.asarray(distances, dtype=np.float64)
    noises = np.asarray(noises, dtype=np.float64)

    floor = noises[:-1] + noises[1:]

    ratio = np.full(len(distances), np.nan)

    usable = np.isfinite(floor) & (floor > 0)
    ratio[usable] = distances[usable] / floor[usable]

    return ratio


def fidelity_susceptibility_series(run_folder, layer):
    """
    fidelity_susceptibility of the module subspace over a run, aligned with the checkpoints: the
    value at checkpoint i belongs to the transition (i - 1) -> i, and the first entry is NaN.
    """
    run_folder = Path(run_folder)

    steps = storage.finished_steps(run_folder)

    distances = subspace_change_series(run_folder, layer)[1:]

    values = fidelity_susceptibility(distances, training_time(steps))

    return np.concatenate([[np.nan], values])


def subspace_change_signal_to_noise_series(run_folder, layer):
    """
    subspace_change_signal_to_noise over a run, aligned with the checkpoints (first entry NaN).
    """
    run_folder = Path(run_folder)

    distances = subspace_change_series(run_folder, layer)[1:]

    noises = scalar_table(run_folder, layer)["module_subspace_noise"]

    ratio = subspace_change_signal_to_noise(distances, noises)

    return np.concatenate([[np.nan], ratio])


def paired_subspace_change_series(run_folder, layer, settings=None, number_of_replicates=12):
    """
    analysis.paired_subspace_change between neighbouring checkpoints over a run, for checkpoints
    whose block moments were saved (collection.save_block_moments). Returns (distance, paired
    standard deviation), each aligned with the checkpoints, first entry NaN, NaN where blocks are
    missing. Feed the standard deviation to change_points as the noise scale, or read
    (d - local median) / sd: on a shared probe this is far more sensitive than
    subspace_change_signal_to_noise (research/round4/sim7).
    """
    from nds.analysis import paired_subspace_change

    run_folder = Path(run_folder)

    steps = storage.finished_steps(run_folder)

    distances = np.full(len(steps), np.nan)
    spreads = np.full(len(steps), np.nan)

    previous = None
    for index, step in enumerate(steps):
        folder = storage.step_folder(run_folder, step)

        current = None
        if (folder / f"layer{layer}_block_moments.npz").exists():
            current = storage.load_block_moments(folder, layer)

        if previous is not None and current is not None:
            result = paired_subspace_change(previous, current, settings, number_of_replicates)
            distances[index] = result["distance"]
            spreads[index] = result["replicate_sd"]

        previous = current

    return distances, spreads


def crossvalidated_subspace_change_series(run_folder, layer, settings=None):
    """
    analysis.crossvalidated_subspace_change between neighbouring checkpoints over a run, where block
    moments were saved (D48). Returns (d_cv, plain d), aligned with the checkpoints (first entry and
    missing blocks NaN). d_cv near 0 means no change beyond noise; unlike plain d it does not
    shrink as the number of tokens grows, so its level means the same thing all through training.
    """
    from nds.analysis import crossvalidated_subspace_change

    run_folder = Path(run_folder)

    steps = storage.finished_steps(run_folder)

    crossed = np.full(len(steps), np.nan)
    plain = np.full(len(steps), np.nan)

    previous = None
    for index, step in enumerate(steps):
        folder = storage.step_folder(run_folder, step)

        current = None
        if (folder / f"layer{layer}_block_moments.npz").exists():
            current = storage.load_block_moments(folder, layer)

        if previous is not None and current is not None:
            result = crossvalidated_subspace_change(previous, current, settings)
            crossed[index] = result["d_cv"]
            plain[index] = result["plain"]

        previous = current

    return crossed, plain
