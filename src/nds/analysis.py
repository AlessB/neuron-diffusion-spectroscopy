"""
analysis.py
===========

Everything we measure for one (checkpoint, layer), from its neuron moments. Pure numpy/scipy:
this runs anywhere, including on results copied off the GPU machine.

For one layer it returns:

    graph         number of live neurons, dead fraction, massive-activation neurons
    spectrum      Laplacian eigenvalues (first 50), eigenvectors, module count, DSE(t), IPR
    baselines     RankMe (two versions), alpha-ReQ, matrix entropy        (covariance side)
    rotation      the same spectrum numbers after random rotations of the live neurons (5, D36)
    bootstrap     the same spectrum numbers on resampled blocks of probe text (noise band)
    null          the same spectrum numbers for independent neurons at the same effective number of
                  tokens (the shuffle null, drawn directly as a Wishart matrix; noise.py, D36)

The same settings dict (configs/*.yaml) drives every call, so a run can be repeated exactly.
"""

import numpy as np

from nds.baselines import baseline_summary
from nds.controls import rotate_live_covariance
from nds.flow import restrict_to_common_neurons
from nds.graphs import find_dead_neurons
from nds.graphs import correlation_distance_squared
from nds.graphs import fixed_bandwidth_gaussian_kernel
from nds.graphs import neuron_graph_from_covariance
from nds.modules import conductances
from nds.modules import spectral_partition
from nds.modules import upsilon
from nds.noise import wishart_null_correlation
from nds.spectra import diffusion_spectral_entropy
from nds.spectra import diffusion_spectrum
from nds.spectra import eigengap_module_count
from nds.spectra import chordal_subspace_distance
from nds.spectra import crossvalidated_projector_distance
from nds.spectra import davis_kahan_ratio
from nds.spectra import leading_eigenvectors
from nds.spectra import operator_norm
from nds.spectra import spectrum_summary
from nds.spectra import symmetric_diffusion_operator


DEFAULT_SETTINGS = {
    "neighbour_rank": 10,
    "kernel_type": "self_tuning",
    "decay": 40.0,
    "threshold": 0.0,
    "anisotropy": 0.0,
    "signed": True,
    "keep_self_loops": False,
    "minimum_variance": 1e-10,
    "minimum_firing_rate": 1e-4,
    "number_of_modes": 50,
    "maximum_modules": 20,
    "diffusion_times": [1, 2, 4, 8, 16, 32, 64],
    "ipr_modes": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "massive_activation_factor": 100.0,
    "exclude_massive_neurons": False,
    "number_of_rotations": 5,
    "number_of_null_draws": 1,
    "number_of_bootstrap_replicates": 5,
    "random_seed": 0,
    "fixed_bandwidth_dse": True,
    "module_partition": True,
}


def merged_settings(settings=None):
    """
    DEFAULT_SETTINGS with any given values on top.
    """
    combined = dict(DEFAULT_SETTINGS)

    if settings is not None:
        combined.update(settings)

    return combined


def find_massive_neurons(max_abs_activation, factor=100.0):
    """
    Neurons whose largest |activation| is more than `factor` times the median neuron's.

    These "massive activation" outliers (Sun et al. 2024) can dominate a covariance. We record
    them always and can exclude them as a control.

    Credit: M. Sun, X. Chen, J. Z. Kolter, Z. Liu (2024). "Massive activations in large language
    models." COLM 2024.
    """
    max_abs_activation = np.asarray(max_abs_activation)

    median_value = np.median(max_abs_activation)

    if median_value <= 0:
        return np.zeros(len(max_abs_activation), dtype=bool)

    return max_abs_activation > factor * median_value


def spectrum_numbers(covariance, firing_rate, settings):
    """
    Build the neuron graph from a covariance and summarise its spectrum.
    """
    graph = neuron_graph_from_covariance(
        covariance,
        firing_rate=firing_rate,
        neighbour_rank=settings["neighbour_rank"],
        kernel_type=settings["kernel_type"],
        decay=settings["decay"],
        signed=settings["signed"],
        keep_self_loops=settings["keep_self_loops"],
        threshold=settings["threshold"],
        anisotropy=settings["anisotropy"],
        minimum_variance=settings["minimum_variance"],
        minimum_firing_rate=settings["minimum_firing_rate"],
    )

    summary = spectrum_summary(
        graph["kernel"],
        number_of_modes=settings["number_of_modes"],
        maximum_modules=settings["maximum_modules"],
        diffusion_times=settings["diffusion_times"],
        ipr_modes=settings["ipr_modes"],
    )

    summary["live_indices"] = graph["live_indices"]
    summary["dead_fraction"] = graph["dead_fraction"]

    # Which neuron belongs to which module, and how well separated they are (Peng-Sun-Zanetti)
    if settings.get("module_partition", False):
        number_of_modules = summary["module_count"]

        labels = spectral_partition(summary["symmetric_vectors"], number_of_modules)

        module_conductances = conductances(graph["kernel"], labels)

        summary["module_labels"] = labels

        # how strongly each neuron belongs to the module subspace: its squared row norm in modes
        # 0 .. k*-1, times m / k* (1 = an even share). Neurons in no module sit near 1 / k* (only the
        # near-constant mode 0 reaches them); k-means still gives them a label, which flickers
        # between checkpoints and dominated 1 - ARI on synthetic runs (research/round5, 23 Oct).
        # partition_stability_series compares only neurons above 0.5 at both checkpoints.
        coordinates = summary["symmetric_vectors"][:, :max(int(number_of_modules), 2)]
        summary["module_attachment"] = np.sum(coordinates ** 2, axis=1) * coordinates.shape[0] / coordinates.shape[1]
        summary["module_sizes"] = np.bincount(labels, minlength=number_of_modules)
        summary["worst_conductance"] = max(module_conductances.values())
        summary["upsilon"] = upsilon(summary["laplacian_eigenvalues"], graph["kernel"], labels)

    # Robustness check: DSE on a single-bandwidth Gaussian kernel, closer to the published DSE code
    if settings.get("fixed_bandwidth_dse", False):
        squared_distances = correlation_distance_squared(graph["correlation"], signed=settings["signed"])
        fixed_kernel = fixed_bandwidth_gaussian_kernel(squared_distances)
        fixed_spectrum = diffusion_spectrum(fixed_kernel)

        fixed_dse = {}
        for diffusion_time in settings["diffusion_times"]:
            fixed_dse[int(diffusion_time)] = diffusion_spectral_entropy(
                fixed_spectrum["diffusion_eigenvalues"],
                diffusion_time=diffusion_time,
            )

        summary["dse_fixed_bandwidth"] = fixed_dse

    return summary


def diffusion_operator_for(covariance, firing_rate, settings):
    """
    The symmetric diffusion operator of the neuron graph and the live neuron indices it covers
    (the same graph spectrum_numbers builds). Used for the Davis-Kahan trust score.
    """
    graph = neuron_graph_from_covariance(
        covariance,
        firing_rate=firing_rate,
        neighbour_rank=settings["neighbour_rank"],
        kernel_type=settings["kernel_type"],
        decay=settings["decay"],
        signed=settings["signed"],
        keep_self_loops=settings["keep_self_loops"],
        threshold=settings["threshold"],
        anisotropy=settings["anisotropy"],
        minimum_variance=settings["minimum_variance"],
        minimum_firing_rate=settings["minimum_firing_rate"],
    )

    operator, _ = symmetric_diffusion_operator(graph["kernel"])

    return operator, graph["live_indices"]


def module_subspace_of(moments, settings, module_count=None):
    """
    The module subspace (modes 0 .. k*-1 of the symmetric diffusion operator) of one merged
    NeuronMoments, the way analyse_layer builds the graph (massive neurons excluded when the
    settings say so). Returns (vectors, neuron ids of the rows, module count). With module_count
    given, that many modes are taken instead of the eigengap count (bootstrap replicates keep the
    full-data count).
    """
    covariance = moments.covariance()
    firing_rate = moments.firing_rate()

    kept_indices = np.arange(moments.number_of_neurons)

    if settings["exclude_massive_neurons"]:
        is_massive = find_massive_neurons(
            moments.max_abs_activation,
            factor=settings["massive_activation_factor"],
        )
        if np.any(is_massive):
            kept_indices = np.flatnonzero(~is_massive)
            covariance = covariance[np.ix_(kept_indices, kept_indices)]
            firing_rate = firing_rate[kept_indices]

    operator, live = diffusion_operator_for(covariance, firing_rate, settings)

    if module_count is None:
        eigenvalues = np.linalg.eigvalsh(operator)[::-1]
        laplacian_eigenvalues = 1.0 - eigenvalues
        module_count, _, _ = eigengap_module_count(laplacian_eigenvalues)
    module_count = max(int(module_count), 2)

    vectors = leading_eigenvectors(operator, module_count)

    return vectors, kept_indices[live], module_count


def paired_subspace_change(blocked_from, blocked_to, settings=None, number_of_replicates=12, random_seed=0):
    """
    Distance between the module subspaces of two checkpoints, with a paired bootstrap of it.

    Both checkpoints see the same probe text, and block i holds the same sequences at both
    (activations.py assigns sequence j to block j % number_of_blocks). Resampling the same blocks
    for both keeps that shared-token correlation, as common random numbers do in simulation
    (variance reduction for comparing two systems driven by the same input; Kahn & Marshall 1953;
    the paired bootstrap, Efron & Tibshirani 1993). On synthetic runs with a shared probe, quiet transitions sat 50
    times below the unpaired noise floor (each checkpoint's own module_subspace_noise), so the
    unpaired floor called a single neuron changing module noise while the paired z-score was 30 to
    45 (research/round4/sim7).

    blocked_from, blocked_to   moments.BlockedNeuronMoments of the two checkpoints (the saved
                               block moments, storage.load_block_moments)
    Returns a dict: distance (full data), replicate_mean, replicate_sd, replicates (the list).
    """
    settings = merged_settings(settings)

    random_generator = np.random.default_rng(random_seed)

    vectors_from, ids_from, count_from = module_subspace_of(blocked_from.merged(), settings)
    vectors_to, ids_to, count_to = module_subspace_of(blocked_to.merged(), settings)

    restricted_from, restricted_to, _, _ = restrict_to_common_neurons(vectors_from, ids_from, vectors_to, ids_to)
    distance = chordal_subspace_distance(restricted_from, restricted_to)

    number_of_blocks = blocked_from.number_of_blocks

    replicates = []
    for _ in range(number_of_replicates):
        chosen = random_generator.integers(low=0, high=number_of_blocks, size=number_of_blocks)

        replicate_from, replicate_ids_from, _ = module_subspace_of(blocked_from.merged(chosen), settings, count_from)
        replicate_to, replicate_ids_to, _ = module_subspace_of(blocked_to.merged(chosen), settings, count_to)

        restricted_from, restricted_to, _, _ = restrict_to_common_neurons(
            replicate_from,
            replicate_ids_from,
            replicate_to,
            replicate_ids_to,
        )
        replicates.append(chordal_subspace_distance(restricted_from, restricted_to))

    replicates = np.asarray(replicates, dtype=np.float64)

    spread = float("nan")
    if len(replicates) > 1:
        spread = float(np.std(replicates, ddof=1))

    replicate_mean = float("nan")
    if len(replicates) > 0:
        replicate_mean = float(np.mean(replicates))

    return {
        "distance": float(distance),
        "replicate_mean": replicate_mean,
        "replicate_sd": spread,
        "replicates": replicates.tolist(),
    }


def crossvalidated_subspace_change(blocked_from, blocked_to, settings=None):
    """
    The cross-validated distance d_cv between the module subspaces of two checkpoints, and its
    per-neuron parts c_i (spectra.crossvalidated_projector_distance), from the saved block moments.

    The probe text is split once into two halves by block (even and odd blocks); the SAME halves are
    used at both checkpoints, so each half is still paired between checkpoints (common random
    numbers, research/round4 S7) and the product of the two halves removes the noise bias that is
    left (research/round5 S9). k is fixed for the pair (the larger of the two module counts), so a
    change of k* does not break the pairing. Neurons: those in all four half-graphs.

    Returns a dict: d_cv, plain (the biased full-data distance, for comparison), module_count used,
    neuron_ids and per_neuron (c_i, in the order of neuron_ids).
    """
    settings = merged_settings(settings)

    _, _, count_from = module_subspace_of(blocked_from.merged(), settings)
    _, _, count_to = module_subspace_of(blocked_to.merged(), settings)
    count = max(count_from, count_to)

    full_from, ids_full_from, _ = module_subspace_of(blocked_from.merged(), settings, count)
    full_to, ids_full_to, _ = module_subspace_of(blocked_to.merged(), settings, count)
    restricted_from, restricted_to, _, _ = restrict_to_common_neurons(full_from, ids_full_from, full_to, ids_full_to)
    plain = chordal_subspace_distance(restricted_from, restricted_to)

    halves = [list(range(0, blocked_from.number_of_blocks, 2)), list(range(1, blocked_from.number_of_blocks, 2))]

    estimates = []
    for half in halves:
        vectors_from, ids_from, _ = module_subspace_of(blocked_from.merged(half), settings, count)
        vectors_to, ids_to, _ = module_subspace_of(blocked_to.merged(half), settings, count)
        estimates.append((vectors_from, ids_from))
        estimates.append((vectors_to, ids_to))

    common = estimates[0][1]
    for _, ids in estimates[1:]:
        common = np.intersect1d(common, ids)

    aligned = []
    for vectors, ids in estimates:
        position = {int(neuron): row for row, neuron in enumerate(ids)}
        rows = [position[int(neuron)] for neuron in common]
        aligned.append(vectors[rows])

    d_cv, per_neuron = crossvalidated_projector_distance(aligned[0], aligned[1], aligned[2], aligned[3])

    return {
        "d_cv": d_cv,
        "plain": float(plain),
        "module_count": int(count),
        "neuron_ids": common,
        "per_neuron": per_neuron,
    }


def short_summary(summary):
    """
    The scalar part of a spectrum summary (what we keep for each bootstrap / rotation replicate).
    """
    short = {
        "module_count": summary["module_count"],
        "gap_ratio": summary["gap_ratio"],
        "median_localization": summary["median_localization"],
        "slow_subspace_localization": summary["slow_subspace_localization"],
        "first_ten_subspace_localization": summary["first_ten_subspace_localization"],
        "slow_aligned_localization": summary.get("slow_aligned_localization"),
        "first_ten_aligned_localization": summary.get("first_ten_aligned_localization"),
        "module_aligned_localization": summary.get("module_aligned_localization"),
        "module_aligned_support": summary.get("module_aligned_support"),
        "slow_spacing_ratio": summary.get("slow_spacing_ratio"),
        "bulk_spacing_ratio": summary.get("bulk_spacing_ratio"),
        "heat_capacity_peak": summary.get("heat_capacity_peak"),
        "heat_capacity_peak_scale": summary.get("heat_capacity_peak_scale"),
        "degree_floor_localization": summary.get("degree_floor_localization"),
        "median_localization_whole_spectrum": summary.get("median_localization_whole_spectrum"),
        "slow_mode_degree_ratio": summary.get("slow_mode_degree_ratio"),
        "laplacian_eigenvalues": summary["laplacian_eigenvalues"][:20].copy(),
        "dse": dict(summary["dse"]),
        "ipr": dict(summary["ipr"]),
    }

    return short


def analyse_layer(blocked_moments, settings=None, effective_tokens=None):
    """
    All measurements for one layer of one checkpoint.

    blocked_moments   a moments.BlockedNeuronMoments with the probe text's statistics
    effective_tokens  the effective number of independent tokens (noise.effective_token_count, from
                      the lag-one correlation measured during collection). Needed for the null;
                      None skips it.
    """
    settings = merged_settings(settings)

    random_generator = np.random.default_rng(settings["random_seed"])

    full = blocked_moments.merged()
    covariance = full.covariance()
    firing_rate = full.firing_rate()

    is_massive = find_massive_neurons(
        full.max_abs_activation,
        factor=settings["massive_activation_factor"],
    )

    # `kept_indices` maps rows of the covariance we analyse back to the model's neuron numbers
    kept_indices = np.arange(full.number_of_neurons)

    if settings["exclude_massive_neurons"] and np.any(is_massive):
        kept_indices = np.flatnonzero(~is_massive)
        covariance = covariance[np.ix_(kept_indices, kept_indices)]
        firing_rate = firing_rate[kept_indices]

    # --- the neuron graph and its spectrum -------------------------------------------------
    spectrum = spectrum_numbers(covariance, firing_rate, settings)

    # neuron numbers in the model (needed to follow modes across checkpoints, flow.py)
    spectrum["live_neuron_ids"] = kept_indices[spectrum["live_indices"]]

    # --- covariance baselines on the same (live) neurons ------------------------------------
    live = spectrum["live_indices"]
    live_covariance = covariance[np.ix_(live, live)]

    baselines = baseline_summary(live_covariance)

    # --- rotation control -------------------------------------------------------------------
    is_dead = find_dead_neurons(
        covariance,
        firing_rate=firing_rate,
        minimum_variance=settings["minimum_variance"],
        minimum_firing_rate=settings["minimum_firing_rate"],
    )

    # replicates only need the main numbers; skip the extra robustness eigendecomposition there
    replicate_settings = dict(settings)
    replicate_settings["fixed_bandwidth_dse"] = False
    replicate_settings["module_partition"] = False

    rotations = []
    for _ in range(settings["number_of_rotations"]):
        rotated_covariance = rotate_live_covariance(covariance, is_dead, random_generator)

        rotated_spectrum = spectrum_numbers(rotated_covariance, None, replicate_settings)
        rotated_baselines = baseline_summary(rotated_covariance)

        rotations.append(
            {
                "spectrum": short_summary(rotated_spectrum),
                "baselines": rotated_baselines,
            }
        )

    # --- token bootstrap --------------------------------------------------------------------
    # each replicate also gives one perturbation of the diffusion operator, for the Davis-Kahan
    # trust score of the module subspace (spectra.davis_kahan_ratio)
    # and one draw of how far the module subspace moves from token noise alone: the noise floor of
    # the subspace change between checkpoints (timeline.subspace_change_signal_to_noise)
    full_operator, full_live = diffusion_operator_for(covariance, firing_rate, settings)
    perturbation_norms = []
    subspace_noise = []
    module_dimension = max(int(spectrum["module_count"]), 2)
    full_module_vectors = spectrum["symmetric_vectors"][:, :module_dimension]

    bootstrap = []
    for _ in range(settings["number_of_bootstrap_replicates"]):
        replicate = blocked_moments.bootstrap_replicate(random_generator)

        replicate_covariance = replicate.covariance()
        replicate_firing_rate = replicate.firing_rate()

        replicate_covariance = replicate_covariance[np.ix_(kept_indices, kept_indices)]
        replicate_firing_rate = replicate_firing_rate[kept_indices]

        replicate_spectrum = spectrum_numbers(replicate_covariance, replicate_firing_rate, replicate_settings)

        bootstrap.append(short_summary(replicate_spectrum))

        replicate_operator, replicate_live = diffusion_operator_for(replicate_covariance, replicate_firing_rate, settings)
        if len(replicate_live) == len(full_live) and np.all(replicate_live == full_live):
            perturbation_norms.append(operator_norm(replicate_operator - full_operator))

            replicate_module_vectors = leading_eigenvectors(replicate_operator, module_dimension)
            subspace_noise.append(chordal_subspace_distance(full_module_vectors, replicate_module_vectors))

    spectrum["module_subspace_noise"] = float("nan")
    if len(subspace_noise) > 0:
        spectrum["module_subspace_noise"] = float(np.mean(subspace_noise))

    spectrum["davis_kahan_ratio"] = davis_kahan_ratio(
        perturbation_norms,
        spectrum["laplacian_eigenvalues"],
        spectrum["module_count"],
    )

    # --- null: independent neurons at the same noise level ---------------------------------
    null = []
    number_of_live = len(spectrum["live_indices"])
    if effective_tokens is not None and effective_tokens > number_of_live + 1:
        for _ in range(settings["number_of_null_draws"]):
            null_correlation = wishart_null_correlation(number_of_live, effective_tokens, random_generator)

            null_spectrum = spectrum_numbers(null_correlation, None, replicate_settings)

            null.append(short_summary(null_spectrum))

    result = {
        "settings": settings,
        "effective_tokens": effective_tokens,
        "null": null,
        "token_count": full.count,
        "number_of_neurons": full.number_of_neurons,
        "massive_neurons": np.flatnonzero(is_massive),
        "spectrum": spectrum,
        "baselines": baselines,
        "rotations": rotations,
        "bootstrap": bootstrap,
    }

    return result


def layer_summary(result):
    """
    The JSON-friendly part of an analyse_layer result: every scalar, nothing large.
    Eigenvectors and module labels go to the layer's .npz file instead (storage.save_layer_spectrum).

    One function used by collect.py, reanalyse.py, dry_run.py and the tests, so every run writes
    exactly the same fields.
    """
    spectrum = result["spectrum"]

    spectrum_scalars = {
        "module_count": spectrum["module_count"],
        "gap_ratio": spectrum["gap_ratio"],
        "laplacian_eigenvalues": spectrum["laplacian_eigenvalues"][:20],
        "dse": spectrum["dse"],
        "dse_fixed_bandwidth": spectrum.get("dse_fixed_bandwidth"),
        "ipr": spectrum["ipr"],
        "localization": spectrum["localization"],
        "slow_subspace_localization": spectrum["slow_subspace_localization"],
        "random_subspace_baseline": spectrum["random_subspace_baseline"],
        "first_ten_subspace_localization": spectrum["first_ten_subspace_localization"],
        "slow_aligned_localization": spectrum.get("slow_aligned_localization"),
        "first_ten_aligned_localization": spectrum.get("first_ten_aligned_localization"),
        "module_aligned_localization": spectrum.get("module_aligned_localization"),
        "module_aligned_support": spectrum.get("module_aligned_support"),
        "slow_spacing_ratio": spectrum.get("slow_spacing_ratio"),
        "bulk_spacing_ratio": spectrum.get("bulk_spacing_ratio"),
        "heat_capacity_peak": spectrum.get("heat_capacity_peak"),
        "heat_capacity_peak_scale": spectrum.get("heat_capacity_peak_scale"),
        "degree_floor_localization": spectrum.get("degree_floor_localization"),
        "median_localization_whole_spectrum": spectrum.get("median_localization_whole_spectrum"),
        "slow_mode_degree_ratio": spectrum.get("slow_mode_degree_ratio"),
        "davis_kahan_ratio": spectrum.get("davis_kahan_ratio"),
        "module_subspace_noise": spectrum.get("module_subspace_noise"),
        "upsilon": spectrum.get("upsilon"),
        "worst_conductance": spectrum.get("worst_conductance"),
        "module_sizes": spectrum.get("module_sizes"),
        "dead_fraction": spectrum["dead_fraction"],
        "number_of_live_neurons": len(spectrum["live_neuron_ids"]),
    }

    summary = {
        "spectrum": spectrum_scalars,
        "baselines": result["baselines"],
        "rotations": result["rotations"],
        "bootstrap": result["bootstrap"],
        "null": result.get("null", []),
        "effective_tokens": result.get("effective_tokens"),
        "massive_neurons": result["massive_neurons"],
        "token_count": result["token_count"],
    }

    return summary
