"""
collect.py
==========

For every checkpoint in a config: load the model, run the probe text, keep neuron statistics,
analyse every layer, save results. Resumes where it stopped.

    python scripts/collect.py --config configs/pythia70m-standard_week1.yaml

Needs PyTorch + transformers (see docs/gpu-runbook.md). Everything after the model run is numpy.
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nds import checkpoints
from nds import storage
from nds.activations import collect_neuron_moments
from nds.activations import load_pythia
from nds.analysis import analyse_layer
from nds.analysis import layer_summary
from nds.baselines import htsr_alpha
from nds.baselines import knowledge_entropy
from nds.induction import induction_scores
from nds.neuron_stats import layer_neuron_statistics
from nds.neuron_stats import vocabulary_alignment_moments
from nds.induction import repeated_random_tokens
from nds.noise import effective_token_count
from nds.probe_text import load_probe_set
from nds.token_sample import choose_token_positions
from nds.token_sample import token_sample_baselines


def resolve_steps(step_setting):
    """
    The config may name a preset ("all", "week_one") or give an explicit list of steps.
    """
    if step_setting == "all":
        return checkpoints.all_pythia_steps()

    if step_setting == "week_one":
        return checkpoints.week_one_steps()

    return [int(step) for step in step_setting]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--only-step", type=int, default=None, help="run a single step (testing)")
    arguments = parser.parse_args()

    with open(arguments.config, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    run_folder = storage.run_folder_for(config["run_name"])
    run_folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(arguments.config, run_folder / "config.yaml")

    import torch

    device = config["collection"].get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"

    tokens, manifest = load_probe_set(
        config["probe"]["tokens"],
        config["probe"]["manifest"],
    )
    print(f"probe set {tokens.shape}, sha256 {manifest['sha256'][:12]}...")

    induction_settings = config.get("induction", {})
    induction_sequences = repeated_random_tokens(
        number_of_sequences=induction_settings.get("number_of_sequences", 64),
        repeat_length=induction_settings.get("repeat_length", 128),
        random_generator=np.random.default_rng(induction_settings.get("seed", 0)),
    )

    # the fixed token sample (D31): the same positions at every checkpoint, chosen once from the seed
    skip_first = config["collection"].get("skip_first_positions", 0)
    sample_size = config["collection"].get("token_sample_size", 0)
    token_positions = None
    if sample_size:
        token_positions = choose_token_positions(
            tokens.shape[0],
            tokens.shape[1],
            sample_size=sample_size,
            skip_first_positions=skip_first,
            seed=config["collection"].get("token_sample_seed", 0),
        )

    steps = resolve_steps(config["steps"])
    if arguments.only_step is not None:
        steps = [arguments.only_step]

    already_done = set(storage.finished_steps(run_folder))

    for step in steps:
        if step in already_done:
            print(f"step {step}: already done, skipping")
            continue

        step_start = time.time()
        folder = storage.step_folder(run_folder, step)

        model, _ = load_pythia(
            config["model_name"],
            step,
            device=device,
            dtype_name=config["collection"].get("dtype", "float32"),
            cache_dir=config["collection"].get("cache_dir"),
        )

        collected = collect_neuron_moments(
            model,
            tokens,
            batch_size=config["collection"].get("batch_size", 16),
            number_of_blocks=config["collection"].get("number_of_blocks", 8),
            layers=config["collection"].get("layers"),
            skip_first_positions=skip_first,
            compute_prompt_entropy=config["collection"].get("prompt_entropy", True),
            token_positions=token_positions,
        )

        induction = induction_scores(model, induction_sequences)

        # weight statistics need the model; Gurnee et al.'s vocabulary moments of cos(W_U, w_out)
        vocabulary_moments = {}
        weight_alphas = {}
        if config["collection"].get("weight_statistics", True):
            unembedding = model.embed_out.weight.detach().float().cpu().numpy()
            for layer in collected["moments"].keys():
                output_weights = model.gpt_neox.layers[layer].mlp.dense_4h_to_h.weight.detach().float().cpu().numpy()
                vocabulary_moments[layer] = vocabulary_alignment_moments(unembedding, output_weights)

                # WeightWatcher-style alpha of the MLP weights (rotation-invariant weight baseline)
                input_weights = model.gpt_neox.layers[layer].mlp.dense_h_to_4h.weight.detach().float().cpu().numpy()
                alpha_in, _, distance_in = htsr_alpha(input_weights)
                alpha_out, _, distance_out = htsr_alpha(output_weights)
                weight_alphas[layer] = {
                    "htsr_alpha_w_in": alpha_in,
                    "htsr_ks_w_in": distance_in,
                    "htsr_alpha_w_out": alpha_out,
                    "htsr_ks_w_out": distance_out,
                }

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

        layer_results = {}

        tokens_per_layer_statistics = tokens.shape[0] * (tokens.shape[1] - skip_first)

        for layer, blocked_moments in collected["moments"].items():
            # the noise level of this layer's correlations: effective independent tokens (noise.py)
            lag_one = collected["lag_one_correlation"].get(layer)
            effective_tokens = None
            if lag_one is not None and np.isfinite(lag_one):
                effective_tokens = effective_token_count(tokens_per_layer_statistics, lag_one)

            result = analyse_layer(
                blocked_moments,
                settings=config.get("analysis"),
                effective_tokens=effective_tokens,
            )

            # the covariance lets us re-analyse later without the model; it grows as neurons^2,
            # so big models can switch it off (configs: collection.save_covariance)
            if config["collection"].get("save_covariance", True):
                storage.save_layer_moments(folder, layer, blocked_moments.merged())
            storage.save_layer_spectrum(folder, layer, result["spectrum"])

            # every bootstrap block, so the bootstrap can be redone later (D48, proposed)
            if config["collection"].get("save_block_moments", False):
                storage.save_block_moments(folder, layer, blocked_moments)

            layer_results[layer] = layer_summary(result)
            layer_results[layer]["lag_one_correlation"] = lag_one

            # token sample: saved, and the token-side baselines computed from it (D31, D34, D35)
            if layer in collected["token_sample"]:
                sample = collected["token_sample"][layer]
                storage.save_token_sample(folder, layer, sample, token_positions[0], token_positions[1])
                layer_results[layer]["baselines_token_sample"] = token_sample_baselines(sample)

            # neuron-family statistics (Gurnee et al.) and the position-0 spike (neuron_stats.py)
            merged = blocked_moments.merged()
            vocabulary_skew = None
            vocabulary_kurtosis = None
            if layer in vocabulary_moments:
                vocabulary_skew, vocabulary_kurtosis = vocabulary_moments[layer]
            neuron_arrays, neuron_scalars = layer_neuron_statistics(
                merged.mean,
                np.diag(merged.covariance()),
                collected["raw_moments"][layer],
                position_zero=collected["position_zero"][layer],
                max_absolute_elsewhere=merged.max_abs_activation,
                vocabulary_skew=vocabulary_skew,
                vocabulary_kurtosis=vocabulary_kurtosis,
            )
            if layer in weight_alphas:
                neuron_scalars.update(weight_alphas[layer])
            storage.save_neuron_statistics(folder, layer, neuron_arrays)
            layer_results[layer]["neuron_statistics"] = neuron_scalars

            # Kim et al. knowledge entropy over the whole probe text (needs sums of |x|)
            if layer in collected["mean_absolute_activation"]:
                mean_absolute = collected["mean_absolute_activation"][layer]
                layer_results[layer]["baselines_knowledge_entropy"] = knowledge_entropy(mean_absolute)

            # Skean et al. prompt entropy is measured during collection (it needs each prompt's
            # activations), so it is added here next to the covariance baselines
            if layer in collected["prompt_entropy"]:
                layer_results[layer]["baselines_prompt_entropy"] = collected["prompt_entropy"][layer]

        summary = {
            "model_name": config["model_name"],
            "step": step,
            "tokens_seen": checkpoints.tokens_seen(step),
            "probe_sha256": manifest["sha256"],
            "probe_loss": collected["probe_loss"],
            "induction": induction,
            "layers": layer_results,
            "seconds_model": collected["elapsed_seconds"],
            "seconds_total": time.time() - step_start,
            "device": device,
        }

        storage.save_summary(folder, summary)

        if config["collection"].get("delete_checkpoint_after", False):
            delete_cached_revision(config["model_name"], step, config["collection"].get("cache_dir"))

        print(
            f"step {step}: loss {collected['probe_loss']:.3f}, "
            f"induction drop {induction['induction_loss_drop']:.2f}, "
            f"{summary['seconds_total']:.0f} s"
        )


def delete_cached_revision(model_name, step, cache_dir):
    """
    Free disk on a rented machine: remove this checkpoint's files from the Hugging Face cache.
    """
    from huggingface_hub import scan_cache_dir

    cache_info = scan_cache_dir(cache_dir)

    for repository in cache_info.repos:
        if repository.repo_id != f"EleutherAI/{model_name}":
            continue

        for revision in repository.revisions:
            if f"step{int(step)}" in revision.refs:
                strategy = cache_info.delete_revisions(revision.commit_hash)
                strategy.execute()


if __name__ == "__main__":
    main()
