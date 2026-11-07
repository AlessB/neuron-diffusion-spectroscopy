"""
check_gurnee_pearson.py
=======================

Gurnee et al. (2024) StreamingPearsonComputer, ported to numpy from
/home/claude/refcode/wesg52__universal-neurons/correlations_fast.py
    lines 15-82   class StreamingPearsonComputer (raw sums: sum x, sum x^2, sum x y; float64 accumulators)
    lines 85-87   save_activation_hook stores activations as float16
    lines 43-50   cross products per batch in float32 (einsum), then added to the float64 accumulator
    lines 55-58   sum and sum of squares computed on the float16 tensor (output float16), then added
    lines 69-79   r = (Sxy - Sx Sy / n) / sqrt((Sxx - Sx^2/n)(Syy - Sy^2/n)), cast to float16
compared with nds.moments.NeuronMoments (Chan-Golub-LeVeque merge of batch-centred scatter):
    (a) add_batch          : float64 everywhere
    (b) add_batch_summary  : batch mean/scatter in float32 (what activations.summarise_on_device does
                             on the GPU), merged in float64
Reference: np.corrcoef in float64 on the full data, after the same float32 cast the model produces.
"""

import sys

import numpy as np

sys.path.insert(0, "/home/claude/pg/repo/neuron-diffusion-spectroscopy/src")
sys.path.insert(0, "/home/claude/pg/research/checks")

from nds.graphs import correlation_from_covariance  # noqa: E402
from nds.moments import NeuronMoments  # noqa: E402
from synthetic_activations import gelu_neurons  # noqa: E402


def torch_like_half_sum(values_half, axis):
    """torch.sum on a float16 tensor accumulates in float32 and returns float16."""
    return values_half.astype(np.float32).sum(axis=axis).astype(np.float16)


def gurnee_streaming_pearson(data, batch_tokens=16384, faithful_dtypes=True, half_activations_only=False):
    """data: (tokens, neurons). Same model on both sides (within-model correlation)."""
    neurons = data.shape[1]
    s1 = np.zeros(neurons, dtype=np.float64)
    s1_sq = np.zeros(neurons, dtype=np.float64)
    s12 = np.zeros((neurons, neurons), dtype=np.float64)
    n = 0
    for start in range(0, len(data), batch_tokens):
        batch = data[start:start + batch_tokens].T                 # (neurons, tokens), Gurnee layout
        if half_activations_only:                                  # fp16 storage, float64 sums
            b = batch.astype(np.float16).astype(np.float64)
            s12 += b @ b.T
            s1 += b.sum(axis=-1)
            s1_sq += (b ** 2).sum(axis=-1)
        elif faithful_dtypes:
            half = batch.astype(np.float16)                        # save_activation_hook
            f32 = half.astype(np.float32)
            with np.errstate(over="ignore"):
                s12 += (f32 @ f32.T).astype(np.float64)            # einsum in float32
                s1 += torch_like_half_sum(half, axis=-1)
                s1_sq += torch_like_half_sum((half ** 2).astype(np.float16), axis=-1)
        else:
            s12 += batch @ batch.T
            s1 += batch.sum(axis=-1)
            s1_sq += (batch ** 2).sum(axis=-1)
        n += batch.shape[1]
    numerator = s12 - np.outer(s1, s1) / n
    with np.errstate(invalid="ignore"):
        norm = (s1_sq - s1 ** 2 / n) ** 0.5
        corr = numerator / np.outer(norm, norm)
    return corr.astype(np.float16) if faithful_dtypes else corr


def ours(data, batch_tokens=512, gpu_float32=False):
    moments = NeuronMoments(data.shape[1])
    for start in range(0, len(data), batch_tokens):
        batch = data[start:start + batch_tokens]
        if gpu_float32:
            b32 = batch.astype(np.float32)
            mean = b32.mean(axis=0)
            centred = b32 - mean
            moments.add_batch_summary(len(b32), mean.astype(np.float64), (centred.T @ centred).astype(np.float64),
                                      (b32 > 0).sum(axis=0), np.abs(b32).max(axis=0))
        else:
            moments.add_batch(batch)
    return correlation_from_covariance(moments.covariance())


def max_offdiag_error(estimate, reference):
    diff = np.abs(np.asarray(estimate, dtype=np.float64) - reference)
    np.fill_diagonal(diff, 0.0)
    return float(np.nanmax(diff)), int(np.sum(~np.isfinite(np.asarray(estimate, dtype=np.float64))))


def run(name, data):
    data = data.astype(np.float32).astype(np.float64)               # model output is float32
    reference = np.corrcoef(data, rowvar=False)
    rows = [
        ("Gurnee, faithful dtypes (fp16 acts)", gurnee_streaming_pearson(data)),
        ("Gurnee, fp16 acts but float64 sums", gurnee_streaming_pearson(data, half_activations_only=True)),
        ("Gurnee formula, all float64", gurnee_streaming_pearson(data, faithful_dtypes=False)),
        ("nds add_batch (float64)", ours(data)),
        ("nds add_batch_summary (float32 batch)", ours(data, gpu_float32=True)),
    ]
    print(f"--- {name}: tokens={data.shape[0]} neurons={data.shape[1]}")
    for label, estimate in rows:
        err, nonfinite = max_offdiag_error(estimate, reference)
        print(f"  {label:40s} max|r - r_ref| = {err:.3e}   non-finite entries = {nonfinite}")


if __name__ == "__main__":
    base = gelu_neurons(number_of_tokens=51200, number_of_neurons=256, seed=11)
    run("GELU neurons (typical scale)", base)

    shifted = 50.0 + 0.05 * (base - base.mean(axis=0)) / base.std(axis=0)
    run("large mean / small spread (mean 50, std 0.05)", shifted)

    massive = base.copy()
    massive[:, 0] *= 300.0 / np.abs(massive[:, 0]).max()             # one neuron reaching |x| = 300
    run("one neuron with |x| up to 300 (fp16 x^2 overflows above 255.9)", massive)

    # memory of the accumulators (bytes), Pythia-70m (6 x 2048) and 160m (12 x 3072)
    print("--- accumulator memory")
    for label, layers, d_mlp in [("pythia-70m", 6, 2048), ("pythia-160m", 12, 3072), ("pythia-410m", 24, 4096)]:
        gurnee = (layers * d_mlp) ** 2 * 8
        nds_blocks = layers * 8 * d_mlp ** 2 * 8
        print(f"  {label:12s} Gurnee all-layer cross matrix {gurnee / 1e9:6.2f} GB | "
              f"nds per-layer scatter x 8 blocks x layers {nds_blocks / 1e9:6.2f} GB")
