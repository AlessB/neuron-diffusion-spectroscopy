"""
S3. A trust score for the module subspace from matrix perturbation theory.

Numerical analysis: if A is perturbed to A + E, the subspace of its top k eigenvectors moves by an
angle theta with sin(theta) <= 2 ||E|| / delta, where delta is the gap between the k-th and
(k+1)-th eigenvalues (Davis & Kahan 1970; the version without an eigenvalue separation condition
is Yu, Wang & Samworth 2015). For us A is the symmetric diffusion operator, E the difference
between a bootstrap replicate's operator and the full one, and delta the eigengap that ends the
module subspace. So

    r = ||E||_2 / delta

is computable at every checkpoint from the bootstrap we already run. r << 1: the module subspace
(and everything read from it: count, varimax basis, localisation, lineage) is pinned down by the
data. r near or above 1: it is not, whatever the readouts say.

Test: one layer, 256 neurons, 8 modules of 24 plus 64 free neurons, module strength from 0.05 to
0.5, 4,000 tokens in 8 blocks, 8 bootstrap replicates. Reported per strength: r, the bound
2r (clipped at 1), the actual subspace distance between replicate and full, and the spread of the
module-aligned localisation across replicates.

    python research/round4/sim3_davis_kahan.py
"""

import numpy as np

from sims_common import MODULE_NEURONS
from sims_common import MODULES
from sims_common import NEURONS

from nds.analysis import merged_settings
from nds.graphs import neuron_graph_from_covariance
from nds.moments import BlockedNeuronMoments
from nds.spectra import aligned_localization
from nds.spectra import diffusion_spectrum
from nds.spectra import eigengap_module_count
from nds.spectra import symmetric_diffusion_operator
from nds.timeline import subspace_distance

SETTINGS = merged_settings({"minimum_firing_rate": 0.0})


def operator_and_subspace(covariance, module_count=None):
    graph = neuron_graph_from_covariance(covariance, minimum_firing_rate=0.0)
    operator, _ = symmetric_diffusion_operator(graph["kernel"])
    spectrum = diffusion_spectrum(graph["kernel"])
    if module_count is None:
        module_count, _, _ = eigengap_module_count(spectrum["laplacian_eigenvalues"])
    module_count = max(int(module_count), 2)
    values = spectrum["diffusion_eigenvalues"]
    gap = values[module_count - 1] - values[module_count]
    vectors = spectrum["symmetric_vectors"][:, :module_count]
    return operator, vectors, gap, module_count


def main():
    print(f"{'strength':>8s} {'k*':>3s} {'gap':>7s} {'||E||':>7s} {'r':>6s} {'bound 2r':>9s} {'actual dist':>12s} {'loc':>6s} {'loc spread':>11s}")
    for strength in [0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5]:
        rows = []
        for seed in range(3):
            generator = np.random.default_rng(300 + seed)
            labels = np.full(NEURONS, -1)
            labels[:MODULE_NEURONS] = np.repeat(np.arange(MODULES), MODULE_NEURONS // MODULES)
            latents = generator.standard_normal((4000, MODULES))
            values = np.sqrt(1 - strength) * generator.standard_normal((4000, NEURONS))
            inside = labels >= 0
            values[:, inside] += np.sqrt(strength) * latents[:, labels[inside]]
            values[:, ~inside] += np.sqrt(strength) * generator.standard_normal((4000, int(np.sum(~inside))))

            blocked = BlockedNeuronMoments(NEURONS, number_of_blocks=8)
            for block in range(8):
                blocked.add_batch(values[block * 500:(block + 1) * 500], block)

            full_operator, full_vectors, gap, module_count = operator_and_subspace(blocked.merged().covariance())
            full_loc = aligned_localization(full_vectors)

            norms, distances, locs = [], [], []
            for _ in range(8):
                replicate = blocked.bootstrap_replicate(generator).covariance()
                operator, vectors, _, _ = operator_and_subspace(replicate, module_count)
                norms.append(np.linalg.norm(operator - full_operator, 2))
                distances.append(subspace_distance(full_vectors, vectors))
                locs.append(aligned_localization(vectors))
            rows.append((module_count, gap, np.mean(norms), np.mean(distances), full_loc, np.ptp(locs)))
        rows = np.asarray(rows, dtype=float)
        k, gap, norm, distance, loc, spread = rows.mean(axis=0)
        r = norm / max(gap, 1e-12)
        bound = min(2 * r, 1.0) ** 2
        print(f"{strength:8.2f} {k:3.0f} {gap:7.4f} {norm:7.4f} {r:6.2f} {bound:9.3f} {distance:12.4f} {loc:6.2f} {spread:11.3f}")
    print("bound = (min(2r, 1))^2, comparable to the subspace distance (mean sin^2 of the principal angles)")


if __name__ == "__main__":
    main()
