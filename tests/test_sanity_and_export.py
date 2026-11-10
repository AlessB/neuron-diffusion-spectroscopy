"""
scripts/sanity.py (P1.4) and scripts/export_results.py on a small fake run folder.
"""

import sys
from pathlib import Path

import numpy as np

from nds import storage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def fake_summary(step, probe_loss, induction_drop, module_sizes):
    layer = {
        "spectrum": {
            "module_count": len(module_sizes),
            "module_sizes": module_sizes,
            "gap_ratio": 10.0,
            "dead_fraction": 0.0,
            "module_aligned_localization": 2.0,
        },
        "rotations": [
            {"spectrum": {"module_aligned_localization": 0.8}},
            {"spectrum": {"module_aligned_localization": 0.9}},
        ],
        "massive_neurons": [],
    }
    summary = {
        "step": step,
        "probe_loss": probe_loss,
        "induction": {"induction_loss_drop": induction_drop},
        "layers": {"0": layer},
    }
    return summary


def write_fake_step(run_folder, step, labels, kurtosis):
    folder = storage.step_folder(run_folder, step)
    sizes = np.bincount(labels).tolist()
    storage.save_summary(folder, fake_summary(step, 10.9 if step == 0 else 4.0, 0.0, sizes))
    number_of_neurons = len(labels)
    storage.save_layer_spectrum(
        folder,
        0,
        {
            "laplacian_eigenvalues": np.ones(3),
            "symmetric_vectors": np.zeros((number_of_neurons, 3)),
            "live_neuron_ids": np.arange(number_of_neurons),
            "module_labels": labels.astype(np.int32),
        },
    )
    storage.save_neuron_statistics(
        folder,
        0,
        {
            "activation_kurtosis": kurtosis,
            "positive_fraction": np.full(number_of_neurons, 0.2),
            "position_zero_mean_absolute": np.full(number_of_neurons, 0.1),
        },
    )


def test_sanity_finds_the_small_module_and_its_neurons(tmp_path):
    from sanity import sanity_tables
    from sanity import step_zero_check

    run_folder = tmp_path / "run"
    number_of_neurons = 100

    # step 0: two big modules, nothing small
    labels_step_zero = np.array([0] * 50 + [1] * 50)
    write_fake_step(run_folder, 0, labels_step_zero, np.full(number_of_neurons, 3.0))

    # step 10: neurons 7, 8, 9 split off (3 % of the layer) and have a huge kurtosis
    labels_step_ten = np.zeros(number_of_neurons, dtype=int)
    labels_step_ten[[7, 8, 9]] = 1
    kurtosis_step_ten = np.full(number_of_neurons, 3.0)
    kurtosis_step_ten[[7, 8, 9]] = 500.0
    write_fake_step(run_folder, 10, labels_step_ten, kurtosis_step_ten)

    # step 20: neurons 8, 9, 40 split off
    labels_step_twenty = np.zeros(number_of_neurons, dtype=int)
    labels_step_twenty[[8, 9, 40]] = 1
    write_fake_step(run_folder, 20, labels_step_twenty, kurtosis_step_ten)

    rows, small_rows = sanity_tables(run_folder, small_fraction=0.05)

    assert len(rows) == 3
    assert rows[0]["module_sizes"] == "50 50"
    assert rows[0]["rotated_module_aligned_max"] == 0.9

    assert [row["step"] for row in small_rows] == [10, 20]
    assert small_rows[0]["neurons"] == "7 8 9"
    assert small_rows[0]["size"] == 3
    assert small_rows[0]["activation_kurtosis_small"] == 500.0
    assert small_rows[0]["activation_kurtosis_rest"] == 3.0
    assert small_rows[0]["also_in_previous_small_module"] == ""
    assert small_rows[1]["also_in_previous_small_module"] == 2

    lines = step_zero_check(rows)
    assert "ok" in lines[0]
    assert "ok" in lines[1]


def test_export_copies_summaries_and_analysis_but_never_npz(tmp_path):
    from export_results import export_comparison
    from export_results import export_run

    run_folder = tmp_path / "runs" / "run"
    labels = np.array([0] * 5 + [1] * 5)
    write_fake_step(run_folder, 0, labels, np.full(10, 3.0))
    write_fake_step(run_folder, 4, labels, np.full(10, 3.0))

    analysis_folder = run_folder / "analysis"
    analysis_folder.mkdir()
    (analysis_folder / "layer0_table.csv").write_text("step\n0\n4\n")
    (analysis_folder / "layer0_overview.png").write_bytes(b"png")
    (analysis_folder / "layer0_crossings.json").write_text("[]")
    np.savez(analysis_folder / "layer0_flow.npz", overlap=np.zeros(2))

    comparison_folder = tmp_path / "runs" / "comparisons" / "a__vs__b"
    comparison_folder.mkdir(parents=True)
    (comparison_folder / "layer0_compare.csv").write_text("x\n")
    (comparison_folder / "layer0_compare.png").write_bytes(b"png")

    output = tmp_path / "results"
    files, size = export_run(run_folder, output / "run")
    comparison_files, comparison_size = export_comparison(comparison_folder, output / "comparisons" / "a__vs__b")

    assert files == 5
    assert comparison_files == 2
    assert (output / "run" / "step000000" / "summary.json").exists()
    assert (output / "run" / "step000004" / "summary.json").exists()
    assert (output / "run" / "analysis" / "layer0_overview.png").exists()
    assert (output / "comparisons" / "a__vs__b" / "layer0_compare.csv").exists()

    copied_npz = list(output.rglob("*.npz"))
    assert copied_npz == []
