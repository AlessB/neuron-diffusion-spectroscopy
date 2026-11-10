"""
export_results.py
=================

Copy the small, shareable part of finished runs into the repo, so it can go to GitHub (D70: code and
results are public; the large arrays are not).

    python scripts/export_results.py --configs configs/pythia70m-standard_week1.yaml configs/pythia70m-deduped_week1.yaml --comparisons pythia70m-standard_week1__vs__pythia70m-deduped_week1 --out results/week1

Copies, for each run:
    step<NNNNNN>/summary.json              every readout of that checkpoint (about 170 KB)
    analysis/*.csv, *.png, *.json          the tables, figures and crossing lists of analyse.py
and for each comparison folder under <runs>/comparisons/:
    *.csv, *.png                           the figures and tables of compare_runs.py
Never copies .npz files (moments, spectra, token samples: about 24 GB per run). Those stay on the
runs drive and its backup.

Running it again overwrites the copies with the current files, so it is safe to repeat after a
rerun of analyse.py.
"""

import argparse
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nds import storage

ANALYSIS_SUFFIXES = (".csv", ".png", ".json")
COMPARISON_SUFFIXES = (".csv", ".png")


def copy_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination.stat().st_size


def export_run(run_folder, output_folder):
    """
    Copy one run's summaries and analysis outputs. Returns (number of files, bytes).
    """
    run_folder = Path(run_folder)
    output_folder = Path(output_folder)

    number_of_files = 0
    number_of_bytes = 0

    for step in storage.finished_steps(run_folder):
        source_folder = storage.step_folder(run_folder, step)
        source = source_folder / "summary.json"
        destination = output_folder / source_folder.name / "summary.json"
        number_of_bytes += copy_file(source, destination)
        number_of_files += 1

    analysis_folder = run_folder / "analysis"

    if analysis_folder.exists():
        for source in sorted(analysis_folder.iterdir()):
            if not source.is_file():
                continue
            if source.suffix not in ANALYSIS_SUFFIXES:
                continue
            destination = output_folder / "analysis" / source.name
            number_of_bytes += copy_file(source, destination)
            number_of_files += 1

    return number_of_files, number_of_bytes


def export_comparison(comparison_folder, output_folder):
    comparison_folder = Path(comparison_folder)
    output_folder = Path(output_folder)

    number_of_files = 0
    number_of_bytes = 0

    for source in sorted(comparison_folder.iterdir()):
        if not source.is_file():
            continue
        if source.suffix not in COMPARISON_SUFFIXES:
            continue
        destination = output_folder / source.name
        number_of_bytes += copy_file(source, destination)
        number_of_files += 1

    return number_of_files, number_of_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--comparisons", nargs="*", default=[])
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()

    output_root = Path(arguments.out)

    total_files = 0
    total_bytes = 0

    for config_path in arguments.configs:
        with open(config_path, "r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
        run_name = config["run_name"]
        run_folder = storage.run_folder_for(run_name)

        files, size = export_run(run_folder, output_root / run_name)
        total_files += files
        total_bytes += size
        print(f"{run_name}: {files} files, {size / 1e6:.1f} MB")

    for comparison_name in arguments.comparisons:
        comparison_folder = storage.runs_root() / "comparisons" / comparison_name

        if not comparison_folder.exists():
            print(f"{comparison_name}: not found under {storage.runs_root() / 'comparisons'}, skipped")
            continue

        files, size = export_comparison(comparison_folder, output_root / "comparisons" / comparison_name)
        total_files += files
        total_bytes += size
        print(f"{comparison_name}: {files} files, {size / 1e6:.1f} MB")

    print(f"total: {total_files} files, {total_bytes / 1e6:.1f} MB in {output_root}")


if __name__ == "__main__":
    main()
