"""
worlds_report.py
================

The rank test of the real neurons against the rotated worlds (D41 = E), after

    python scripts/reanalyse.py --config <run config> --worlds 0-19

    python scripts/worlds_report.py --config <run config>

writes <run>/analysis/layer<L>_worlds.csv (one row per checkpoint: each readout's real value, the
worlds' range and the p-value) and prints the checkpoints where a readout beats every world.
p = 0.05 is the smallest possible with 19 worlds: "beats every world".
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nds import storage
from nds.worlds import NUMBER_OF_WORLDS
from nds.worlds import world_rank_table
from nds.worlds import world_run_name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--number-of-worlds", type=int, default=NUMBER_OF_WORLDS)
    arguments = parser.parse_args()

    with open(arguments.config, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    run_name = config.get("source_run", config["run_name"])

    real_folder = storage.run_folder_for(world_run_name(run_name, 0))
    first_step = storage.finished_steps(real_folder)[0]
    first_summary = storage.load_summary(storage.step_folder(real_folder, first_step))
    layers = sorted(int(layer) for layer in first_summary["layers"].keys())

    output_folder = storage.run_folder_for(run_name) / "analysis"
    output_folder.mkdir(parents=True, exist_ok=True)

    smallest_p = 1.0 / (1.0 + arguments.number_of_worlds)

    for layer in layers:
        table = world_rank_table(run_name, layer, arguments.number_of_worlds)
        names = [name for name in table.keys() if name != "step"]

        header = ["step"]
        for name in names:
            header.extend([f"{name}", f"{name}_world_min", f"{name}_world_max", f"{name}_p"])
            if name.endswith("_strength"):
                header.append(f"{name}_p_where_modules_exist")

        lines = [",".join(header)]
        for row in range(len(table["step"])):
            cells = [str(int(table["step"][row]))]
            for name in names:
                entry = table[name]
                for key in ["real", "world_minimum", "world_maximum", "p"]:
                    cells.append(f"{entry[key][row]:.6g}")
                if "p_where_modules_exist" in entry:
                    cells.append(f"{entry['p_where_modules_exist'][row]:.6g}")
            lines.append(",".join(cells))

        path = output_folder / f"layer{layer}_worlds.csv"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        for name in names:
            key = "p"
            if "p_where_modules_exist" in table[name]:
                key = "p_where_modules_exist"
            beats = [int(table["step"][row]) for row in range(len(table["step"])) if table[name][key][row] <= smallest_p + 1e-12]
            print(f"layer {layer} {name:32s} beats all {arguments.number_of_worlds} worlds at steps: {beats}")

        print(f"wrote {path}")


if __name__ == "__main__":
    main()
