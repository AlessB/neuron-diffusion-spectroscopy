"""
D42, asked by AB: compare the 0.10 and 0.15 decade coincidence windows properly, with the wider
implications and secondary effects.

An "event" is found by change-point detection at a transition between two checkpoints, so its
time is only known to within that transition's interval. D37/D42 then call two events
"coincident" when they are close in t = log10(step + 1). This script measures, on the real
checkpoint grids:

  A. Grid geometry: how many checkpoints a window spans, early, middle and late in training.
     Pythia's full grid (154), the week-1 grid (25) and Stanford CRFM GPT-2 small (609, D45).
  B. Chance coincidence: with an event of ours at a random transition and K baseline change points
     at random transitions, how often do they "coincide" by luck? Two null models: change points
     uniform over transitions (what PELT on a checkpoint-indexed series does) and uniform in
     log-time.
  C. Power: a real shared event, seen by our readout and a baseline with a true lead or lag of
     delta decades and some timing noise (PELT can be off by one transition): how often is it
     called coincident?
  D. Rules compared: point distance (the current timeline.coincide, the later checkpoint of each
     transition) at 0.10 and 0.15; interval distance (0 if the two transitions overlap or touch)
     at 0.10 and 0.15; and a three-way rule (coincident <= 0.10 < ambiguous <= 0.15 < separate).

    python research/round5/d42_coincidence_windows.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from nds.checkpoints import all_pythia_steps
from nds.checkpoints import crfm_gpt2_small_steps
from nds.checkpoints import week_one_steps

TRIALS = 20000
GENERATOR = np.random.default_rng(42)


def log_time(steps):
    return np.log10(np.asarray(steps, dtype=np.float64) + 1.0)


def transitions(steps):
    """
    Each transition as (start time, end time) in decades.
    """
    times = log_time(steps)
    return np.stack([times[:-1], times[1:]], axis=1)


def point_distance(grid, index_a, index_b):
    return np.abs(grid[index_a, 1] - grid[index_b, 1])


def interval_distance(grid, index_a, index_b):
    gap_one = grid[index_b, 0] - grid[index_a, 1]
    gap_two = grid[index_a, 0] - grid[index_b, 1]
    return np.maximum(0.0, np.maximum(gap_one, gap_two))


def transition_at(grid, time):
    """
    Index of the transition containing a log-time (clipped to the grid).
    """
    index = np.searchsorted(grid[:, 1], time, side="left")
    return np.clip(index, 0, len(grid) - 1)


def section_a(grids):
    print("A. How many checkpoints lie within +-w decades of a checkpoint (median), by stretch")
    print(f"   {'grid':22s} {'stretch':>18s} " + " ".join(f"{'w=' + str(w):>7s}" for w in [0.05, 0.10, 0.15, 0.20]))
    for name, steps in grids.items():
        times = log_time(steps)
        stretches = [("steps < 1,000", np.asarray(steps) < 1000), ("1,000 to 10,000", (np.asarray(steps) >= 1000) & (np.asarray(steps) <= 10000)), ("above 10,000", np.asarray(steps) > 10000)]
        for label, mask in stretches:
            if not np.any(mask):
                continue
            cells = []
            for width in [0.05, 0.10, 0.15, 0.20]:
                counts = [int(np.sum(np.abs(times - time) <= width)) - 1 for time in times[mask]]
                cells.append(f"{np.median(counts):7.0f}")
            print(f"   {name:22s} {label:>18s} " + " ".join(cells))
        gaps = np.diff(times)
        widest = int(np.argmax(gaps[1:])) + 1
        print(f"   {name:22s} widest gap after step 0: {steps[widest]} -> {steps[widest + 1]} = {gaps[widest]:.2f} decades")


def chance_coincidence(grid, number_of_baseline_points, rule, width, null):
    hits = 0
    total_time = grid[-1, 1] - grid[1, 0]
    for _ in range(TRIALS):
        if null == "index":
            ours = GENERATOR.integers(1, len(grid))
            theirs = GENERATOR.integers(1, len(grid), size=number_of_baseline_points)
        else:
            ours = transition_at(grid, grid[1, 0] + GENERATOR.random() * total_time)
            theirs = transition_at(grid, grid[1, 0] + GENERATOR.random(number_of_baseline_points) * total_time)
        distances = rule(grid, np.full(len(theirs), ours), theirs)
        if np.any(distances <= width + 1e-12):
            hits += 1
    return hits / TRIALS


def section_b(grids):
    print("\nB. Chance coincidence: P(an event of ours 'coincides' with one of K random baseline change points)")
    rules = [("point", point_distance), ("interval", interval_distance)]
    for name in ["Pythia 154", "Pythia week-1 (25)"]:
        grid = transitions(grids[name])
        for null in ["index", "log-time"]:
            print(f"   {name}, change points uniform over {null}:")
            for rule_name, rule in rules:
                cells = []
                for count in [1, 3, 5]:
                    for width in [0.10, 0.15]:
                        cells.append(f"K={count} w={width:.2f}: {chance_coincidence(grid, count, rule, width, null):.2f}")
                print(f"      {rule_name:8s} " + "  ".join(cells))


def power(grid, lead, timing_noise, width, rule, off_by_one=0.3):
    total_time = grid[-1, 1] - grid[2, 0]
    hits = 0
    for _ in range(TRIALS):
        true_time = grid[2, 0] + GENERATOR.random() * total_time
        ours = transition_at(grid, true_time + GENERATOR.normal(0.0, timing_noise))
        theirs = transition_at(grid, true_time + lead + GENERATOR.normal(0.0, timing_noise))
        if GENERATOR.random() < off_by_one:
            ours = int(np.clip(ours + GENERATOR.choice([-1, 1]), 1, len(grid) - 1))
        if GENERATOR.random() < off_by_one:
            theirs = int(np.clip(theirs + GENERATOR.choice([-1, 1]), 1, len(grid) - 1))
        if rule(grid, np.array([ours]), np.array([theirs]))[0] <= width + 1e-12:
            hits += 1
    return hits / TRIALS


def section_c(grids):
    print("\nC. Power: P(called coincident) for a shared event whose baseline change lags ours by `lead` decades")
    print("   timing noise 0.03 decades per readout, PELT off by one transition 30% of the time")
    leads = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
    for name in ["Pythia 154", "Pythia week-1 (25)", "CRFM GPT-2 (609)"]:
        grid = transitions(grids[name])
        print(f"   {name}: lead " + " ".join(f"{lead:>6.2f}" for lead in leads))
        for rule_name, rule in [("point", point_distance), ("interval", interval_distance)]:
            for width in [0.10, 0.15]:
                values = [power(grid, lead, 0.03, width, rule) for lead in leads]
                print(f"      {rule_name:8s} w={width:.2f}  " + " ".join(f"{value:6.2f}" for value in values))


def main():
    grids = {
        "Pythia 154": all_pythia_steps(),
        "Pythia week-1 (25)": week_one_steps(),
        "CRFM GPT-2 (609)": crfm_gpt2_small_steps(),
    }
    section_a(grids)
    section_b(grids)
    section_c(grids)


if __name__ == "__main__":
    main()
