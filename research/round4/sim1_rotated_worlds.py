"""
S1. Option E of D41, prototyped: the whole analysis repeated in 19 rotated worlds (one fixed Haar
rotation per world, applied at every checkpoint to the neurons alive throughout), next to the
original neurons, on planted runs.

Questions
  1. Single checkpoint: does the original's module-aligned localisation beat all 19 worlds where
     modules exist, and not where they do not (forming run before the onset)?
  2. Between checkpoints: each world judges its own changes against its own quiet transitions:
         strength = change at a transition / median change over the other transitions.
     At the planted event, is the original's strength above all 19 worlds? In the stable run, how
     often does any world (or the original) produce a "strength" as large, i.e. a false event?

    python research/round4/sim1_rotated_worlds.py
"""

import numpy as np

from sims_common import CHECKPOINTS
from sims_common import EVENT
from sims_common import always_alive
from sims_common import compare
from sims_common import make_run
from sims_common import summarise

from nds.controls import random_orthogonal_matrix
from nds.controls import rotate_covariance

WORLDS = 19
SEEDS = 2
MEASURES = ["one_minus_ari", "subspace_distance", "lineage_loss"]


def run_worlds(covariances, generator):
    alive = always_alive(covariances)
    worlds = []
    rotations = [None]
    for _ in range(WORLDS):
        rotations.append(random_orthogonal_matrix(len(alive), generator))
    for rotation in rotations:
        summaries = []
        for covariance in covariances:
            block = covariance[np.ix_(alive, alive)]
            if rotation is not None:
                block = rotate_covariance(block, rotation)
            summaries.append(summarise(block, np.arange(len(alive))))
        changes = {measure: [] for measure in MEASURES}
        for index in range(1, len(summaries)):
            result = compare(summaries[index - 1], summaries[index])
            for measure in MEASURES:
                changes[measure].append(result[measure])
        worlds.append({"summaries": summaries, "changes": {m: np.asarray(v) for m, v in changes.items()}})
    return worlds


def strength(changes, transition):
    others = np.delete(changes, transition)
    return changes[transition] / max(np.median(others), 1e-9)


def main():
    for scenario in ["stable", "reshuffle", "forming"]:
        print(f"\n===== {scenario} =====")
        for seed in range(SEEDS):
            generator = np.random.default_rng(100 + seed)
            worlds = run_worlds(make_run(scenario, generator), generator)
            original = worlds[0]
            rotated = worlds[1:]

            # 1. single checkpoint
            for name in ["module_aligned_localization", "module_aligned_support"]:
                cells = []
                for index in range(CHECKPOINTS):
                    value = original["summaries"][index][name]
                    best_rotated = max(world["summaries"][index][name] for world in rotated)
                    mark = "+" if value > best_rotated else "-"
                    cells.append(f"{value:.2f}/{best_rotated:.2f}{mark}")
                print(f"seed {seed} {name[:20]:20s} orig/max-of-19: " + " ".join(cells))

            # 2. between checkpoints (transition EVENT-1 is checkpoint EVENT-1 -> EVENT)
            transition = EVENT - 1
            for measure in MEASURES:
                if scenario == "stable":
                    original_score = max(strength(original["changes"][measure], t) for t in range(CHECKPOINTS - 1))
                    world_scores = [max(strength(w["changes"][measure], t) for t in range(CHECKPOINTS - 1)) for w in rotated]
                    label = "largest strength anywhere (false-event score)"
                else:
                    original_score = strength(original["changes"][measure], transition)
                    world_scores = [strength(w["changes"][measure], transition) for w in rotated]
                    label = "strength at the event"
                beats = int(np.sum(original_score > np.asarray(world_scores)))
                print(f"seed {seed} {measure:18s} {label}: original {original_score:7.2f}; worlds median {np.median(world_scores):6.2f}, max {np.max(world_scores):6.2f}; original beats {beats}/19")
                if scenario != "stable":
                    print(f"         raw change at event: original {original['changes'][measure][transition]:.3f}, worlds median {np.median([w['changes'][measure][transition] for w in rotated]):.3f}")


if __name__ == "__main__":
    main()
