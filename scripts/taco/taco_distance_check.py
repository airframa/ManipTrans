"""
Follow-up check on M2's hand/object assignment: plot wrist-to-object distance
over the whole sequence to confirm right-to-tool stays below right-to-target
(and symmetrically for the left hand), rather than relying on per-frame 3D
scatter plots with independently-scaled axes.

Must be run from the repo root: all data paths (data/taco/..., data/retargeting/...)
are resolved relative to CWD, not to this file's location.

Usage (from repo root): python scripts/taco/taco_distance_check.py
"""

from isaacgym import gymapi  # noqa: F401 -- must be first import in the whole process

from main.dataset.taco_dataset_dexhand import run_distance_check

if __name__ == "__main__":
    run_distance_check()
