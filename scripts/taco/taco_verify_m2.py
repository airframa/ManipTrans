"""
M2 verification entry point. Must be run directly (not with -m) so that isaacgym is
imported before anything imports `main.dataset`, whose package __init__.py auto-imports
every loader in that directory -- including mano2dexhand.py, which requires isaacgym to
be imported before torch anywhere in the process.

Must be run from the repo root: all data paths (data/taco/..., data/retargeting/...)
are resolved relative to CWD, not to this file's location.

Usage (from repo root): python scripts/taco/taco_verify_m2.py
"""

from isaacgym import gymapi  # noqa: F401 -- must be first import in the whole process

from main.dataset.taco_dataset_dexhand import run_verification

if __name__ == "__main__":
    run_verification()
