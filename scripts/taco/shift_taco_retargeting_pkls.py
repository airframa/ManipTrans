"""
Regenerates TACO's existing retargeted pkl files to reflect the new
TACO_HEIGHT_DELTA table-height correction (taco_dataset_dexhand.py), WITHOUT
re-running mano2dexhand.py's optimization.

Why this is exact, not an approximation: TACO_HEIGHT_DELTA is a pure
translation along raw Y, applied before Mano2Dexhand's table transform. That
table transform maps raw Y -> final Z (verified empirically), so the net
effect on every target position (wrist, object, MANO joints) the optimizer
sees is a constant [0, 0, TACO_HEIGHT_DELTA] shift -- identical for every
frame. A rigid translation of every target is exactly compensated by the same
translation of the optimal wrist position; local DOF angles and wrist
orientation are frame-invariant, so they don't change at all.

So: opt_wrist_pos += [0,0,delta], opt_joints_pos += [0,0,delta] (every body,
every frame), opt_wrist_rot and opt_dof_pos untouched.

Must be run from the repo root: python scripts/taco/shift_taco_retargeting_pkls.py
"""

# isaacgym must be imported before torch anywhere in the process (pre-existing repo
# quirk: main/dataset/__init__.py auto-imports every file in that directory, including
# mano2dexhand.py). We don't need isaacgym ourselves here, but importing
# main.dataset.taco_dataset_dexhand triggers that same package init regardless.
from isaacgym import gymapi  # noqa: F401

import pickle

import numpy as np

from main.dataset.taco_dataset_dexhand import SEQUENCES, TACO_HEIGHT_DELTA

SHIFT = np.array([0.0, 0.0, TACO_HEIGHT_DELTA], dtype=np.float32)


def shift_pkl(path):
    with open(path, "rb") as f:
        opt = pickle.load(f)

    if opt.get("_height_corrected"):
        print(f"  {path}: already corrected, skipping")
        return

    opt["opt_wrist_pos"] = opt["opt_wrist_pos"] + SHIFT
    opt["opt_joints_pos"] = opt["opt_joints_pos"] + SHIFT[None, None, :]
    opt["_height_corrected"] = True  # marker so this script is idempotent

    with open(path, "wb") as f:
        pickle.dump(opt, f)
    print(f"  {path}: shifted by {SHIFT.tolist()}")


def main():
    print(f"Shifting all TACO retargeting pkls by {SHIFT.tolist()} (TACO_HEIGHT_DELTA={TACO_HEIGHT_DELTA})")
    for idx, seq in enumerate(SEQUENCES):
        for side, dexhand_str in [("right", "inspire_rh"), ("left", "inspire_lh")]:
            path = f"data/retargeting/taco/mano2{dexhand_str}/{seq['seq_name']}@{side}.pkl"
            shift_pkl(path)


if __name__ == "__main__":
    main()
