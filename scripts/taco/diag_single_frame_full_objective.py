"""
Decisive bisection requested: does TACO t3 LH frame 100, optimized ALONE
(num_envs=1) with the FULL objective (tracking + collision, weight=1, mean
aggregation -- mean over 1 env is a no-op, so this exactly matches the loss
shape used in the full 211-env run) and the SAME default initial pose the
full run uses, reproduce the full run's bad frame-100 result, or does it
converge cleanly?

If clean: the problem is something about multi-frame optimization (shared
state, LR schedule, batch effects) despite per-env parameter independence.
If bad: frame 100 is intrinsically hard for this objective regardless of
batching, and the multi-frame run isn't the cause.

Must be run from the repo root: python scripts/taco/diag_single_frame_full_objective.py
"""

from isaacgym import gymapi  # noqa: F401 -- must be first import in the whole process

import numpy as np
import torch
from termcolor import cprint

from main.dataset.mano2dexhand import Mano2Dexhand, pack_data
from main.dataset.factory import ManipDataFactory
from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory

FRAME = 100


class FakeArgs:
    headless = True
    physics_engine = gymapi.SIM_PHYSX
    num_threads = 4
    use_gpu = True
    use_gpu_pipeline = True
    sim_device = "cuda:0"
    compute_device_id = 0
    graphics_device_id = 0
    iter = 0


def main():
    dexhand = DexHandFactory.create_hand("inspire", "left")
    dataset_type = ManipDataFactory.dataset_type("t3")
    demo_d = ManipDataFactory.create_data(
        manipdata_type=dataset_type,
        side="left",
        device="cuda:0",
        mujoco2gym_transf=torch.eye(4, device="cuda:0"),
        dexhand=dexhand,
        verbose=False,
    )
    demo_data = pack_data([demo_d["t3"]], dexhand)

    obj_trajectory = demo_data["obj_trajectory"][FRAME : FRAME + 1]
    target_wrist_pos = demo_data["wrist_pos"][FRAME : FRAME + 1]
    target_wrist_rot = demo_data["wrist_rot"][FRAME : FRAME + 1]
    target_mano_joints = demo_data["mano_joints"][FRAME : FRAME + 1].view(1, -1, 3)

    args = FakeArgs()
    args.num_envs = 1
    mano2inspire = Mano2Dexhand(args, dexhand, demo_data["obj_urdf_path"][0])

    cprint(f"Optimizing frame {FRAME} ALONE, full objective (tracking + collision, weight=1, default init, 5000 iters)", "cyan")

    to_dump = mano2inspire.fitting(
        5000,
        obj_trajectory,
        target_wrist_pos,
        target_wrist_rot,
        target_mano_joints,
        collision_weight=1.0,
        # tracking_weight_scale defaults to 1.0, init_* default to None (same default-pose init
        # the full 211-env run uses) -- deliberately NOT overridden, to exactly match what env
        # 100 experiences inside the full run.
    )

    print("\n=== single-frame-100 result ===")
    print("opt_wrist_pos:", to_dump["opt_wrist_pos"])
    print("opt_wrist_rot (axis-angle):", to_dump["opt_wrist_rot"])
    print("opt_dof_pos:", to_dump["opt_dof_pos"])


if __name__ == "__main__":
    main()
