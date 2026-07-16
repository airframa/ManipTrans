"""
Warm-start test: does TACO t3 LH frame 100, optimized ALONE with the FULL
objective (tracking + collision, weight=1, mean agg -- identical settings
to diag_single_frame_full_objective.py, which plateaued at collision_loss
~0.0034-0.0039 from the default init pose), converge to near-zero
penetration with NO side-crossing when warm-started from the pre-collision
M3 retargeting pose instead (MANO-clean, correctly-sided, just 2-5mm
penetrating)?

If yes: initialization was the whole problem, no new loss term needed.
If it still tunnels/plateaus: crossing happens even from a correct start
(tracking pulling through), and a side-aware penalty is the next step.

Must be run from the repo root: python scripts/taco/diag_warmstart_test.py
"""

from isaacgym import gymapi  # noqa: F401 -- must be first import in the whole process

import pickle

import numpy as np
import torch
from termcolor import cprint

from main.dataset.mano2dexhand import Mano2Dexhand, pack_data
from main.dataset.factory import ManipDataFactory
from main.dataset.transform import aa_to_rot6d
from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory

FRAME = 100
BASELINE_PKL = "/tmp/taco_retarget_backup/20231020_232@left.pkl"


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

    with open(BASELINE_PKL, "rb") as f:
        baseline = pickle.load(f)
    init_wrist_pos = baseline["opt_wrist_pos"][FRAME : FRAME + 1]  # already in gym/transformed frame
    init_wrist_rot_aa = baseline["opt_wrist_rot"][FRAME : FRAME + 1]  # axis-angle, per dump format
    init_wrist_rot = aa_to_rot6d(torch.tensor(init_wrist_rot_aa, device="cuda:0", dtype=torch.float32)).cpu().numpy()
    init_dof_pos = baseline["opt_dof_pos"][FRAME : FRAME + 1]

    cprint(f"Warm-starting frame {FRAME} from the PRE-COLLISION baseline pose, full objective (weight=1, 5000 iters)", "cyan")

    to_dump = mano2inspire.fitting(
        5000,
        obj_trajectory,
        target_wrist_pos,
        target_wrist_rot,
        target_mano_joints,
        collision_weight=1.0,
        init_wrist_pos=init_wrist_pos,
        init_wrist_rot=init_wrist_rot,
        init_dof_pos=init_dof_pos,
    )

    print("\n=== warm-start frame-100 result ===")
    print("opt_wrist_pos:", to_dump["opt_wrist_pos"])
    print("opt_wrist_rot (axis-angle):", to_dump["opt_wrist_rot"])
    print("opt_dof_pos:", to_dump["opt_dof_pos"])


if __name__ == "__main__":
    main()
