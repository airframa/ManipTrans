"""
Feasibility check requested before any aggregation fix: starting from the
BASELINE (pre-collision-term) converged pose at TACO t3 LH frame 100 --
where thumb_distal/thumb_intermediate are known to penetrate the plate --
run gradient descent on the COLLISION TERM ALONE (tracking_weight_scale
near-zero) and see whether it can reach a genuinely collision-free
configuration nearby, or plateaus/oscillates at a nonzero floor.

Starting AT the baseline pose (not a fresh/default pose) matters: with the
collision loss's one-sided hinge, gradient is exactly zero once a link
clears the surface, so if a nearby collision-free configuration exists,
pure local gradient descent should find it and then STOP (Adam's moving
averages decay, loss goes flat at 0). If no such configuration exists near
this pose, the optimizer will keep pushing different links back and forth
(whack-a-mole via DOF coupling) and the loss will plateau above zero
instead of reaching it.

Must be run from the repo root: python scripts/taco/diag_collision_only_feasibility.py
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
    num_envs_full = demo_data["mano_joints"].shape[0]
    print(f"full sequence: {num_envs_full} frames")

    # slice out just frame 100, KEEPING the leading dim (num_envs=1)
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

    cprint(f"Starting from BASELINE frame-{FRAME} pose (known penetration: thumb_distal +5.04mm, thumb_intermediate +1.19mm)", "cyan")
    cprint("Running collision-ONLY optimization (tracking_weight_scale=0.001, collision_weight=50)...", "cyan")

    to_dump = mano2inspire.fitting(
        2000,
        obj_trajectory,
        target_wrist_pos,
        target_wrist_rot,
        target_mano_joints,
        collision_weight=50.0,
        tracking_weight_scale=0.001,
        init_wrist_pos=init_wrist_pos,
        init_wrist_rot=init_wrist_rot,
        init_dof_pos=init_dof_pos,
    )

    final_wrist_pos = to_dump["opt_wrist_pos"]
    final_dof_pos = to_dump["opt_dof_pos"]
    wrist_drift_mm = np.linalg.norm(final_wrist_pos - init_wrist_pos, axis=-1) * 1000
    dof_drift = np.abs(final_dof_pos - init_dof_pos).mean()
    cprint(f"\nwrist position drift from baseline pose: {wrist_drift_mm} mm", "magenta")
    cprint(f"mean abs DOF drift from baseline pose: {dof_drift:.5f} rad", "magenta")


if __name__ == "__main__":
    main()
