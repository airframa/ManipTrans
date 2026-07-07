"""
M3 methodological check (Part 1): verify TACO and GRAB actually land in the
same coordinate convention before reaching Mano2Dexhand's optimizer, since a
hidden frame mismatch (origin/up-axis/handedness) could inflate TACO's
tracking-error numbers without that reflecting genuine kinematic difficulty.

Must be run from the repo root: python scripts/taco/check_coord_frames.py
"""

from isaacgym import gymapi  # noqa: F401 -- must be first import in the whole process

import numpy as np
import torch

from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory
import maniptrans_envs.lib.envs.dexhands  # noqa: F401

from main.dataset.taco_dataset_dexhand import TACORightData, TACOLeftData
from main.dataset.grab_dataset_dexhand import GrabDemoDexHand
from main.dataset.transform import aa_to_rotmat


def describe_rotmat(name, R):
    print(f"{name}:\n{R}")
    for axis_name, v in [("x", np.array([1, 0, 0])), ("y", np.array([0, 1, 0])), ("z", np.array([0, 0, 1]))]:
        print(f"    world {axis_name}-axis -> {R @ v}")


def main():
    device = "cuda:0"
    identity = torch.eye(4, dtype=torch.float32, device=device)

    dexhand_rh = DexHandFactory.create_hand("inspire", "right")
    dexhand_lh = DexHandFactory.create_hand("inspire", "left")

    # --- 1/2: what transform (if any) does each loader bake into self.mujoco2gym_transf,
    # applied inside process_data() BEFORE Mano2Dexhand ever sees the data? ---
    print("=" * 70)
    print("GRAB's baked-in transf_offset (grab_dataset_dexhand.py lines ~78-84):")
    grab_transf_offset = np.eye(4)
    grab_transf_offset[:3, :3] = aa_to_rotmat(np.array([-np.pi / 2, 0, 0])) @ aa_to_rotmat(np.array([0, 0, np.pi / 2]))
    grab_transf_offset[:3, 3] = np.array([0.0, 0.018, 0.0])
    describe_rotmat("  rotation", grab_transf_offset[:3, :3])
    print(f"  translation: {grab_transf_offset[:3, 3]}")

    print()
    print("TACO's taco_dataset_dexhand.py: no transf_offset override -- self.mujoco2gym_transf")
    print("stays exactly whatever the caller passes in (identity here), same as OakInk2's own loader.")

    print()
    print("=" * 70)
    print("Mano2Dexhand.__init__'s own separate table transform (mano2dexhand.py lines ~166-176),")
    print("applied identically to EVERY dataset inside fitting(), on top of whatever came out of process_data():")
    table_transf = np.eye(4)
    table_transf[:3, :3] = aa_to_rotmat(np.array([0, 0, -np.pi / 2])) @ aa_to_rotmat(np.array([np.pi / 2, 0, 0]))
    table_pos_z = 0.4
    table_half_height = 0.015
    table_transf[:3, 3] = np.array([0, 0, table_pos_z + table_half_height])
    describe_rotmat("  rotation", table_transf[:3, :3])
    print(f"  translation: {table_transf[:3, 3]}")

    # --- 3: concrete first-frame comparison, in the frame each ManipData hands to Mano2Dexhand ---
    print()
    print("=" * 70)
    print("First-frame data AS RETURNED BY EACH LOADER (mujoco2gym_transf=identity passed in,")
    print("i.e. exactly what mano2dexhand.py's run() does) -- this is what process_data() already applied:")

    fdata_rh = TACORightData(mujoco2gym_transf=identity, device=device, dexhand=dexhand_rh)
    fdata_lh = TACOLeftData(mujoco2gym_transf=identity, device=device, dexhand=dexhand_lh)
    data_rh = fdata_rh["t0"]
    data_lh = fdata_lh["t0"]

    fdata_g = GrabDemoDexHand(mujoco2gym_transf=identity, device=device, dexhand=dexhand_rh)
    data_g = fdata_g["g0"]

    def report(name, data):
        wrist0 = data["wrist_pos"][0].detach().cpu().numpy()
        obj0 = data["obj_trajectory"][0, :3, 3].detach().cpu().numpy()
        obj_rot0 = data["obj_trajectory"][0, :3, :3].detach().cpu().numpy()
        print(f"  [{name}]")
        print(f"    wrist_pos[0]        = {wrist0}")
        print(f"    obj_trajectory[0].t = {obj0}")
        print(f"    obj_trajectory[0].R row2 (local z-axis in world) = {obj_rot0[:, 2]}")

    report("TACO right (tool=brush)", data_rh)
    report("TACO left (target=pan)", data_lh)
    report("GRAB right (g0)", data_g)

    # --- apply Mano2Dexhand's table transform on top, exactly as fitting() does at its start ---
    print()
    print("=" * 70)
    print("SAME first-frame data AFTER Mano2Dexhand's table transform is additionally applied")
    print("(i.e. the actual frame the optimizer sees; table surface sits at z={:.3f}):".format(table_pos_z + table_half_height))

    table_transf_t = torch.tensor(table_transf, dtype=torch.float32, device=device)

    def report_table_frame(name, data):
        wrist0 = data["wrist_pos"][0]
        wrist0_t = (table_transf_t[:3, :3] @ wrist0) + table_transf_t[:3, 3]
        obj0 = data["obj_trajectory"][0, :3, 3]
        obj0_t = (table_transf_t[:3, :3] @ obj0) + table_transf_t[:3, 3]
        print(f"  [{name}]")
        print(f"    wrist_pos[0]        = {wrist0_t.detach().cpu().numpy()}")
        print(f"    obj_trajectory[0].t = {obj0_t.detach().cpu().numpy()}")

    report_table_frame("TACO right (tool=brush)", data_rh)
    report_table_frame("TACO left (target=pan)", data_lh)
    report_table_frame("GRAB right (g0)", data_g)


if __name__ == "__main__":
    main()
