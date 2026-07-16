"""
Sanity-check the two anti-tunneling penalty design fixes before implementing
them in mano2dexhand.py:
  1. Per-point margin via ray-marching to the far-side SDF zero-crossing
     along -n0_k (not a fixed global constant).
  2. Cross-check n0_k (SDF gradient at x0_k) against m0_k (direction toward
     the corresponding MANO reference joint) -- flag and correct
     disagreements.

Run for LH frame 100's 13 collision-link sample points (link ORIGIN used as
a representative x0_k per link, for this sanity check -- the real
implementation will do this per actual sample point, ~25 per link).

Must be run from the repo root: python scripts/taco/diag_tunnel_margin_check.py
"""

from isaacgym import gymapi  # noqa: F401 -- must be first import in the whole process

import os
import pickle

import numpy as np
import torch

from main.dataset.mano2dexhand import pack_data
from main.dataset.factory import ManipDataFactory
from main.dataset.transform import aa_to_rotmat, rot6d_to_rotmat, rotmat_to_rot6d, aa_to_rot6d
from main.dataset.collision_sdf import load_or_build_object_sdf, query_sdf, sample_link_points
from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory
import pytorch_kinematics as pk

FRAME = 100
BASELINE_PKL = "/tmp/taco_retarget_backup/20231020_232@left.pkl"


def main():
    device = "cuda:0"
    dexhand = DexHandFactory.create_hand("inspire", "left")
    dataset_type = ManipDataFactory.dataset_type("t3")
    demo_d = ManipDataFactory.create_data(
        manipdata_type=dataset_type, side="left", device=device,
        mujoco2gym_transf=torch.eye(4, device=device), dexhand=dexhand, verbose=False,
    )
    demo_data = pack_data([demo_d["t3"]], dexhand)

    mujoco2gym_transf = np.eye(4)
    mujoco2gym_transf[:3, :3] = aa_to_rotmat(np.array([0, 0, -np.pi / 2])) @ aa_to_rotmat(np.array([np.pi / 2, 0, 0]))
    mujoco2gym_transf[:3, 3] = np.array([0, 0, 0.415])
    M = torch.tensor(mujoco2gym_transf, dtype=torch.float32, device=device)

    obj_trajectory = M @ demo_data["obj_trajectory"][FRAME]
    obj_R = obj_trajectory[:3, :3]
    obj_t = obj_trajectory[:3, 3]

    # MANO reference joint world positions for this frame (target_mano_joints order matches
    # dexhand.body_names excluding wrist; wrist itself = target_wrist_pos)
    mano_joints_flat = (M[:3, :3] @ demo_data["mano_joints"][FRAME].view(-1, 3).T).T + M[:3, 3]
    wrist_world = (M[:3, :3] @ demo_data["wrist_pos"][FRAME]) + M[:3, 3]

    non_wrist_body_names = [b for b in dexhand.body_names if b != "L_hand_base_link"]
    mano_ref_world = {name: mano_joints_flat[i] for i, name in enumerate(non_wrist_body_names)}
    mano_ref_world["L_hand_base_link"] = wrist_world

    # warm-start pose (pre-collision baseline pkl)
    with open(BASELINE_PKL, "rb") as f:
        baseline = pickle.load(f)
    wrist_pos = torch.tensor(baseline["opt_wrist_pos"][FRAME], dtype=torch.float32, device=device)
    wrist_rot_aa = torch.tensor(baseline["opt_wrist_rot"][FRAME], dtype=torch.float32, device=device)
    dof_pos = torch.tensor(baseline["opt_dof_pos"][FRAME], dtype=torch.float32, device=device)[None]
    wrist_R = rot6d_to_rotmat(rotmat_to_rot6d(aa_to_rotmat(wrist_rot_aa)))[None]

    chain = pk.build_chain_from_urdf(open(dexhand.urdf_path).read()).to(dtype=torch.float32, device=device)
    dof_names_isaac_order = dexhand.dof_names
    chain_joint_names = chain.get_joint_parameter_names()
    isaac2chain_order = [dof_names_isaac_order.index(j) for j in chain_joint_names]
    ret = chain.forward_kinematics(dof_pos[:, isaac2chain_order])

    sdf, lo, hi = load_or_build_object_sdf("166", voxel_mm=0.4, device=device)

    inspire_root = os.path.split(dexhand.urdf_path)[0]
    link_local_points = sample_link_points(inspire_root, "L", n_points_per_link=25, device=device)

    def ray_march_margin(x0, n0_final, sdf_x0):
        step = 0.0004  # 0.4mm, matches grid voxel
        max_range = 0.06  # 60mm search bound
        crossed_in = sdf_x0 > 0
        n_steps = int(max_range / step)
        pts = x0[None, :] - n0_final[None, :] * (torch.arange(1, n_steps + 1, device=device).float() * step)[:, None]
        sdf_ray = query_sdf(sdf, lo, hi, pts).detach().cpu().numpy()
        for i, v in enumerate(sdf_ray):
            tt = (i + 1) * step
            if not crossed_in and v > 0:
                crossed_in = True
            elif crossed_in and v < 0:
                return tt
        return max_range

    print(f"{'link':22s} {'pt':>3s} {'SDF(x0) mm':>10s} {'n0.dot(m0)':>11s} {'flip?':>6s} {'margin(ray) mm':>15s}")
    for body_name in dexhand.body_names:
        link_key = body_name[2:]
        if link_key not in link_local_points:
            continue  # *_tip links: visual-only, no collision geometry
        local_pts = link_local_points[link_key]  # (nP, 3) in link-local frame

        chain_mat = ret[body_name].get_matrix()  # (1,4,4)
        chain_R = chain_mat[:, :3, :3]
        chain_t = chain_mat[:, :3, 3]
        pts_chain = local_pts[None] @ chain_R.transpose(-1, -2) + chain_t[:, None, :]  # (1,nP,3)
        pts_world = (wrist_R @ pts_chain.transpose(-1, -2)).transpose(-1, -2) + wrist_pos[None, None, :]
        pts_obj_local = ((pts_world[0] - obj_t[None]) @ obj_R)  # (nP,3)

        sdf_vals = query_sdf(sdf, lo, hi, pts_obj_local).detach().cpu().numpy()
        worst_idx = int(np.argmax(sdf_vals))  # deepest-penetrating (or least-clear) sample point
        x0 = pts_obj_local[worst_idx]

        x0_g = x0.clone().requires_grad_(True)
        sdf_val = query_sdf(sdf, lo, hi, x0_g[None])
        (grad,) = torch.autograd.grad(sdf_val.sum(), x0_g)
        n0 = grad / grad.norm().clamp_min(1e-8)

        mano_world = mano_ref_world[body_name]
        mano_local = (mano_world - obj_t) @ obj_R
        m0_raw = mano_local - x0
        m0 = m0_raw / m0_raw.norm().clamp_min(1e-8)

        dot = (n0 * m0).sum().item()
        flip = dot < 0
        n0_final = m0 if flip else n0

        sdf_x0 = float(sdf_vals[worst_idx])
        margin_m = ray_march_margin(x0, n0_final, sdf_x0)

        print(f"{body_name:22s} {worst_idx:3d} {sdf_x0*1000:10.2f} {dot:11.3f} {str(flip):>6s} {margin_m*1000:15.2f}")


if __name__ == "__main__":
    main()
