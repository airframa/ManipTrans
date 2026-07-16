"""
Tunneling check (item 2): for the single-frame-100 LH converged result
(opt_wrist_pos/opt_dof_pos from diag_single_frame_full_objective.py), check
which SIDE of the plate each contact-bearing link ends up on, vs. where the
MANO reference fingertips are. The plate is thin+flat (23mm thick vs
243x243mm footprint) -- its local "up" axis is whichever principal axis has
the smallest extent. A one-sided hinge penalty can drive a link straight
through to the far side, where the SDF reads "clear" (negative) even though
the swept path deeply interpenetrated -- this check catches that by sign,
not magnitude.

Must be run from the repo root: python scripts/taco/diag_tunneling_check.py
"""

from isaacgym import gymapi  # noqa: F401 -- must be first import in the whole process

import numpy as np
import torch
import trimesh

from main.dataset.mano2dexhand import pack_data
from main.dataset.factory import ManipDataFactory
from main.dataset.transform import aa_to_rotmat, rotmat_to_rot6d, rot6d_to_rotmat
from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory
import pytorch_kinematics as pk
import os

import sys

_arg = sys.argv[1] if len(sys.argv) > 1 else "100"
FRAME = _arg if not _arg.lstrip("-").isdigit() else int(_arg)

if FRAME == 100:
    # converged single-frame-100 result from diag_single_frame_full_objective.py (rerun with
    # opt_wrist_rot captured)
    OPT_WRIST_POS = np.array([[-0.5984999, -0.11148248, 0.2545526]])
    OPT_WRIST_ROT_AA = np.array([[-1.9280379, -0.12222517, 1.9594206]])  # axis-angle
    OPT_DOF_POS = np.array(
        [[0.33073765, 0.9792572, 0.72945845, 0.7575655, 0.91382396, 1.0951825, 0.91767627, 0.5843085, 1.2827746, 0.5, 0.06082727, 0.0]]
    )
elif FRAME == "100warm":
    # warm-started (from pre-collision baseline pose) converged result
    OPT_WRIST_POS = np.array([[-0.6336855, -0.11075272, 0.26158702]])
    OPT_WRIST_ROT_AA = np.array([[2.82912, -0.40224487, -2.3909829]])
    OPT_DOF_POS = np.array(
        [[0.29880825, 1.019583, 0.63907695, 0.78726405, 0.49858767, 1.4559038, 1.0439752, 0.24585627, 1.3, 0.47900873, 0.03014361, 0.0853598]]
    )
    FRAME = 100  # for obj_trajectory/MANO reference lookup
elif FRAME == 130:
    # from the saved weight=1 LH pkl (full 211-env run), frame 130 -- not a single-frame
    # reoptimization, but sufficient to check side/tunneling on the actual full-run result.
    OPT_WRIST_POS = np.array([[-0.56181306, -0.10607983, 0.24940175]])
    OPT_WRIST_ROT_AA = np.array([[-1.9481547, -0.50056124, 2.1589887]])
    OPT_DOF_POS = np.array(
        [[0.21041952, 1.1591719, 0.7177322, 0.72911245, 0.9791311, 1.1570634, 0.9384481, 0.54952466, 1.2473342, 0.5, 0.13280809, 0.0]]
    )
elif FRAME == 60:
    # current warm-started LH pkl, frame 60
    OPT_WRIST_POS = np.array([[-0.6611467, -0.04048292, 0.23790106]])
    OPT_WRIST_ROT_AA = np.array([[-0.9648747, -0.27790284, 2.1279092]])
    OPT_DOF_POS = np.array(
        [[0.51108205, 1.0738748, 0.84621894, 1.7, 1.7, 1.0746402, 1.4591666, 1.373901, 0.5559118, 0.46227327, 0.03305111, 0.0]]
    )
elif FRAME == 160:
    # current warm-started LH pkl, frame 160
    OPT_WRIST_POS = np.array([[-0.64672154, -0.09902698, 0.2586794]])
    OPT_WRIST_ROT_AA = np.array([[2.6855206, -0.54777396, -2.600332]])
    OPT_DOF_POS = np.array(
        [[0.4525144, 0.83995014, 0.8468586, 0.52427745, 0.59264797, 1.4703153, 1.1377494, 0.4624444, 1.3, 0.4154378, 0.08087854, 0.0]]
    )
else:
    raise ValueError(f"no hardcoded pose for frame {FRAME}")


def main():
    device = "cuda:0"
    dexhand = DexHandFactory.create_hand("inspire", "left")
    dataset_type = ManipDataFactory.dataset_type("t3")
    demo_d = ManipDataFactory.create_data(
        manipdata_type=dataset_type, side="left", device=device,
        mujoco2gym_transf=torch.eye(4, device=device), dexhand=dexhand, verbose=False,
    )
    demo_data = pack_data([demo_d["t3"]], dexhand)

    # mujoco2gym_transf used by the real fitting()/env pipeline (table transform)
    mujoco2gym_transf = np.eye(4)
    mujoco2gym_transf[:3, :3] = aa_to_rotmat(np.array([0, 0, -np.pi / 2])) @ aa_to_rotmat(np.array([np.pi / 2, 0, 0]))
    mujoco2gym_transf[:3, 3] = np.array([0, 0, 0.415])
    M = torch.tensor(mujoco2gym_transf, dtype=torch.float32, device=device)

    obj_trajectory = (M @ demo_data["obj_trajectory"][FRAME])
    obj_R = obj_trajectory[:3, :3].cpu().numpy()
    obj_t = obj_trajectory[:3, 3].cpu().numpy()

    # MANO reference fingertip positions (world, transformed) for this frame
    mano_joints_flat = demo_data["mano_joints"][FRAME].view(-1, 3)
    mano_joints_world = (M[:3, :3] @ mano_joints_flat.T).T + M[:3, 3]
    mano_joints_world = mano_joints_world.cpu().numpy()
    # order matches dexhand.body_names (excluding wrist) -- thumb_tip is last of thumb chain etc;
    # just use the mean of all MANO joints as a reference "which side" anchor along the plate's
    # thin axis, robust to exact indexing.
    mano_local = (mano_joints_world - obj_t) @ obj_R  # (N,3)

    # plate's thin axis = smallest-extent principal axis of the collision mesh, in ITS OWN
    # local frame (same frame obj_R/obj_t transform world points INTO)
    mesh = trimesh.load("data/taco/urdfs/166/collision.obj", process=False, force="mesh")
    extent = mesh.bounds[1] - mesh.bounds[0]
    thin_axis = int(np.argmin(extent))
    mesh_center = (mesh.bounds[1] + mesh.bounds[0]) / 2
    print(f"plate mesh extent (local frame): {extent*1000} mm, thin_axis={thin_axis} ('x','y','z'[thin_axis])")
    print(f"plate mesh center (local frame): {mesh_center*1000} mm")

    mano_side = np.sign(mano_local[:, thin_axis] - mesh_center[thin_axis])
    print(f"\nMANO reference joints' side (thin axis={thin_axis}): {mano_side}  (mean={mano_side.mean():.3f})")

    # Now compute the OPTIMIZED (converged single-frame-100) link positions via the SAME FK
    # chain fitting() uses, and check their side.
    chain = pk.build_chain_from_urdf(open(dexhand.urdf_path).read()).to(dtype=torch.float32, device=device)
    isaac2chain_order = None  # not needed: we drive the chain directly with dof order = dof_names order
    dof_pos_t = torch.tensor(OPT_DOF_POS, dtype=torch.float32, device=device)
    wrist_pos_t = torch.tensor(OPT_WRIST_POS, dtype=torch.float32, device=device)

    # dof order for pk chain matches chain.get_joint_parameter_names(); dexhand dof_names order
    # may differ, but OPT_DOF_POS was produced by fitting() which uses isaac2chain_order to
    # reorder BEFORE calling chain.forward_kinematics -- since we don't have that mapping handy
    # here, and body-level SIGN (not exact position) is what we need, use the SAME mapping via
    # a fresh Mano2Dexhand-style order lookup instead of re-deriving isaac2chain_order.
    from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory as _DHF  # noqa
    dof_names_isaac_order = dexhand.dof_names
    chain_joint_names = chain.get_joint_parameter_names()
    isaac2chain_order = [dof_names_isaac_order.index(j) for j in chain_joint_names]
    ret = chain.forward_kinematics(dof_pos_t[:, isaac2chain_order])

    body_names = dexhand.body_names
    link_pos_chain = torch.stack([ret[k].get_matrix()[:, :3, 3] for k in body_names], dim=1)  # (1, nBody, 3)

    wrist_rot_t = torch.tensor(OPT_WRIST_ROT_AA, dtype=torch.float32, device=device)
    wrist_R = rot6d_to_rotmat(rotmat_to_rot6d(aa_to_rotmat(wrist_rot_t)))  # round-trip just for consistent dtype/device
    link_pos_world = (wrist_R @ link_pos_chain.transpose(-1, -2)).transpose(-1, -2) + torch.tensor(
        OPT_WRIST_POS, dtype=torch.float32, device=device
    )[:, None, :]
    link_pos_world = link_pos_world[0].cpu().numpy()

    link_local = (link_pos_world - obj_t) @ obj_R
    link_side = np.sign(link_local[:, thin_axis] - mesh_center[thin_axis])

    print(f"\nOptimized link positions' side (thin axis={thin_axis}), per body:")
    for name, side, pos in zip(body_names, link_side, link_local):
        flag = "  <-- OPPOSITE SIDE from MANO mean" if side != np.sign(mano_side.mean()) else ""
        print(f"  {name:22s} side={side:+.0f}  local_thin_axis_pos={pos[thin_axis]*1000:+7.2f}mm{flag}")


if __name__ == "__main__":
    main()
