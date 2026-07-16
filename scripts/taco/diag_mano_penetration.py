"""
Suspect 3: does TACO's own MANO reference hand penetrate the object mesh at
frame 100 (t3, RH/tool), or only the retargeted Inspire hand? MANO mesh
vertices are true skin-surface points (unlike the dexhand's skeletal joint
origins used in diag_reset_state.py's penetration dump, which don't account
for each link's own collision shape thickness) -- a cleaner containment test.

If MANO is clean and Inspire penetrates: retargeting introduces it
(mano2dexhand.py has no collision term, only minimizes fingertip distance).
If MANO ALSO penetrates: it's inherited from TACO's own hand/object data.

Must be run from the repo root: python scripts/taco/diag_mano_penetration.py
"""

from isaacgym import gymapi  # noqa: F401 -- must be first import; main.dataset auto-imports mano2dexhand.py

import numpy as np
import torch
import trimesh

from main.dataset.taco_dataset_dexhand import TACORightData, TACOLeftData, TACO_ROOT
from main.dataset.transform import aa_to_rotmat

device = "cuda:0"

mujoco2gym_transf = np.eye(4)
mujoco2gym_transf[:3, :3] = aa_to_rotmat(np.array([0, 0, -np.pi / 2])) @ aa_to_rotmat(np.array([np.pi / 2, 0, 0]))
mujoco2gym_transf[:3, 3] = np.array([0, 0, 0.415])
mujoco2gym_transf_t = torch.tensor(mujoco2gym_transf, dtype=torch.float32, device=device)

from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory
import maniptrans_envs.lib.envs.dexhands  # noqa: F401

dexhand_rh = DexHandFactory.create_hand("inspire", "right")
dexhand_lh = DexHandFactory.create_hand("inspire", "left")

fdata_rh = TACORightData(mujoco2gym_transf=mujoco2gym_transf_t, device=device, dexhand=dexhand_rh)
fdata_lh = TACOLeftData(mujoco2gym_transf=mujoco2gym_transf_t, device=device, dexhand=dexhand_lh)
data_rh = fdata_rh["t3"]
data_lh = fdata_lh["t3"]

FRAME = 100

for side, data, obj_side_name in [("RH/tool", data_rh, "rh"), ("LH/target", data_lh, "lh")]:
    obj_id = data["obj_id"]
    obj_pos = data["obj_trajectory"][FRAME, :3, 3].detach().cpu().numpy()
    obj_rotmat = data["obj_trajectory"][FRAME, :3, :3].detach().cpu().numpy()

    visual = trimesh.load(f"{TACO_ROOT}/meshes_m/{obj_id}_m.obj", process=False, force="mesh")
    coll = trimesh.load(f"{TACO_ROOT}/urdfs/{obj_id}/collision.obj", process=False, force="mesh")

    # MANO fingertip joints (mano_joints dict, world frame at FRAME) -- transform to object-local
    tip_names = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]
    mano_tips_world = np.stack([data["mano_joints"][n][FRAME].detach().cpu().numpy() for n in tip_names])
    mano_tips_local = (mano_tips_world - obj_pos) @ obj_rotmat  # local = R^T @ (world - pos), R orthonormal

    vis_dist = trimesh.proximity.signed_distance(visual, mano_tips_local)
    coll_dist = trimesh.proximity.signed_distance(coll, mano_tips_local)

    print(f"=== {side} (obj {obj_id}), frame {FRAME}: MANO reference fingertip penetration ===")
    for i, name in enumerate(tip_names):
        print(f"  {name:12s}: visual_signed_dist={vis_dist[i]*1000:+7.2f}mm   collision_signed_dist={coll_dist[i]*1000:+7.2f}mm")

    # also check the FULL MANO hand mesh (778 verts) for any deeper penetration than just the 5 tips
    mano_all_verts_world = torch.stack(
        [data["mano_joints"][k][FRAME] for k in data["mano_joints"].keys()]
    ).detach().cpu().numpy()  # (20, 3) -- all 20 named joints (proximal/intermediate/distal/tip x 5 fingers)
    mano_all_local = (mano_all_verts_world - obj_pos) @ obj_rotmat
    vis_dist_all = trimesh.proximity.signed_distance(visual, mano_all_local)
    coll_dist_all = trimesh.proximity.signed_distance(coll, mano_all_local)
    max_pen_vis = vis_dist_all.max()
    max_pen_coll = coll_dist_all.max()
    print(f"  ALL 20 MANO joints: max visual penetration={max_pen_vis*1000:+.2f}mm   max collision penetration={max_pen_coll*1000:+.2f}mm")
    print()
