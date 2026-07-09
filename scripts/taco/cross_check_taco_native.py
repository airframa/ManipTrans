"""
Isolates whether TACO's large fingertip-to-object gap (found in the M4
debugging session) is (a) a real property of TACO's data, or (b) a bug in
our TACODataBase pipeline. Loads the same spoon/bowl sequence using TACO's
OWN reference code (manopth, from the TACO-Instructions clone) -- completely
independent of our manotorch-based TACODataBase -- and computes the same
fingertip-to-object-surface distance metric.

Must be run from the repo root: python scripts/taco/cross_check_taco_native.py
"""

import sys

sys.path.insert(0, "/home/fmb/projects/TACO-Instructions/dataset_utils")

import numpy as np
import torch
import trimesh

from manopth.manopth.manolayer import ManoLayer as ManopthManoLayer

TRIPLET = "(put in, spoon, bowl)"
SEQ_NAME = "20231104_179"
TOOL_ID = "198"  # spoon
TARGET_ID = "180"  # bowl
FINGERTIP_VERTEX_IDS = {"index": 353, "middle": 467, "pinky": 695, "ring": 576, "thumb": 766}

DATA_ROOT = "/home/fmb/projects/ManipTrans/data/taco"
MANO_ROOT = "/home/fmb/projects/ManipTrans/data/mano_v1_2/models"


def load_hand_native(side, obj_id_role):
    import pickle

    hand_dir = f"{DATA_ROOT}/Hand_Poses/{TRIPLET}/{SEQ_NAME}"
    with open(f"{hand_dir}/{side}_hand.pkl", "rb") as f:
        hand_data = pickle.load(f)
    with open(f"{hand_dir}/{side}_hand_shape.pkl", "rb") as f:
        beta = pickle.load(f)["hand_shape"].numpy().astype(np.float32)

    frame_keys = sorted(hand_data.keys())
    theta = np.stack([hand_data[k]["hand_pose"].numpy() for k in frame_keys]).astype(np.float32)
    trans = np.stack([hand_data[k]["hand_trans"].numpy() for k in frame_keys]).astype(np.float32)

    device = "cuda:0"
    mano_layer = ManopthManoLayer(
        mano_root=MANO_ROOT, use_pca=False, ncomps=45, side="right" if side == "right" else "left", center_idx=0
    ).to(device)

    theta_t = torch.tensor(theta, device=device)
    trans_t = torch.tensor(trans, device=device)
    beta_t = torch.tensor(beta, device=device).unsqueeze(0).repeat(theta_t.shape[0], 1)

    verts, joints, _ = mano_layer(theta_t, beta_t)
    verts = verts / 1000.0 + trans_t.unsqueeze(1)  # TACO's own convention: mm -> m, then add trans
    return verts.detach().cpu().numpy(), frame_keys


def load_obj_native(role, obj_id):
    obj_dir = f"{DATA_ROOT}/Object_Poses/{TRIPLET}/{SEQ_NAME}"
    traj = np.load(f"{obj_dir}/{role}_{obj_id}.npy").astype(np.float32)  # (T,4,4), already meters
    mesh = trimesh.load(f"{DATA_ROOT}/object_models_released/{obj_id}_cm.obj", process=False, force="mesh", skip_materials=True)
    verts_m = (mesh.vertices * 0.01).astype(np.float32)  # cm -> m, TACO's own documented convention
    return traj, verts_m


def nearest_dist(points, surface_pts):
    # points: (5,3) fingertips, surface_pts: (N,3) object surface samples
    d = np.linalg.norm(points[:, None, :] - surface_pts[None, :, :], axis=-1)  # (5, N)
    return d.min(axis=-1)  # (5,)


def main():
    print(f"Sequence: {TRIPLET}/{SEQ_NAME}  tool={TOOL_ID}(spoon)  target={TARGET_ID}(bowl)")

    right_verts, frame_keys = load_hand_native("right", "tool")
    T = right_verts.shape[0]
    print(f"T={T} frames (native 30Hz, no resampling -- this cross-check doesn't need it)")

    tool_traj, tool_verts_local = load_obj_native("tool", TOOL_ID)
    assert tool_traj.shape[0] == T

    tip_idx = list(FINGERTIP_VERTEX_IDS.values())
    tip_names = list(FINGERTIP_VERTEX_IDS.keys())

    all_dists = []
    for t in range(T):
        fingertips = right_verts[t, tip_idx]  # (5,3)
        R, tvec = tool_traj[t, :3, :3], tool_traj[t, :3, 3]
        obj_surface_world = (R @ tool_verts_local.T).T + tvec  # (Nverts,3)
        # subsample surface for speed
        idx = np.random.RandomState(0).choice(len(obj_surface_world), size=min(2000, len(obj_surface_world)), replace=False)
        d = nearest_dist(fingertips, obj_surface_world[idx])
        all_dists.append(d)
    all_dists = np.stack(all_dists)  # (T, 5)

    print("\n=== TACO's OWN pipeline (manopth), RIGHT hand vs tool (spoon), fingertip-to-surface distance ===")
    print(f"  mean={all_dists.mean()*100:.2f}cm  min={all_dists.min()*100:.2f}cm  max={all_dists.max()*100:.2f}cm  frame0={all_dists[0].mean()*100:.2f}cm")
    for i, name in enumerate(tip_names):
        print(f"  {name:8s} tip: mean={all_dists[:,i].mean()*100:.2f}cm  min={all_dists[:,i].min()*100:.2f}cm")


if __name__ == "__main__":
    main()
