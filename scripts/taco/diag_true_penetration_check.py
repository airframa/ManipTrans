"""
Offline (no isaacgym needed) mesh-vs-mesh penetration check: loads the
ACTUAL Inspire hand collision STL for each finger's last collision-bearing
phalanx (thumb_distal, {index,middle,ring,pinky}_intermediate -- the
fingertip "tip" links are visual-only 5mm spheres with NO collision
geometry, so they were never the real contact surface), transforms each
into world space using the dumped body pose, and computes penetration
depth (max signed distance of any phalanx vertex into the object mesh)
against the object's actual collision mesh.

Consumes .npz files produced by diag_true_penetration_dump.py.

Usage: python scripts/taco/diag_true_penetration_check.py <dump.npz> <dataset: taco|oakink2>
"""

import sys

import numpy as np
import trimesh

INSPIRE_ROOT = "maniptrans_envs/assets/inspire_hand"

STL_MAP_R = {
    "palm": "R_hand_base_link.STL",
    "thumb_base": "Link11_R.STL",
    "thumb_proximal": "Link12_R.STL",
    "thumb_intermediate": "Link13_R.STL",
    "thumb_distal": "Link14_R.STL",
    "index_proximal": "Link15_R.STL",
    "index_intermediate": "Link16_R.STL",
    "middle_proximal": "Link17_R.STL",
    "middle_intermediate": "Link18_R.STL",
    "ring_proximal": "Link19_R.STL",
    "ring_intermediate": "Link20_R.STL",
    "pinky_proximal": "Link21_R.STL",
    "pinky_intermediate": "Link22_R.STL",
}
STL_MAP_L = {
    "palm": "L_hand_base_link.STL",
    "thumb_base": "Link11_L.STL",
    "thumb_proximal": "Link12_L.STL",
    "thumb_intermediate": "Link13_L.STL",
    "thumb_distal": "Link14_L.STL",
    "index_proximal": "Link15_L.STL",
    "index_intermediate": "Link16_L.STL",
    "middle_proximal": "Link17_L.STL",
    "middle_intermediate": "Link18_L.STL",
    "ring_proximal": "Link19_L.STL",
    "ring_intermediate": "Link20_L.STL",
    "pinky_proximal": "Link21_L.STL",
    "pinky_intermediate": "Link22_L.STL",
}


def quat_to_R(q):  # xyzw
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def main():
    npz_path = sys.argv[1]
    dataset = sys.argv[2]  # "taco" or "oakink2"
    d = np.load(npz_path, allow_pickle=True)

    for side, stl_map in [("rh", STL_MAP_R), ("lh", STL_MAP_L)]:
        obj_id = str(d[f"{side}_obj_id"])
        obj_pos = d[f"{side}_obj_pos"]
        obj_quat = d[f"{side}_obj_quat"]
        R_obj = quat_to_R(obj_quat)

        if dataset == "taco":
            visual_path = f"data/taco/meshes_m/{obj_id}_m.obj"
            coll_path = f"data/taco/urdfs/{obj_id}/collision.obj"
        else:
            visual_path = f"data/OakInk-v2/coacd_object_preview/align_ds/{obj_id}/scan.ply"
            coll_path = visual_path  # OakInk-V2 URDF uses the same file for visual and collision

        visual = trimesh.load(visual_path, process=False, force="mesh")
        coll = trimesh.load(coll_path, process=False, force="mesh")
        if not coll.is_watertight:
            coll = coll.convex_hull  # fallback so signed_distance is still meaningful

        print(f"\n=== {dataset} {side} object {obj_id} ===")
        print(f"  visual mesh: verts={len(visual.vertices)} watertight={visual.is_watertight}")
        print(f"  collision mesh: verts={len(coll.vertices)} watertight={coll.is_watertight}")

        for finger, stl_name in stl_map.items():
            pos = d[f"{side}_{finger}_pos"]
            quat = d[f"{side}_{finger}_quat"]
            R_link = quat_to_R(quat)

            phalanx = trimesh.load(f"{INSPIRE_ROOT}/meshes/{stl_name}", force="mesh")
            verts_world = phalanx.vertices @ R_link.T + pos  # link-local -> world
            verts_obj_local = (verts_world - obj_pos) @ R_obj  # world -> object-local

            vis_sd = trimesh.proximity.signed_distance(visual, verts_obj_local)
            coll_sd = trimesh.proximity.signed_distance(coll, verts_obj_local)

            print(
                f"  {finger:8s} ({stl_name:14s} {len(phalanx.vertices):5d} verts): "
                f"max_visual_penetration={vis_sd.max()*1000:+7.2f}mm   max_collision_penetration={coll_sd.max()*1000:+7.2f}mm"
            )


if __name__ == "__main__":
    main()
