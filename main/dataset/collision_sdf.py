"""
Precomputed anisotropic SDF grids for TACO object collision meshes, used by
mano2dexhand.py's optional collision-penetration term. See
scripts/taco/build_and_validate_sdf.py for the original design/validation
work this was extracted from (full derivation, sign convention, and the
axis-order unit test live there).

Sign convention: SDF > 0 means INSIDE the mesh (penetrating), matching
trimesh.proximity.signed_distance. Grid built via fast KDTree-unsigned-distance
(against a dense mesh-surface sample) + mesh.contains() sign, NOT trimesh's
native signed_distance (too slow for a fine grid: ~2.6k pts/sec vs ~60k/sec).
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from scipy.spatial import cKDTree


def _build_sdf_grid(collision_obj_path, voxel_mm, pad_mm, target_sample_spacing_mm, local_center, local_halfwidth_mm):
    mesh = trimesh.load(collision_obj_path, process=False, force="mesh")
    assert mesh.is_watertight, f"{collision_obj_path}: not watertight, mesh.contains() sign would be unreliable"

    target_spacing = target_sample_spacing_mm / 1000.0
    n_surface_samples = max(int(mesh.area / target_spacing**2), 50_000)

    voxel = voxel_mm / 1000.0
    if local_center is not None:
        halfwidth = np.array(local_halfwidth_mm) / 1000.0
        lo = np.array(local_center) - halfwidth
        hi = np.array(local_center) + halfwidth
    else:
        pad = pad_mm / 1000.0
        lo = mesh.bounds[0] - pad
        hi = mesh.bounds[1] + pad
    extent = hi - lo
    dims = np.maximum(np.ceil(extent / voxel).astype(int), 2)

    xs = np.linspace(lo[0], hi[0], dims[0])
    ys = np.linspace(lo[1], hi[1], dims[1])
    zs = np.linspace(lo[2], hi[2], dims[2])
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    grid_pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1).astype(np.float32)

    surf_pts, _ = trimesh.sample.sample_surface(mesh, n_surface_samples)
    tree = cKDTree(surf_pts)
    unsigned_dist, _ = tree.query(grid_pts, workers=-1)
    inside = mesh.contains(grid_pts)
    sdf = np.where(inside, unsigned_dist, -unsigned_dist).astype(np.float32).reshape(dims)
    return sdf, lo.astype(np.float32), hi.astype(np.float32), dims


def load_or_build_object_sdf(
    obj_id,
    taco_root="data/taco",
    voxel_mm=0.4,
    pad_mm=15.0,
    target_sample_spacing_mm=0.1,
    local_center=None,
    local_halfwidth_mm=25.0,
    device="cuda:0",
):
    """Returns (sdf_tensor, lo, hi) -- lo/hi are numpy (3,) grid bounds in object-local meters.

    If local_center is given (object-local-frame point, meters), builds/caches a narrow local
    grid around it instead of covering the whole object -- necessary for objects whose full
    bounding box is too large to cover at fine resolution (e.g. a wide flat plate); see
    scripts/taco/build_and_validate_sdf.py's plate case, where a full-object grid at the
    required resolution needed ~250M+ voxels and still failed the accuracy validation gate,
    while a 25mm-halfwidth local grid at 0.15mm passed cleanly.
    """
    local_mode = local_center is not None
    suffix = "_local" if local_mode else ""
    cache_path = os.path.join(taco_root, "urdfs", obj_id, f"sdf_{voxel_mm}mm{suffix}.npz")

    if local_mode:
        # local grids are frame-specific (center depends on the grasp region for that
        # particular fitting problem) -- don't reuse a stale cache from a different frame/pose.
        cached = None
    elif os.path.exists(cache_path):
        cached = np.load(cache_path)
    else:
        cached = None

    if cached is not None:
        sdf, lo, hi = cached["sdf"], cached["lo"], cached["hi"]
    else:
        collision_obj_path = os.path.join(taco_root, "urdfs", obj_id, "collision.obj")
        halfwidth = [local_halfwidth_mm] * 3 if np.isscalar(local_halfwidth_mm) else local_halfwidth_mm
        sdf, lo, hi, dims = _build_sdf_grid(
            collision_obj_path, voxel_mm, pad_mm, target_sample_spacing_mm, local_center, halfwidth
        )
        if not local_mode:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.savez(cache_path, sdf=sdf, lo=lo, hi=hi, voxel_mm=voxel_mm)

    sdf_t = torch.tensor(sdf, dtype=torch.float32, device=device)
    return sdf_t, lo, hi


def query_sdf(sdf_t, lo, hi, world_pts_local_frame):
    """world_pts_local_frame: (..., 3) tensor, OBJECT-LOCAL frame, meters. Returns (...) SDF values.

    Differentiable w.r.t. world_pts_local_frame via F.grid_sample's bilinear (here: trilinear,
    3D) interpolation. See scripts/taco/build_and_validate_sdf.py for the axis-order unit test
    this mapping was validated against (grid_sample's (N,C,D,H,W)/(x,y,z)->(W,H,D) convention
    is a classic silent-transpose-bug source).
    """
    device = sdf_t.device
    lo_t = torch.tensor(lo, dtype=torch.float32, device=device)
    extent_t = torch.tensor(hi - lo, dtype=torch.float32, device=device)
    orig_shape = world_pts_local_frame.shape[:-1]
    pts_flat = world_pts_local_frame.reshape(-1, 3)
    norm = 2.0 * (pts_flat - lo_t) / extent_t - 1.0

    vol = sdf_t[None, None]  # (1,1,D,H,W) = (1,1,nx,ny,nz)
    grid = norm[None, None, None, :, :]  # (1,1,1,N,3), last-dim order currently (x,y,z)
    grid = grid[..., [2, 1, 0]]  # reorder to (z,y,x) to match grid_sample's (W,H,D) <- (x,y,z)
    out = F.grid_sample(vol, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return out.view(*orig_shape)


COLLISION_LINK_STL = {
    "hand_base_link": "{side}_hand_base_link.STL",
    "thumb_proximal_base": "Link11_{side}.STL",
    "thumb_proximal": "Link12_{side}.STL",
    "thumb_intermediate": "Link13_{side}.STL",
    "thumb_distal": "Link14_{side}.STL",
    "index_proximal": "Link15_{side}.STL",
    "index_intermediate": "Link16_{side}.STL",
    "middle_proximal": "Link17_{side}.STL",
    "middle_intermediate": "Link18_{side}.STL",
    "ring_proximal": "Link19_{side}.STL",
    "ring_intermediate": "Link20_{side}.STL",
    "pinky_proximal": "Link21_{side}.STL",
    "pinky_intermediate": "Link22_{side}.STL",
}


def sample_link_points(inspire_root, side_letter, n_points_per_link=25, device="cuda:0"):
    """Precompute a fixed set of surface sample points per collision-bearing link, in the
    link's own local frame (meters). side_letter: "R" or "L".

    Excludes *_tip links deliberately -- they're visual-only 5mm sphere markers with NO
    <collision> element in the Inspire URDF; the real contact surface for each fingertip is
    the last real phalanx (thumb_distal, {index,middle,ring,pinky}_intermediate).
    """
    points = {}
    for link_key, stl_tpl in COLLISION_LINK_STL.items():
        stl_path = os.path.join(inspire_root, "meshes", stl_tpl.format(side=side_letter))
        mesh = trimesh.load(stl_path, force="mesh")
        pts, _ = trimesh.sample.sample_surface(mesh, n_points_per_link)
        points[link_key] = torch.tensor(pts, dtype=torch.float32, device=device)
    return points
