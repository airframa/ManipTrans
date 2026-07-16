"""
Builds an anisotropic SDF grid for a TACO object's COACD collision mesh at
a resolution matched to the object's aspect ratio (not a wasteful cubic
grid on a 4:1 object), sized well below the ~2mm penetration signal we're
trying to resolve. Then validates it against known ground-truth
measurements (direct STL-vertex proximity queries, already trusted from
the earlier investigation) before it's used for anything.

Fast signed-distance approximation (trimesh's native signed_distance is
~2.6k pts/sec -- far too slow for a fine dense grid): unsigned distance via
scipy cKDTree against a dense mesh-surface point sample (~60k pts/sec), sign
via trimesh's mesh.contains() (~500k pts/sec, ray-based, exact for
watertight meshes). This is a standard, fast approximation whose error is
bounded by the surface sample density, which we set far finer than the
target grid resolution.

Must be run from the repo root: python scripts/taco/build_and_validate_sdf.py <obj_id> [voxel_mm] [pad_mm]
"""

import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from scipy.spatial import cKDTree

TACO_ROOT = "data/taco"


def build_sdf_grid(obj_id, voxel_mm=0.4, pad_mm=15.0, target_sample_spacing_mm=0.1, local_center=None, local_halfwidth_mm=None):
    coll_path = f"{TACO_ROOT}/urdfs/{obj_id}/collision.obj"
    mesh = trimesh.load(coll_path, process=False, force="mesh")
    assert mesh.is_watertight, f"{obj_id} collision mesh not watertight -- contains() sign would be unreliable"

    # KDTree-vs-mesh-surface distance is a biased OVERESTIMATE when the surface sample is too
    # sparse relative to local curvature -- the underestimated-penetration failure we hit on
    # the plate (larger area, same fixed sample count as the knife -> coarser spacing) came
    # from exactly this. Scale sample count to the mesh's actual surface area so spacing stays
    # well below both the target grid voxel size and the ~1mm penetration signal.
    target_spacing = target_sample_spacing_mm / 1000.0
    n_surface_samples = max(int(mesh.area / target_spacing**2), 50_000)
    print(f"  mesh area={mesh.area*1e4:.1f}cm^2 -> {n_surface_samples:,} surface samples for ~{target_sample_spacing_mm}mm spacing")

    voxel = voxel_mm / 1000.0
    if local_center is not None:
        # Narrow-band / local-grid fallback for objects whose full bounding box is too large
        # to cover at fine resolution (e.g. the plate: 24x24cm footprint makes a full-object
        # grid at <0.4mm impractical). Since retargeting fits a KNOWN region (targets are
        # given, not searched over unboundedly), a grid scoped to the grasp region only is
        # sufficient and much cheaper -- viable specifically because the object pose is fixed
        # per fitting frame.
        halfwidth = np.array(local_halfwidth_mm) / 1000.0
        lo = np.array(local_center) - halfwidth
        hi = np.array(local_center) + halfwidth
    else:
        pad = pad_mm / 1000.0
        lo = mesh.bounds[0] - pad
        hi = mesh.bounds[1] + pad
    extent = hi - lo
    dims = np.maximum(np.ceil(extent / voxel).astype(int), 2)  # anisotropic: independent per axis

    mem_mb = dims.prod() * 4 / 1e6
    print(f"obj {obj_id}: mesh extent={ (mesh.bounds[1]-mesh.bounds[0])*1000 } mm, padded grid extent={extent*1000} mm")
    print(f"  voxel={voxel_mm}mm  grid dims={dims.tolist()}  total voxels={dims.prod():,}  memory={mem_mb:.1f}MB")

    xs = np.linspace(lo[0], hi[0], dims[0])
    ys = np.linspace(lo[1], hi[1], dims[1])
    zs = np.linspace(lo[2], hi[2], dims[2])
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    grid_pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1).astype(np.float32)
    print(f"  querying {len(grid_pts):,} grid points...")

    t0 = time.time()
    surf_pts, _ = trimesh.sample.sample_surface(mesh, n_surface_samples)
    tree = cKDTree(surf_pts)
    unsigned_dist, _ = tree.query(grid_pts, workers=-1)
    print(f"  KDTree unsigned distance: {time.time()-t0:.1f}s")

    t0 = time.time()
    inside = mesh.contains(grid_pts)
    print(f"  contains() sign: {time.time()-t0:.1f}s")

    sdf = np.where(inside, unsigned_dist, -unsigned_dist).astype(np.float32).reshape(dims)
    return sdf, lo, hi, dims, voxel, mesh


def query_sdf(sdf_t, lo, hi, dims, world_pts_t, device):
    # world_pts_t: (N,3) torch tensor, object-local frame (already), meters
    # normalize to [-1, 1] for grid_sample, per-axis (extent differs per axis -- anisotropic)
    lo_t = torch.tensor(lo, dtype=torch.float32, device=device)
    extent_t = torch.tensor(hi - lo, dtype=torch.float32, device=device)
    norm = 2.0 * (world_pts_t - lo_t) / extent_t - 1.0
    # grid_sample 3D: input (N,C,D,H,W), grid (N,D_out,H_out,W_out,3) with LAST-DIM order (x,y,z) mapping to (W,H,D)
    # i.e. grid[...,0]=W-axis coord, grid[...,1]=H-axis coord, grid[...,2]=D-axis coord.
    # Our sdf array axes are (dim0=X, dim1=Y, dim2=Z) matching dims=[nx,ny,nz] built via
    # meshgrid(xs,ys,zs, indexing='ij') -- so array axis0=X=D, axis1=Y=H, axis2=Z=W in
    # grid_sample's (D,H,W) convention. Grid coord order must then be (z,y,x) to match (W,H,D).
    vol = sdf_t[None, None]  # (1,1,D,H,W) = (1,1,nx,ny,nz)
    grid = norm[None, None, None, :, :]  # (1,1,1,N,3)
    grid = grid[..., [2, 1, 0]]  # reorder (x,y,z) -> (z,y,x) to match grid_sample's (W,H,D) <- (x,y,z) convention
    out = F.grid_sample(vol, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return out.view(-1)


def main():
    local_mode = "--local" in sys.argv
    positional = [a for a in sys.argv[1:] if not a.startswith("--")]
    obj_id = positional[0]
    voxel_mm = float(positional[1]) if len(positional) > 1 else 0.4
    pad_mm = float(positional[2]) if len(positional) > 2 else 15.0

    local_center, local_halfwidth_mm = None, None
    if local_mode:
        # Center the local grid on the known grasp-contact cluster (mean of the 3 ground-truth
        # link positions, transformed into object-local frame) with a generous halfwidth --
        # covers the whole hand's grasp region for this frame, not just the single test points.
        d = np.load("/tmp/taco_t3_truepen_full.npz", allow_pickle=True)
        side = "rh" if obj_id == "063" else "lh"

        def quat_to_R(q):
            x, y, z, w = q
            return np.array(
                [
                    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                ]
            )

        obj_pos = d[f"{side}_obj_pos"]
        obj_quat = d[f"{side}_obj_quat"]
        R_obj = quat_to_R(obj_quat)
        all_link_pos_world = np.stack(
            [d[f"{side}_{k}_pos"] for k in ["thumb_distal", "thumb_intermediate", "index_intermediate", "ring_intermediate"]]
        )
        all_link_pos_local = (all_link_pos_world - obj_pos) @ R_obj
        local_center = all_link_pos_local.mean(axis=0)
        local_halfwidth_mm = [25.0, 25.0, 25.0]
        print(f"Local grid mode: center={local_center*1000} mm  halfwidth={local_halfwidth_mm} mm")

    sdf, lo, hi, dims, voxel, mesh = build_sdf_grid(
        obj_id, voxel_mm, pad_mm, local_center=local_center, local_halfwidth_mm=local_halfwidth_mm
    )

    cache_path = f"{TACO_ROOT}/urdfs/{obj_id}/sdf_{voxel_mm}mm{'_local' if local_mode else ''}.npz"
    np.savez(cache_path, sdf=sdf, lo=lo, hi=hi, voxel_mm=voxel_mm)
    print(f"Cached SDF to {cache_path}")

    # --- VALIDATION GATE: reproduce known ground-truth values from the earlier STL-vertex
    # test (diag_true_penetration_check.py, TACO t3 RH, frame 100, collision mesh):
    #   thumb_distal max penetration  = +2.20mm
    #   ring_intermediate max pen.    = +2.45mm
    #   index_intermediate max pen.   = -10.65mm
    GROUND_TRUTH = {
        "063": {
            "side": "rh",
            "expected": {"thumb_distal": +2.20, "ring_intermediate": +2.45, "index_intermediate": -10.65},
            "stl_map": {"thumb_distal": "Link14_R.STL", "ring_intermediate": "Link20_R.STL", "index_intermediate": "Link16_R.STL"},
        },
        "166": {
            "side": "lh",
            "expected": {"thumb_distal": +5.04, "thumb_intermediate": +1.19, "index_intermediate": -36.93},
            "stl_map": {"thumb_distal": "Link14_L.STL", "thumb_intermediate": "Link13_L.STL", "index_intermediate": "Link16_L.STL"},
        },
    }

    if obj_id in GROUND_TRUTH:
        d = np.load("/tmp/taco_t3_truepen_full.npz", allow_pickle=True)
        gt = GROUND_TRUTH[obj_id]
        side = gt["side"]

        def quat_to_R(q):
            x, y, z, w = q
            return np.array(
                [
                    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                ]
            )

        obj_pos = d[f"{side}_obj_pos"]
        obj_quat = d[f"{side}_obj_quat"]
        R_obj = quat_to_R(obj_quat)

        device = "cpu"
        sdf_t = torch.tensor(sdf, device=device)

        print(f"\n=== VALIDATION GATE: grid-recovered vs known ground truth (t3 {side.upper()}, frame 100, obj {obj_id}) ===")
        all_pass = True
        for link, exp_mm in gt["expected"].items():
            if local_mode and exp_mm < -20:
                print(f"  {link:20s}: expected={exp_mm:+.2f}mm  SKIPPED (far-field, outside local grid bounds by design -- "
                      f"a one-sided hinge collision loss zeroes this regime out regardless of exact value)")
                continue
            pos = d[f"{side}_{link}_pos"]
            quat = d[f"{side}_{link}_quat"]
            R_link = quat_to_R(quat)
            phalanx = trimesh.load(f"maniptrans_envs/assets/inspire_hand/meshes/{gt['stl_map'][link]}", force="mesh")
            verts_world = phalanx.vertices @ R_link.T + pos
            verts_obj_local = (verts_world - obj_pos) @ R_obj

            pts_t = torch.tensor(verts_obj_local, dtype=torch.float32, device=device)
            sdf_vals = query_sdf(sdf_t, lo, hi, dims, pts_t, device).numpy()
            got_mm = sdf_vals.max() * 1000
            diff = abs(got_mm - exp_mm)
            status = "PASS" if diff <= 0.2 else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(f"  {link:20s}: expected={exp_mm:+.2f}mm  grid_recovered={got_mm:+.2f}mm  diff={diff:.3f}mm  [{status}]")

        print(f"\nVALIDATION GATE: {'PASSED' if all_pass else 'FAILED'} (tolerance 0.2mm)")


if __name__ == "__main__":
    main()
