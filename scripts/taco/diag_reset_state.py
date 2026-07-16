"""
Decisive reset-state comparison (Step 4 of the initial-state investigation):
for the SAME controlled start frame, build the REAL DexHandManipBiHEnv, call
reset(), and read the reset state directly off the simulated actors (no RL
policy, no pre_physics_step action routing) -- then step raw physics
(env.gym.simulate(), like physics_sanity_check.py) with NO action for 8
steps, re-reading state each step. Compares TACO (t3) vs OakInk-V2 (20aed@0):

  1. INITIAL GRIP CONSISTENCY: min fingertip-to-object-surface distance at
     reset, per hand (chamfer-style, using the SIMULATED object pose and
     the dataset's own sampled point cloud -- self-consistent, not the
     static reference).
  2. INITIAL VELOCITY: object linear/angular velocity magnitude seeded at
     reset (same code path for both datasets -- checking whether the DATA
     itself differs, not the seeding mechanism).
  3./4. Step physics 8x with zero action (position-control DOFs hold their
     reset targets via low-level PD control; only the object is free to
     respond to contact/gravity) and track object position/rotation drift
     from its reset pose, and tip-to-surface distance drift, side by side.

Must be run from the repo root: python scripts/taco/diag_reset_state.py <data_idx> [rolloutBegin]

Note on rolloutBegin: with randomStateInit=false and rolloutBegin unset,
seq_idx is forced to frame 0 of the sequence. For TACO that's a raw,
untrimmed "approach" frame (hand nowhere near the object yet) -- NOT
representative of a typical mid-motion grip a real randomStateInit=true
training reset would sample. Pass an explicit rolloutBegin to pick a
mid-sequence frame for a fair, representative comparison.
"""

from isaacgym import gymapi  # noqa: F401 -- must be first import in the whole process

import os
import sys

import hydra
import numpy as np
import torch
from isaacgym.torch_utils import quat_rotate
from omegaconf import OmegaConf

import lib  # noqa: F401
import maniptrans_envs.lib  # noqa: F401

TIP_NAMES = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]


def tip_surface_distances(env, side, dexhand):
    states = getattr(env, f"{side}_states")
    demo_data = getattr(env, f"demo_data_{side}")

    obj_verts_local = demo_data["obj_verts"]  # (nE, 1000, 3), object-local frame
    obj_pos = states["manip_obj_pos"]  # (nE, 3)
    obj_quat = states["manip_obj_quat"]  # (nE, 4) xyzw

    nE, nPts, _ = obj_verts_local.shape
    obj_quat_exp = obj_quat[:, None, :].expand(nE, nPts, 4).reshape(-1, 4)
    obj_verts_flat = obj_verts_local.reshape(-1, 3)
    obj_verts_world = quat_rotate(obj_quat_exp, obj_verts_flat).reshape(nE, nPts, 3) + obj_pos[:, None, :]

    tip_body_idx = {
        name: dexhand.body_names.index([b for b in dexhand.body_names if b.endswith(name)][0]) for name in TIP_NAMES
    }
    joints_state = states["joints_state"]  # (nE, n_body, 10)

    dists = {}
    for name, idx in tip_body_idx.items():
        tip_pos = joints_state[:, idx, :3]  # (nE, 3)
        d = torch.norm(obj_verts_world - tip_pos[:, None, :], dim=-1)  # (nE, nPts)
        dists[name] = d.min(dim=-1).values  # (nE,)
    return dists


def main():
    data_idx = sys.argv[1]
    rollout_begin = sys.argv[2] if len(sys.argv) > 2 else None

    config_dir = os.path.abspath("main/cfg")
    overrides = [
        "task=ResDexHand",
        "dexhand=inspire",
        "side=BiH",
        "headless=true",
        "num_envs=2",  # num_envs=1 hits the pack_data() squeeze bug (see physics_sanity_check.py)
        "test=true",
        "randomStateInit=false",  # deterministic: every reset starts at the same frame
        f"dataIndices=[{data_idx}]",
        "rh_base_model_checkpoint=assets/imitator_ckp/imitator_rh_inspire.pth",
        "lh_base_model_checkpoint=assets/imitator_ckp/imitator_lh_inspire.pth",
        "experiment=diag_reset_state",
    ]
    if rollout_begin is not None:
        overrides.append(f"rolloutBegin={rollout_begin}")
    with hydra.initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        cfg = hydra.compose(config_name="config", overrides=overrides)

    env = maniptrans_envs.lib.make(
        sim_device="cuda:0",
        rl_device="cuda:0",
        graphics_device_id=0,
        cfg=cfg.task,
        display=False,
        record=False,
        has_headless_arg=True,
        headless=True,
    )

    print(f"=== data_idx={data_idx}  rolloutBegin={rollout_begin} ===")
    print(f"RH object mass (as simulated, post-clamp): {env.manip_obj_rh_mass[0].item()*1000:.2f} g")
    print(f"LH object mass (as simulated, post-clamp): {env.manip_obj_lh_mass[0].item()*1000:.2f} g")
    rh_props = env.gym.get_actor_rigid_body_properties(env.envs[0], env.obj_rh_handle)
    lh_props = env.gym.get_actor_rigid_body_properties(env.envs[0], env.obj_lh_handle)
    print(f"RH object inertia tensor (as simulated): {rh_props[0].inertia.x.x:.3e}, {rh_props[0].inertia.y.y:.3e}, {rh_props[0].inertia.z.z:.3e} (diag, kg*m^2)")
    print(f"LH object inertia tensor (as simulated): {lh_props[0].inertia.x.x:.3e}, {lh_props[0].inertia.y.y:.3e}, {lh_props[0].inertia.z.z:.3e} (diag, kg*m^2)")

    env.reset()

    # TRUE SEEDED VELOCITY: the object ROOT STATE (position/velocity) is directly settable
    # and readable via set/refresh_actor_root_state_tensor WITHOUT any simulate() call --
    # unlike the hand's RIGID BODY state (below), it does NOT need a physics step to reflect
    # what reset_idx() just wrote. Read this BEFORE any simulate() call, or we'd be reading
    # post-contact-response velocity (e.g. a de-penetration impulse) and mislabeling it as
    # "seeded".
    env.gym.refresh_actor_root_state_tensor(env.sim)
    true_seed_obj_vel_rh = env.rh_states.get("manip_obj_vel")
    # rh_states/lh_states dicts are only populated by _update_states(); call it once here
    # without any simulate() in between so the values reflect pure reset-time seeding.
    env._update_states()
    print("\n--- TRUE seeded velocity at reset (env 0, BEFORE any simulate() call) ---")
    print(f"  RH-side object: |lin_vel|={env.rh_states['manip_obj_vel'][0].norm().item():.4f} m/s  |ang_vel|={env.rh_states['manip_obj_ang_vel'][0].norm().item():.4f} rad/s")
    print(f"  LH-side object: |lin_vel|={env.lh_states['manip_obj_vel'][0].norm().item():.4f} m/s  |ang_vel|={env.lh_states['manip_obj_ang_vel'][0].norm().item():.4f} rad/s")

    # IMPORTANT: directly after gym.set_actor_root_state_tensor_indexed /
    # set_dof_state_tensor_indexed (inside reset()), the RIGID BODY state tensor (which
    # joints_state / fingertip world positions read from) has NOT yet been recomputed via
    # forward kinematics -- it still reflects whatever pose was there before reset. One
    # simulate()+fetch_results() cycle is required before the rigid body tensor reflects the
    # newly-set reset state. Without this, tip-distance numbers would be a stale-tensor
    # artifact. Settle once now -- but note this ALSO runs one physics step on the object, so
    # from here on "step 0" reflects post-physics state, not pure seeding (that's the point:
    # comparing pre- vs post-simulate velocity below isolates whether the drift is seeded or
    # a first-step contact-response impulse).
    env.gym.simulate(env.sim)
    env.gym.fetch_results(env.sim, True)
    env.gym.refresh_net_contact_force_tensor(env.sim)
    env._refresh()
    env._update_states()

    # Direct contact-manifold query: exact penetration depth Isaac Gym's own solver sees,
    # bypassing all mesh-proximity approximation (which can't account for each dexhand link's
    # own collision-shape thickness beyond its joint origin).
    contacts = env.gym.get_env_rigid_contacts(env.envs[0])
    if len(contacts) > 0:
        deepest = sorted(contacts, key=lambda c: c["separation"])[:8]
        print(f"\n--- Isaac Gym contact manifold after 1st simulate() (env 0, {len(contacts)} contacts total) ---")
        print("  (separation < 0 means real geometric penetration depth, in meters)")
        for c in deepest:
            print(f"  body0={c['body0']:4d} body1={c['body1']:4d} separation={c['separation']*1000:+8.3f}mm")
    else:
        print("\n--- Isaac Gym contact manifold after 1st simulate(): NO CONTACTS reported ---")

    print("\n--- velocity immediately AFTER the first simulate() call (env 0) ---")
    print(f"  RH-side object: |lin_vel|={env.rh_states['manip_obj_vel'][0].norm().item():.4f} m/s  |ang_vel|={env.rh_states['manip_obj_ang_vel'][0].norm().item():.4f} rad/s")
    print(f"  LH-side object: |lin_vel|={env.lh_states['manip_obj_vel'][0].norm().item():.4f} m/s  |ang_vel|={env.lh_states['manip_obj_ang_vel'][0].norm().item():.4f} rad/s")
    if hasattr(env, "_manip_obj_rh_cf"):
        print(f"  RH-side object net contact force: {env._manip_obj_rh_cf[0].norm().item():.2f} N")
        print(f"  LH-side object net contact force: {env._manip_obj_lh_cf[0].norm().item():.2f} N")

    dexhand_rh = env.dexhand_rh
    dexhand_lh = env.dexhand_lh

    obj_pos_rh_0 = env.rh_states["manip_obj_pos"][0].clone()
    obj_quat_rh_0 = env.rh_states["manip_obj_quat"][0].clone()
    obj_pos_lh_0 = env.lh_states["manip_obj_pos"][0].clone()
    obj_quat_lh_0 = env.lh_states["manip_obj_quat"][0].clone()

    print("\n--- Initial grip consistency at reset (min tip-to-surface distance, env 0, cm) ---")
    d_rh = tip_surface_distances(env, "rh", dexhand_rh)
    d_lh = tip_surface_distances(env, "lh", dexhand_lh)
    for name in TIP_NAMES:
        print(f"  RH {name:12s}: {d_rh[name][0].item()*100:6.2f}cm    LH {name:12s}: {d_lh[name][0].item()*100:6.2f}cm")

    if os.environ.get("DUMP_PENETRATION_NPZ"):
        # ALL hand bodies, not just the 5 fingertips -- finger segments/palm may penetrate
        # much more severely than the tips specifically.
        rh_all_pos_world = env.rh_states["joints_state"][0, :, :3].detach().cpu().numpy()  # (n_body, 3)
        lh_all_pos_world = env.lh_states["joints_state"][0, :, :3].detach().cpu().numpy()
        np.savez(
            os.environ["DUMP_PENETRATION_NPZ"],
            rh_all_pos_world=rh_all_pos_world,
            rh_body_names=np.array(dexhand_rh.body_names),
            lh_all_pos_world=lh_all_pos_world,
            lh_body_names=np.array(dexhand_lh.body_names),
            rh_obj_pos=env.rh_states["manip_obj_pos"][0].detach().cpu().numpy(),
            rh_obj_quat=env.rh_states["manip_obj_quat"][0].detach().cpu().numpy(),
            lh_obj_pos=env.lh_states["manip_obj_pos"][0].detach().cpu().numpy(),
            lh_obj_quat=env.lh_states["manip_obj_quat"][0].detach().cpu().numpy(),
            rh_obj_id=str(env.demo_data_rh["obj_id"][0]),
            lh_obj_id=str(env.demo_data_lh["obj_id"][0]),
        )
        print(f"Dumped penetration-check data to {os.environ['DUMP_PENETRATION_NPZ']}")

    print("\n--- Zero-action physics rollout (position-control DOFs hold reset targets; object free) ---")
    print(f"{'step':>4s} {'RH obj drift(cm)':>18s} {'RH obj rot drift(deg)':>22s} {'RH min tip-dist(cm)':>20s}   {'LH obj drift(cm)':>18s} {'LH obj rot drift(deg)':>22s} {'LH min tip-dist(cm)':>20s}")

    def quat_angle_diff(q1, q0):
        # both xyzw; angle between them via dot product
        dot = torch.clamp((q1 * q0).sum(-1).abs(), -1.0, 1.0)
        return 2 * torch.acos(dot) / np.pi * 180

    for step in range(9):
        env._refresh()
        env._update_states()
        d_rh = tip_surface_distances(env, "rh", dexhand_rh)
        d_lh = tip_surface_distances(env, "lh", dexhand_lh)
        rh_min = torch.stack([d_rh[n] for n in TIP_NAMES]).min(dim=0).values
        lh_min = torch.stack([d_lh[n] for n in TIP_NAMES]).min(dim=0).values

        obj_pos_rh = env.rh_states["manip_obj_pos"][0]
        obj_quat_rh = env.rh_states["manip_obj_quat"][0]
        obj_pos_lh = env.lh_states["manip_obj_pos"][0]
        obj_quat_lh = env.lh_states["manip_obj_quat"][0]

        rh_pos_drift = (obj_pos_rh - obj_pos_rh_0).norm().item() * 100
        lh_pos_drift = (obj_pos_lh - obj_pos_lh_0).norm().item() * 100
        rh_rot_drift = quat_angle_diff(obj_quat_rh[None], obj_quat_rh_0[None]).item()
        lh_rot_drift = quat_angle_diff(obj_quat_lh[None], obj_quat_lh_0[None]).item()

        print(
            f"{step:4d} {rh_pos_drift:18.3f} {rh_rot_drift:22.2f} {rh_min[0].item()*100:20.2f}   "
            f"{lh_pos_drift:18.3f} {lh_rot_drift:22.2f} {lh_min[0].item()*100:20.2f}"
        )

        if step < 8:
            env.gym.simulate(env.sim)
            env.gym.fetch_results(env.sim, True)


if __name__ == "__main__":
    main()
