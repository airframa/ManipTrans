"""
Closes the measurement gap flagged in review: the earlier penetration check
used dexhand JOINT ORIGINS (fingertip marker points, which for the Inspire
URDF have NO collision geometry at all -- <link name="R_thumb_tip"> etc are
visual-only 5mm spheres). The REAL collision surface nearest each fingertip
is the last phalanx's mesh (thumb: *_distal, other 4 fingers: *_intermediate,
since they have no separate distal link) -- an actual STL mesh, not a point
or simple capsule.

This script only DUMPS world pose (position + quaternion) of those 5
collision-bearing phalanx links per hand, plus the object's pose/id, for a
given dataset+frame. A separate offline script (no isaacgym needed) loads
the actual STL collision meshes and the object's collision mesh and computes
true mesh-vs-mesh minimum distance/penetration.

Must be run from the repo root: python scripts/taco/diag_true_penetration_dump.py <data_idx> <rolloutBegin> <out.npz>
"""

from isaacgym import gymapi  # noqa: F401 -- must be first import in the whole process

import os
import sys

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

import lib  # noqa: F401
import maniptrans_envs.lib  # noqa: F401

# ALL collision-bearing links (excludes *_tip, which are visual-only 5mm sphere markers
# with no <collision> element at all -- see inspire_hand_right.urdf). Base link (palm) +
# every proximal/intermediate/distal segment.
COLLISION_LINK_SUFFIX = {
    "palm": "hand_base_link",
    "thumb_base": "thumb_proximal_base",
    "thumb_proximal": "thumb_proximal",
    "thumb_intermediate": "thumb_intermediate",
    "thumb_distal": "thumb_distal",
    "index_proximal": "index_proximal",
    "index_intermediate": "index_intermediate",
    "middle_proximal": "middle_proximal",
    "middle_intermediate": "middle_intermediate",
    "ring_proximal": "ring_proximal",
    "ring_intermediate": "ring_intermediate",
    "pinky_proximal": "pinky_proximal",
    "pinky_intermediate": "pinky_intermediate",
}


def main():
    data_idx = sys.argv[1]
    rollout_begin = sys.argv[2]
    out_path = sys.argv[3]

    config_dir = os.path.abspath("main/cfg")
    overrides = [
        "task=ResDexHand",
        "dexhand=inspire",
        "side=BiH",
        "headless=true",
        "num_envs=2",
        "test=true",
        "randomStateInit=false",
        f"dataIndices=[{data_idx}]",
        "rh_base_model_checkpoint=assets/imitator_ckp/imitator_rh_inspire.pth",
        "lh_base_model_checkpoint=assets/imitator_ckp/imitator_lh_inspire.pth",
        "experiment=diag_true_penetration",
        f"rolloutBegin={rollout_begin}",
    ]
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

    env.reset()
    env.gym.simulate(env.sim)
    env.gym.fetch_results(env.sim, True)
    env._refresh()
    env._update_states()

    result = {}
    for side, dexhand in [("rh", env.dexhand_rh), ("lh", env.dexhand_lh)]:
        states = getattr(env, f"{side}_states")
        for finger, suffix in COLLISION_LINK_SUFFIX.items():
            body_name = [b for b in dexhand.body_names if b.endswith(suffix)][0]
            idx = dexhand.body_names.index(body_name)
            pose10 = states["joints_state"][0, idx, :7].detach().cpu().numpy()  # pos(3)+quat(4) xyzw
            result[f"{side}_{finger}_pos"] = pose10[:3]
            result[f"{side}_{finger}_quat"] = pose10[3:7]
            result[f"{side}_{finger}_bodyname"] = body_name

        result[f"{side}_obj_pos"] = states["manip_obj_pos"][0].detach().cpu().numpy()
        result[f"{side}_obj_quat"] = states["manip_obj_quat"][0].detach().cpu().numpy()
        demo_data = getattr(env, f"demo_data_{side}")
        result[f"{side}_obj_id"] = str(demo_data["obj_id"][0])

    result["dexhand_side_rh"] = str(env.dexhand_rh)  # e.g. "inspire" -- lets offline script pick R_/L_ STL set
    result["dexhand_side_lh"] = str(env.dexhand_lh)
    np.savez(out_path, **result)
    print(f"Dumped to {out_path}")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
