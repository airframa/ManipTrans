"""
Dumps the RL env's observation tensors (proprioception, privileged, target)
for a single dataIndices value, to a .npz file, for later side-by-side
comparison (see compare_obs_taco_vs_oakink2.py's diff step). Run once per
dataset, in separate processes -- Isaac Gym is unstable creating a second sim
in the same process.

Must be run from the repo root: python scripts/taco/dump_obs_single.py <data_idx> <out.npz>
Example: python scripts/taco/dump_obs_single.py 20aed@0 /tmp/obs_oakink2.npz
"""

from isaacgym import gymapi  # noqa: F401

import os
import sys

import hydra
import numpy as np

import lib  # noqa: F401
import maniptrans_envs.lib  # noqa: F401


def main():
    data_idx = sys.argv[1]
    out_path = sys.argv[2]

    config_dir = os.path.abspath("main/cfg")
    with hydra.initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        cfg = hydra.compose(
            config_name="config",
            overrides=[
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
                f"experiment=obs_dump_{data_idx.replace('@', '_')}",
            ],
        )

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
    # VecTask.reset() is documented as NOT calculating observations ("Actual reset and
    # observation calculation need to be implemented by user") -- confirmed empirically:
    # obs_dict is all-zero right after reset() for both TACO AND OakInk2, so this isn't
    # a TACO-specific issue. Call compute_observations() directly to get real frame-0 values.
    env.compute_observations()
    obs_dict = env.obs_dict
    print(f"obs_dict keys: {list(obs_dict.keys())}")

    out = {}
    for key in ["proprioception", "privileged", "target"]:
        if key in obs_dict:
            arr = obs_dict[key][0].detach().cpu().numpy()  # env 0
            out[key] = arr
            print(f"  {key}: shape={arr.shape} min={arr.min():.4f} max={arr.max():.4f} mean={arr.mean():.4f} "
                  f"zeros={(arr == 0).sum()}/{arr.size} nans={np.isnan(arr).sum()}")
        else:
            print(f"  {key}: MISSING")

    np.savez(out_path, **out)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
