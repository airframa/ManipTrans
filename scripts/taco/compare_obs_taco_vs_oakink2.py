"""
Systemic-bug investigation: side-by-side comparison of the actual observation
tensors (proprioception, privileged, target) the RL env produces for TACO vs.
OakInk-V2's 20aed@0 -- the only sequence we have confirmed evidence trained
successfully end-to-end (runs/cross_20aed@0_inspire__06-30-10-25-22, sr~0.96-0.98).

Builds two real DexHandManipBiHEnv instances sequentially (same config path
main/rl/train.py uses), calls reset(), and dumps obs_dict['proprioception'] /
['privileged'] / ['target'] for env 0, right hand side, looking for zeros,
NaNs, or scale mismatches between the two datasets' observation vectors.

Must be run from the repo root: python scripts/taco/compare_obs_taco_vs_oakink2.py
"""

from isaacgym import gymapi  # noqa: F401 -- must be first import in the whole process

import os

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

import lib  # noqa: F401 -- registers custom OmegaConf resolvers
import maniptrans_envs.lib  # noqa: F401 -- namespace package; need the submodule imported explicitly


def build_env(data_idx):
    config_dir = os.path.abspath("main/cfg")
    with hydra.initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        cfg = hydra.compose(
            config_name="config",
            overrides=[
                "task=ResDexHand",
                "dexhand=inspire",
                "side=BiH",
                "headless=true",
                "num_envs=2",  # avoid the num_envs=1 pack_data().squeeze() bug
                "test=true",
                "randomStateInit=false",
                f"dataIndices=[{data_idx}]",
                "rh_base_model_checkpoint=assets/imitator_ckp/imitator_rh_inspire.pth",
                "lh_base_model_checkpoint=assets/imitator_ckp/imitator_lh_inspire.pth",
                f"experiment=obs_compare_{data_idx.replace('@','_')}",
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
    return env


def summarize(name, arr):
    arr = np.asarray(arr)
    n_zero = int(np.sum(arr == 0))
    n_nan = int(np.sum(np.isnan(arr)))
    print(
        f"    {name:12s} shape={arr.shape} min={np.nanmin(arr):+.4f} max={np.nanmax(arr):+.4f} "
        f"mean={np.nanmean(arr):+.4f} zeros={n_zero}/{arr.size} nans={n_nan}"
    )


def dump_obs(tag, env):
    obs_dict = env.obs_dict
    print(f"--- {tag} ---")
    for key in ["proprioception", "privileged", "target"]:
        if key not in obs_dict:
            print(f"    {key}: MISSING from obs_dict")
            continue
        v = obs_dict[key][0].detach().cpu().numpy()  # env 0, both-hands-concatenated (united bimanual_mode)
        summarize(key, v)
    return {k: obs_dict[k][0].detach().cpu().numpy() for k in ["proprioception", "privileged", "target"] if k in obs_dict}


def main():
    print("Building OakInk-V2 (20aed@0) env...")
    env_ok = build_env("20aed@0")
    obs_ok = dump_obs("OakInk-V2 20aed@0 (known-good)", env_ok)
    print()

    print("Building TACO (t2, spoon/bowl) env...")
    env_taco = build_env("t2")
    obs_taco = dump_obs("TACO t2 (spoon/bowl, failing)", env_taco)

    print()
    print("=" * 70)
    print("Structural diff (per-block first 10 values):")
    for key in ["proprioception", "privileged", "target"]:
        if key not in obs_ok or key not in obs_taco:
            continue
        print(f"\n[{key}]")
        print("  OakInk2:", np.round(obs_ok[key][:10], 4))
        print("  TACO   :", np.round(obs_taco[key][:10], 4))
        if obs_ok[key].shape != obs_taco[key].shape:
            print(f"  !!! SHAPE MISMATCH: OakInk2={obs_ok[key].shape} vs TACO={obs_taco[key].shape}")


if __name__ == "__main__":
    main()
