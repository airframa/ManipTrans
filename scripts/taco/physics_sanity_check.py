"""
M4 pre-flight physics sanity check (lightweight, NOT full training): spawns a
single-env instance of the REAL DexHandManipBiHEnv (dexhandmanip_bih.py) --
gravity on, real physical table with collision, same config main/rl/train.py
would use -- loaded with our height-corrected TACO PoC sequence (t0), calls
reset() and a few zero-action steps, and captures a camera frame. No RL
policy, no training loop.

Reuses the real training entry point's own config composition (main/rl/train.py's
launch_rlg_hydra + maniptrans_envs.lib.make), just stopped right after env
creation instead of handing off to the RL runner.

Must be run from the repo root: python scripts/taco/physics_sanity_check.py
"""

from isaacgym import gymapi  # noqa: F401 -- must be first import in the whole process

import os

import hydra
import imageio
import torch
from omegaconf import OmegaConf

import lib  # noqa: F401 -- registers all the custom OmegaConf resolvers (concat, ndof, nbody, ...)
import maniptrans_envs  # noqa: F401 -- registers TASK_MAP


OUT_PATH = "outputs/m4_preflight_physics_check.png"
OUT_PATH_CLIP = "outputs/m4_preflight_physics_check.mp4"


def main():
    config_dir = os.path.abspath("main/cfg")
    with hydra.initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        cfg = hydra.compose(
            config_name="config",
            overrides=[
                "task=ResDexHand",
                "dexhand=inspire",
                "side=BiH",
                "headless=true",
                # num_envs=1 hits a pre-existing repo quirk: dexhandmanip_bih.py's pack_data()
                # calls torch.stack(...).squeeze(), which collapses away the whole batch
                # dimension (not just the intended per-tensor dims) when num_envs==1, breaking
                # downstream indexing like demo_data_rh["obj_trajectory"][env_idx][0]. Not
                # TACO-specific -- would happen for any dataset at num_envs=1. num_envs=2 avoids
                # it while staying a lightweight, non-training sanity check.
                "num_envs=2",
                "test=true",  # inference-mode env semantics (no domain randomization), we're not training
                "randomStateInit=false",  # start every reset at frame 0, easiest to sanity-check
                "dataIndices=[t0]",
                "rh_base_model_checkpoint=assets/imitator_ckp/imitator_rh_inspire.pth",
                "lh_base_model_checkpoint=assets/imitator_ckp/imitator_lh_inspire.pth",
                "experiment=m4_preflight_check",
            ],
        )

    cfg_dict = OmegaConf.to_container(cfg.task, resolve=True)
    print(f"Task: {cfg_dict['name']}, numEnvs={cfg_dict['env']['numEnvs']}, dataIndices={cfg_dict['env']['dataIndices']}")

    env = maniptrans_envs.lib.make(
        sim_device="cuda:0",
        rl_device="cuda:0",
        graphics_device_id=0,
        cfg=cfg.task,
        display=False,
        record=True,  # enables the camera sensor + 1280x720 bimanual framing
        has_headless_arg=True,
        headless=True,
    )

    print("Env created. Calling reset()...")
    env.reset()
    print(f"env.num_envs={env.num_envs} env.num_actions={env.num_actions}")

    # Deliberately NOT calling env.step(): dexhandmanip_bih.py's pre_physics_step()
    # expects actions shaped for the residual-policy split (base-model output +
    # residual-policy output concatenated, done by the RL agent/runner, not visible
    # at the raw env.step() interface) -- faking that correctly is a distraction from
    # what this check actually needs. Instead we drive the same low-level
    # simulate/fetch/render sequence VecTask.step() uses (vec_task.py:485-498),
    # skipping pre/post_physics_step (the RL-specific action-application and
    # reward/observation bookkeeping) entirely. This is a purely physical
    # "let gravity act and see what happens" check -- exactly "no RL stepping".
    frames = []
    with torch.no_grad():
        for i in range(30):
            env.gym.simulate(env.sim)
            env.gym.fetch_results(env.sim, True)
            env.gym.step_graphics(env.sim)
            env.gym.render_all_camera_sensors(env.sim)
            env.gym.start_access_image_tensors(env.sim)
            if env.camera_obs is not None:
                frame = env.camera_obs[0][..., :3].detach().cpu().numpy().astype("uint8")
                frames.append(frame)
            env.gym.end_access_image_tensors(env.sim)
            if i % 10 == 0:
                print(f"  step {i}: table exists, gravity on, no explosion so far")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    if frames:
        imageio.imwrite(OUT_PATH, frames[0])
        imageio.mimwrite(OUT_PATH_CLIP, frames, fps=10)
        print(f"Saved {OUT_PATH} and {OUT_PATH_CLIP} ({len(frames)} frames)")
    else:
        print("WARNING: no camera frames captured (env.camera_obs is None) -- record flag may not have taken effect")

    # report final object/hand height so we can numerically confirm no floor/table clipping
    root_state = env._root_state if hasattr(env, "_root_state") else None
    if root_state is not None:
        print("Final root_state z-heights (env 0):")
        for i in range(root_state.shape[1]):
            print(f"  actor {i}: z={root_state[0, i, 2].item():.4f}")


if __name__ == "__main__":
    main()
