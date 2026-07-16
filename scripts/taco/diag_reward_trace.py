"""
Step 1 + Step 2 diagnostic: trace per-step reward terms and raw tracking
error (obj_pos, obj_rot) bucketed by running_progress_buf (steps since
episode reset), comparing TACO vs OakInk-V2, using the FROZEN BASE ALONE
(zero residual) as a clean, dataset-independent proxy for "early in
training" behavior (residual starts near-zero-output at init; we already
confirmed base-only sr ~= trained-residual sr on TACO, so it's a reasonable
stand-in there too).

This answers two things at once:
  Step 1: is reward_obj_pos / reward_obj_rot pinned near zero (saturated)
          within the first ~20 steps on TACO but not on OakInk-V2?
  Step 2: does obj_pos/obj_rot error start near 0 at progress_buf==0/1 and
          GROW over the first 8 steps (physics/grip drift -- cause 2a), or
          is it already large at progress_buf==0/1 (frame/convention or
          reset-index bug -- cause 2b)?

Must be run from the repo root: python scripts/taco/diag_reward_trace.py <data_idx> [num_envs] [num_steps]
"""

from isaacgym import gymapi  # noqa: F401 -- must be first import in the whole process

import os
import sys
from collections import defaultdict

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

import lib  # noqa: F401
from lib.utils.reformat import omegaconf_to_dict
from lib.rl.runner import Runner
from lib.rl.network_builder import DictObsBuilder
from lib.rl.sep_network_builder import SepDictObsBuilder
from lib.rl.models import ModelA2CContinuousLogStd, SepModelA2CContinuousLogStd
from lib.rl.network_builder_residual_sh import ResRHDictObsBuilder, ResLHDictObsBuilder
from lib.rl.network_builder_residual_bih import ResBiHDictObsBuilder
from lib.rl.res_models import (
    ModelA2CContinuousLogStdResRH,
    ModelA2CContinuousLogStdResLH,
    ModelA2CContinuousLogStdResBiH,
)
from rl_games.algos_torch.model_builder import register_network, register_model
from rl_games.common import env_configurations, vecenv
from lib.utils.rlgames_utils import ComplexObsRLGPUEnv, RLGPUAlgoObserver, MultiObserver

import maniptrans_envs.lib  # noqa: F401
import maniptrans_envs.lib.envs.tasks.dexhandmanip_bih as bih_mod

MAX_BUCKET = 20
# bucket -> list of values
BUCKETS = defaultdict(lambda: defaultdict(list))


def make_trace_wrapper(orig_fn):
    def wrapped(*args):
        # args: (reset_buf, progress_buf, running_progress_buf, actions, states,
        #        target_states, max_length, scale_factor, dexhand_weight_idx,
        #        obj_pos_reward_coef, obj_rot_reward_coef) -- positional passthrough,
        # robust to the reward-coefficient-schedule signature change.
        running_progress_buf = args[2]
        states = args[4]
        target_states = args[5]
        rew_buf, reset_buf_out, success_buf, failure_buf, reward_dict, error_buf = orig_fn(*args)

        with torch.no_grad():
            current_obj_pos = states["manip_obj_pos"]
            target_obj_pos = target_states["manip_obj_pos"]
            diff_obj_pos_dist = torch.norm(target_obj_pos - current_obj_pos, dim=-1)

            current_obj_quat = states["manip_obj_quat"]
            target_obj_quat = target_states["manip_obj_quat"]
            from maniptrans_envs.lib.envs.tasks.dexhandmanip_bih import quat_mul, quat_conjugate, quat_to_angle_axis

            diff_obj_rot = quat_mul(target_obj_quat, quat_conjugate(current_obj_quat))
            diff_obj_rot_angle = quat_to_angle_axis(diff_obj_rot)[0].abs() / np.pi * 180

            rp = running_progress_buf.detach().cpu().numpy()
            reward_obj_pos = reward_dict["reward_obj_pos"].detach().cpu().numpy()
            reward_obj_rot = reward_dict["reward_obj_rot"].detach().cpu().numpy()
            d_pos = diff_obj_pos_dist.detach().cpu().numpy()
            d_rot = diff_obj_rot_angle.detach().cpu().numpy()

            for i in range(len(rp)):
                b = int(rp[i])
                if b > MAX_BUCKET:
                    continue
                BUCKETS[b]["reward_obj_pos"].append(float(reward_obj_pos[i]))
                BUCKETS[b]["reward_obj_rot"].append(float(reward_obj_rot[i]))
                BUCKETS[b]["diff_obj_pos_dist"].append(float(d_pos[i]))
                BUCKETS[b]["diff_obj_rot_angle"].append(float(d_rot[i]))

        return rew_buf, reset_buf_out, success_buf, failure_buf, reward_dict, error_buf

    return wrapped


def main():
    data_idx = sys.argv[1]
    num_envs = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    num_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 400
    # "true" (default) = eval mode, obj_pos/obj_rot reward coef forced to *Tight (matches old
    # behavior / matured training). "false" = training mode: gym.get_frame_count(sim) actually
    # advances from 0, so the loose->tight reward-coefficient schedule is genuinely exercised,
    # exactly as it would be at the start of a real training run.
    test_mode = sys.argv[4] if len(sys.argv) > 4 else "true"
    obj_pos_coef_loose = sys.argv[5] if len(sys.argv) > 5 else None
    obj_rot_coef_loose = sys.argv[6] if len(sys.argv) > 6 else None

    _orig = bih_mod.compute_imitation_reward
    bih_mod.compute_imitation_reward = make_trace_wrapper(_orig)

    import builtins

    def _isinstance_allow_wrapper(obj, cls):
        if obj is bih_mod.compute_imitation_reward and cls is torch.jit.ScriptFunction:
            return True
        return builtins.isinstance(obj, cls)

    bih_mod.isinstance = _isinstance_allow_wrapper

    config_dir = os.path.abspath("main/cfg")
    overrides = [
        "task=ResDexHand",
        "dexhand=inspire",
        "side=BiH",
        "headless=true",
        f"num_envs={num_envs}",
        f"test={test_mode}",
        "randomStateInit=true",
        f"dataIndices=[{data_idx}]",
        "rh_base_model_checkpoint=assets/imitator_ckp/imitator_rh_inspire.pth",
        "lh_base_model_checkpoint=assets/imitator_ckp/imitator_lh_inspire.pth",
        "experiment=diag_reward_trace",
    ]
    if obj_pos_coef_loose is not None:
        overrides.append(f"task.env.objPosRewardCoefLoose={obj_pos_coef_loose}")
    if obj_rot_coef_loose is not None:
        overrides.append(f"task.env.objRotRewardCoefLoose={obj_rot_coef_loose}")
    with hydra.initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        cfg = hydra.compose(config_name="config", overrides=overrides)

    register_model("my_continuous_a2c_logstd", ModelA2CContinuousLogStd)
    register_network("dict_obs_actor_critic", DictObsBuilder)
    register_network("sep_dict_obs_actor_critic", SepDictObsBuilder)
    register_model("sep_my_continuous_a2c_logstd", SepModelA2CContinuousLogStd)
    register_network("res_rh_dict_obs_actor_critic", ResRHDictObsBuilder)
    register_model("res_rh_my_continuous_a2c_logstd", ModelA2CContinuousLogStdResRH)
    register_network("res_lh_dict_obs_actor_critic", ResLHDictObsBuilder)
    register_model("res_lh_my_continuous_a2c_logstd", ModelA2CContinuousLogStdResLH)
    register_network("res_bih_dict_obs_actor_critic", ResBiHDictObsBuilder)
    register_model("res_bih_my_continuous_a2c_logstd", ModelA2CContinuousLogStdResBiH)

    def create_isaacgym_env():
        kwargs = dict(
            sim_device=cfg.sim_device,
            rl_device=cfg.rl_device,
            graphics_device_id=cfg.graphics_device_id,
            multi_gpu=cfg.multi_gpu,
            cfg=cfg.task,
            display=cfg.display,
            record=cfg.capture_video,
            has_headless_arg=True,
            headless=cfg.headless,
        )
        return maniptrans_envs.lib.make(**kwargs)

    env_configurations.register("rlgpu", {"vecenv_type": "RLGPU", "env_creator": create_isaacgym_env})
    vecenv.register("RLGPU", lambda config_name, num_actors: ComplexObsRLGPUEnv(config_name))

    from lib.utils.utils import set_np_formatting, set_seed

    set_np_formatting()
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)

    rlg_config_dict = omegaconf_to_dict(cfg.rl_train)
    train_cfg = rlg_config_dict["params"]["config"]
    train_cfg["device"] = cfg.rl_device
    train_cfg["population_based_training"] = False
    train_cfg["pbt_idx"] = None
    train_cfg["full_experiment_name"] = None

    runner = Runner(MultiObserver([RLGPUAlgoObserver()]))
    runner.load(rlg_config_dict)
    runner.reset()

    player = runner.create_player({})
    print(f"Tracing reward/error terms BASE-ONLY on data_idx={data_idx}, num_envs={num_envs}, num_steps={num_steps}")

    obses = player.env_reset(player.env)
    player.get_batch_size(obses, 1)

    done = None
    for step in range(num_steps):
        if done is not None and torch.any(done):
            obses, _ = player.env.reset_done()

        input_dict = {"is_train": False, "prev_actions": None, "obs": obses, "rnn_states": player.states}
        with torch.no_grad():
            res_dict = player.model(input_dict)
        base_actions = res_dict["base_actions"]
        residual_zeroed = torch.zeros_like(res_dict["mus"])
        actions = torch.cat([base_actions, residual_zeroed], dim=1)
        if player.clip_actions:
            actions = torch.clamp(actions, -1.0, 1.0)

        obses, r, done, info = player.env_step(player.env, actions)

    print(f"\n=== Per-progress-step trace (base-only): data_idx={data_idx} ===")
    print(f"{'step':>4s} {'n':>6s} {'reward_obj_pos':>15s} {'reward_obj_rot':>15s} {'diff_pos(cm)':>13s} {'diff_rot(deg)':>14s}")
    for b in range(MAX_BUCKET + 1):
        if b not in BUCKETS:
            continue
        d = BUCKETS[b]
        n = len(d["reward_obj_pos"])
        rp = np.mean(d["reward_obj_pos"])
        rr = np.mean(d["reward_obj_rot"])
        dp = np.mean(d["diff_obj_pos_dist"]) * 100
        dr = np.mean(d["diff_obj_rot_angle"])
        print(f"{b:4d} {n:6d} {rp:15.6f} {rr:15.6f} {dp:13.3f} {dr:14.2f}")


if __name__ == "__main__":
    main()
