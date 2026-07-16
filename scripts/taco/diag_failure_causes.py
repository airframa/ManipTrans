"""
Hypothesis A diagnostic: break down WHY failed_execute trips during a TACO
rollout -- which specific per-step threshold in compute_imitation_reward
(dexhandmanip_bih.py) is responsible for the ~94-95% failure rate.

Monkeypatches the module-level compute_imitation_reward (a plain
torch.jit.script free function, looked up by name at call time in
compute_reward_side) with a wrapper that calls the real scripted function
(so actual training/eval behavior is byte-identical) and ALSO recomputes,
in plain untraced python, which individual sub-condition of failed_execute
is responsible whenever failure_buf fires. Tallies counts across a full
standalone rollout.

Must be run from the repo root: python scripts/taco/diag_failure_causes.py <checkpoint_path> [data_idx] [num_envs] [target_episodes]
"""

from isaacgym import gymapi  # noqa: F401 -- must be first import in the whole process

import os
import sys
from collections import Counter

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

FAIL_COUNTS = Counter()
FAIL_EPISODES = 0  # number of (env,timestep) failure events tallied
MAG_SAMPLES = {"obj_pos": [], "obj_rot_deg": [], "thumb_tip": [], "near_but_no_contact_mindist": []}
RUNNING_PROGRESS_AT_FAIL = []


def make_diagnostic_wrapper(orig_fn):
    def wrapped(*args):
        # args: (reset_buf, progress_buf, running_progress_buf, actions, states,
        #        target_states, max_length, scale_factor, dexhand_weight_idx,
        #        obj_pos_reward_coef, obj_rot_reward_coef) -- positional passthrough,
        # robust to the reward-coefficient-schedule signature change.
        states = args[4]
        target_states = args[5]
        scale_factor = args[7]
        dexhand_weight_idx = args[8]
        rew_buf, reset_buf_out, success_buf, failure_buf, reward_dict, error_buf = orig_fn(*args)

        fail_idx = failure_buf.nonzero(as_tuple=False).flatten()
        if len(fail_idx) > 0:
            with torch.no_grad():
                current_eef_pos = states["base_state"][:, :3]
                target_eef_pos = target_states["wrist_pos"]

                joints_pos = states["joints_state"][:, 1:, :3]
                target_joints_pos = target_states["joints_pos"]
                diff_joints_pos_dist = torch.norm(target_joints_pos - joints_pos, dim=-1)

                def tip_dist(name):
                    idx = [k - 1 for k in dexhand_weight_idx[name]]
                    return diff_joints_pos_dist[:, idx].mean(dim=-1)

                diff_thumb = tip_dist("thumb_tip")
                diff_index = tip_dist("index_tip")
                diff_middle = tip_dist("middle_tip")
                diff_pinky = tip_dist("pinky_tip")
                diff_ring = tip_dist("ring_tip")
                diff_l1 = tip_dist("level_1_joints")
                diff_l2 = tip_dist("level_2_joints")

                current_obj_pos = states["manip_obj_pos"]
                target_obj_pos = target_states["manip_obj_pos"]
                diff_obj_pos_dist = torch.norm(target_obj_pos - current_obj_pos, dim=-1)

                current_obj_quat = states["manip_obj_quat"]
                target_obj_quat = target_states["manip_obj_quat"]

                from maniptrans_envs.lib.envs.tasks.dexhandmanip_bih import quat_mul, quat_conjugate, quat_to_angle_axis

                diff_obj_rot = quat_mul(target_obj_quat, quat_conjugate(current_obj_quat))
                diff_obj_rot_angle = quat_to_angle_axis(diff_obj_rot)[0]

                finger_tip_distance = target_states["tips_distance"]
                tip_contact_state = target_states["tip_contact_state"]

                current_eef_vel = states["base_state"][:, 7:10]
                current_eef_ang_vel = states["base_state"][:, 10:13]
                current_dof_vel = states["dq"]
                current_obj_vel = states["manip_obj_vel"]
                current_obj_ang_vel = states["manip_obj_ang_vel"]
                joints_vel = states["joints_state"][:, 1:, 7:10]

                err_buf_local = (
                    (torch.norm(current_eef_vel, dim=-1) > 100)
                    | (torch.norm(current_eef_ang_vel, dim=-1) > 200)
                    | (torch.norm(joints_vel, dim=-1).mean(-1) > 100)
                    | (torch.abs(current_dof_vel).mean(-1) > 200)
                    | (torch.norm(current_obj_vel, dim=-1) > 100)
                    | (torch.norm(current_obj_ang_vel, dim=-1) > 200)
                )

                conds = {
                    "obj_pos": diff_obj_pos_dist > 0.02 / 0.343 * scale_factor**3,
                    "thumb_tip": diff_thumb > 0.04 / 0.7 * scale_factor,
                    "index_tip": diff_index > 0.045 / 0.7 * scale_factor,
                    "middle_tip": diff_middle > 0.05 / 0.7 * scale_factor,
                    "pinky_tip": diff_pinky > 0.06 / 0.7 * scale_factor,
                    "ring_tip": diff_ring > 0.06 / 0.7 * scale_factor,
                    "level_1": diff_l1 > 0.07 / 0.7 * scale_factor,
                    "level_2": diff_l2 > 0.08 / 0.7 * scale_factor,
                    "obj_rot": diff_obj_rot_angle.abs() / np.pi * 180 > 30 / 0.343 * scale_factor**3,
                    "near_but_no_contact": torch.any(
                        (finger_tip_distance < 0.005) & ~(tip_contact_state.any(1)), dim=-1
                    ),
                    "error_buf_sanity": err_buf_local,
                }

                min_tip_dist = finger_tip_distance.min(dim=-1).values

                for i in fail_idx.tolist():
                    fired = [name for name, mask in conds.items() if bool(mask[i])]
                    if not fired:
                        FAIL_COUNTS["UNKNOWN(running_progress<8?)"] += 1
                    else:
                        for name in fired:
                            FAIL_COUNTS[name] += 1
                    global FAIL_EPISODES
                    FAIL_EPISODES += 1
                    RUNNING_PROGRESS_AT_FAIL.append(int(running_progress_buf[i].item()))
                    if bool(conds["obj_pos"][i]):
                        MAG_SAMPLES["obj_pos"].append(float(diff_obj_pos_dist[i].item()))
                    if bool(conds["obj_rot"][i]):
                        MAG_SAMPLES["obj_rot_deg"].append(float(diff_obj_rot_angle[i].abs().item() / np.pi * 180))
                    if bool(conds["thumb_tip"][i]):
                        MAG_SAMPLES["thumb_tip"].append(float(diff_thumb[i].item()))
                    if bool(conds["near_but_no_contact"][i]):
                        MAG_SAMPLES["near_but_no_contact_mindist"].append(float(min_tip_dist[i].item()))

        return rew_buf, reset_buf_out, success_buf, failure_buf, reward_dict, error_buf

    return wrapped


def main():
    checkpoint = sys.argv[1]
    data_idx = sys.argv[2] if len(sys.argv) > 2 else "t3"
    num_envs = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
    target_episodes = int(sys.argv[4]) if len(sys.argv) > 4 else 500

    _orig_compute_imitation_reward = bih_mod.compute_imitation_reward
    bih_mod.compute_imitation_reward = make_diagnostic_wrapper(_orig_compute_imitation_reward)

    # compute_reward_side has `assert not self.headless or isinstance(compute_imitation_reward,
    # torch.jit.ScriptFunction)` -- a perf/correctness guard that our plain-python diagnostic
    # wrapper legitimately fails. Shadow `isinstance` in this module's namespace only, so the
    # assert still passes for our wrapper while behaving identically for everything else.
    import builtins

    def _isinstance_allow_wrapper(obj, cls):
        if obj is bih_mod.compute_imitation_reward and cls is torch.jit.ScriptFunction:
            return True
        return builtins.isinstance(obj, cls)

    bih_mod.isinstance = _isinstance_allow_wrapper

    config_dir = os.path.abspath("main/cfg")
    with hydra.initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        cfg = hydra.compose(
            config_name="config",
            overrides=[
                "task=ResDexHand",
                "dexhand=inspire",
                "side=BiH",
                "headless=true",
                f"num_envs={num_envs}",
                "test=true",
                "randomStateInit=true",
                f"dataIndices=[{data_idx}]",
                "rh_base_model_checkpoint=assets/imitator_ckp/imitator_rh_inspire.pth",
                "lh_base_model_checkpoint=assets/imitator_ckp/imitator_lh_inspire.pth",
                f"checkpoint={checkpoint}",
                "experiment=diag_failure_causes",
            ],
        )

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
    player.restore(checkpoint)
    print(f"Restored checkpoint: {checkpoint}")
    print(f"Evaluating on data_idx={data_idx}, num_envs={num_envs}, target_episodes={target_episodes}")

    obses = player.env_reset(player.env)
    player.get_batch_size(obses, 1)

    total_success = 0
    total_failure = 0
    total_done = 0

    done = None
    step = 0
    while total_done < target_episodes:
        if done is not None and torch.any(done):
            obses, _ = player.env.reset_done()

        actions = player.get_action(obses, player.is_deterministic)
        obses, r, done, info = player.env_step(player.env, actions)
        step += 1

        done_indices = done.nonzero(as_tuple=False).flatten()
        n_done = len(done_indices)
        if n_done > 0:
            succ = player.env.success_buf[done_indices]
            fail = player.env.failure_buf[done_indices]
            total_success += int(succ.sum().item())
            total_failure += int(fail.sum().item())
            total_done += n_done

        if step % 200 == 0:
            print(f"  step {step}: {total_done} episodes done, sr={total_success/max(total_done,1):.4f}")

    print("\n=== Failure cause breakdown ===")
    print(f"checkpoint: {checkpoint}  data_idx: {data_idx}")
    print(f"total episodes: {total_done}  successes: {total_success}  failures: {total_failure}")
    print(f"total (env,step) failure EVENTS tallied (one per rh+lh side call, may double count bih): {FAIL_EPISODES}")
    print("condition -> count (fraction of failure events it participated in):")
    for name, cnt in FAIL_COUNTS.most_common():
        frac = cnt / max(FAIL_EPISODES, 1)
        print(f"  {name:25s} {cnt:6d}  ({frac*100:5.1f}%)")

    print("\n=== Magnitude of triggering values (how far past threshold) ===")
    thresholds = {"obj_pos": None, "obj_rot_deg": None, "thumb_tip": None, "near_but_no_contact_mindist": 0.005}
    for name, vals in MAG_SAMPLES.items():
        if not vals:
            continue
        arr = np.array(vals)
        print(f"  {name:28s} n={len(arr):5d}  mean={arr.mean():.4f}  median={np.median(arr):.4f}  p10={np.percentile(arr,10):.4f}  p90={np.percentile(arr,90):.4f}")

    rp = np.array(RUNNING_PROGRESS_AT_FAIL)
    print(f"\nrunning_progress_buf at moment of failure: n={len(rp)} mean={rp.mean():.1f} median={np.median(rp):.1f} min={rp.min()} max={rp.max()}")


if __name__ == "__main__":
    main()
