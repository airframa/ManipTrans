"""
Hypothesis B diagnostic: roll out the FROZEN base imitator ALONE (residual
zeroed out) on a TACO sequence vs an OakInk-V2 sequence, and compare tracking
error. If the base policy is coherent on OakInk-V2 (its training distribution)
but incoherent on TACO, the residual is being asked to correct a base that's
fundamentally wrong for TACO's motion style -- a systemic cap independent of
the center_idx fix.

We do NOT restore an RL (residual) checkpoint -- the frozen base_model's
weights come directly from rh_base_model_checkpoint/lh_base_model_checkpoint
(assets/imitator_ckp/*.pth), identically for every sequence, regardless of
which residual checkpoint (if any) is loaded. So a randomly-initialized
residual network is fine: we zero out its contribution before every
env.step() anyway, per dexhandmanip_bih.py's pre_physics_step split
(res_split_idx = actions.shape[1] // 2 when use_pid_control=False).

Must be run from the repo root: python scripts/taco/rollout_base_only.py <data_idx> [num_envs] [target_episodes]
"""

from isaacgym import gymapi  # noqa: F401 -- must be first import in the whole process

import os
import sys

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


def main():
    data_idx = sys.argv[1]
    num_envs = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    target_episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 200

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
                "experiment=rollout_base_only",
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
    # Deliberately NOT calling player.restore(): the frozen base_model's weights
    # are loaded independently from rh/lh_base_model_checkpoint above. The
    # residual network stays randomly initialized, but we zero its contribution
    # to every action below, so its init is irrelevant.
    print(f"Evaluating BASE-ONLY (zero residual) on data_idx={data_idx}, num_envs={num_envs}, target_episodes={target_episodes}")

    obses = player.env_reset(player.env)
    player.get_batch_size(obses, 1)

    total_success = 0
    total_failure = 0
    total_done = 0
    obj_pos_err_samples = []
    obj_rot_err_samples = []

    done = None
    step = 0
    while total_done < target_episodes:
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
        step += 1

        env = player.env
        # env attribute names for object pos/rot tracking error (bimanual env averages both sides internally
        # into success/failure, but we want raw tracking magnitude regardless of pass/fail threshold)
        done_indices = done.nonzero(as_tuple=False).flatten()
        n_done = len(done_indices)
        if n_done > 0:
            succ = env.success_buf[done_indices]
            fail = env.failure_buf[done_indices]
            total_success += int(succ.sum().item())
            total_failure += int(fail.sum().item())
            total_done += n_done

        if step % 100 == 0:
            print(f"  step {step}: {total_done} episodes done, sr={total_success/max(total_done,1):.4f}, fr={total_failure/max(total_done,1):.4f}")

    sr = total_success / total_done
    fr = total_failure / total_done
    print(f"\n=== BASE-ONLY (zero residual) rollout result: data_idx={data_idx} ===")
    print(f"total episodes: {total_done}  successes: {total_success}  failures: {total_failure}")
    print(f"BASE-ONLY sr = {sr:.4f}")
    print(f"BASE-ONLY fr = {fr:.4f}")


if __name__ == "__main__":
    main()
