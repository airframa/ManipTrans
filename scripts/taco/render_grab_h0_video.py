"""
Labeled kinematic-replay video for the M3 methodology check (Part 2): the new,
longer GRAB baseline sequence (h0, 108 frames -- see check_coord_frames.py /
compare_verify_m3.py). Single-hand (right only), single-object, same style as
render_m3_video.py's bimanual TACO video.

Reuses render_m3_video.py's camera/projection/asset-loading helpers directly
(no logic duplicated) -- only the sim setup (one hand + one object instead of
two of each) and data source differ.

Must be run from the repo root: python scripts/taco/render_grab_h0_video.py
"""

from isaacgym import gymapi, gymtorch  # noqa: F401 -- must be first import in the whole process

import os
import pickle

import cv2
import imageio
import torch

from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory
import maniptrans_envs.lib.envs.dexhands  # noqa: F401

from main.dataset.grab_dataset_dexhand_long import GrabDemoLongDexHand
from main.dataset.transform import aa_to_quat, rotmat_to_quat

from render_m3_video import build_view_proj, project_to_screen, build_table_transf, load_hand_asset, load_obj_asset

OUT_PATH = "outputs/grab_h0_retargeting_preview_labeled.mp4"
FPS = 60


def main():
    device = "cuda:0"
    identity = torch.eye(4, dtype=torch.float32, device=device)

    dexhand_rh = DexHandFactory.create_hand("inspire", "right")

    fdata = GrabDemoLongDexHand(mujoco2gym_transf=identity, device=device, dexhand=dexhand_rh)
    data = fdata["h0"]

    table_transf = build_table_transf(device)

    with open(f"data/retargeting/grab_demo_long/mano2{str(dexhand_rh)}/102_sv_dict_st_0_ed_108.pkl", "rb") as f:
        opt = pickle.load(f)

    Tf = opt["opt_wrist_pos"].shape[0]

    opt_wrist_pos = torch.tensor(opt["opt_wrist_pos"], dtype=torch.float32, device=device)
    opt_wrist_quat = aa_to_quat(torch.tensor(opt["opt_wrist_rot"], dtype=torch.float32, device=device))[:, [1, 2, 3, 0]]
    opt_dof_pos = torch.tensor(opt["opt_dof_pos"], dtype=torch.float32, device=device)

    obj_traj = table_transf[None] @ data["obj_trajectory"]
    obj_pos = obj_traj[:, :3, 3]
    obj_quat = rotmat_to_quat(obj_traj[:, :3, :3])[:, [1, 2, 3, 0]]

    gym = gymapi.acquire_gym()
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.substeps = 1
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.use_gpu = True
    sim_params.use_gpu_pipeline = False

    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    assert sim is not None

    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)

    env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 1)

    rh_asset = load_hand_asset(gym, sim, dexhand_rh)
    obj_asset = load_obj_asset(gym, sim, data["obj_urdf_path"])

    rh_actor = gym.create_actor(env, rh_asset, gymapi.Transform(), "dexhand_r", 0, 0)
    obj_actor = gym.create_actor(env, obj_asset, gymapi.Transform(), "obj", 0, 0)

    props = gym.get_actor_dof_properties(env, rh_actor)
    for i in range(len(props)):
        props["driveMode"][i] = gymapi.DOF_MODE_POS
        props["stiffness"][i] = 1e5
        props["damping"][i] = 50
    gym.set_actor_dof_properties(env, rh_actor, props)

    camera_cfg = gymapi.CameraProperties()
    camera_cfg.width = 1280
    camera_cfg.height = 720
    camera_cfg.horizontal_fov = 69.4
    camera = gym.create_camera_sensor(env, camera_cfg)
    gym.set_camera_location(camera, env, gymapi.Vec3(0.80, -0.00, 0.7), gymapi.Vec3(-1, -0.00, 0.3))

    gym.prepare_sim(sim)

    _root_state = gym.acquire_actor_root_state_tensor(sim)
    root_state = gymtorch.wrap_tensor(_root_state).view(-1, 13).clone()
    _dof_state = gym.acquire_dof_state_tensor(sim)
    dof_state = gymtorch.wrap_tensor(_dof_state).view(-1, 2).clone()

    view, proj = build_view_proj(camera_cfg.width, camera_cfg.height)

    frames_labeled = []
    for t in range(Tf):
        root_state[rh_actor, 0:3] = opt_wrist_pos[t]
        root_state[rh_actor, 3:7] = opt_wrist_quat[t]
        root_state[rh_actor, 7:] = 0.0

        root_state[obj_actor, 0:3] = obj_pos[t]
        root_state[obj_actor, 3:7] = obj_quat[t]
        root_state[obj_actor, 7:] = 0.0

        gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_state))

        dof_state[:, 0] = opt_dof_pos[t]
        dof_state[:, 1] = 0.0
        gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_state))

        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.render_all_camera_sensors(sim)

        img = gym.get_camera_image(sim, env, camera, gymapi.IMAGE_COLOR)
        img = img.reshape(camera_cfg.height, camera_cfg.width, 4)[..., :3].copy()

        rh_px = project_to_screen(opt_wrist_pos[t].cpu().numpy(), view, proj, camera_cfg.width, camera_cfg.height)
        RED = (255, 0, 0)
        cv2.putText(img, "RIGHT HAND (GRAB h0)", (rh_px[0] - 90, rh_px[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2, cv2.LINE_AA)
        cv2.circle(img, rh_px, 5, RED, -1)
        frames_labeled.append(img)

        if (t + 1) % 50 == 0 or t == Tf - 1:
            print(f"rendered frame {t + 1}/{Tf}")

    gym.destroy_sim(sim)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    imageio.mimwrite(OUT_PATH, frames_labeled, fps=FPS, quality=8)
    print(f"Saved {OUT_PATH} ({Tf} frames @ {FPS}fps)")


if __name__ == "__main__":
    main()
