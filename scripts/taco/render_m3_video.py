"""
Renders the M3 kinematic-retargeting result (both hands + both objects) as a
single mp4, for visual sharing (not numeric verification -- see
taco_verify_m3.py / compare_verify_m3.py for that).

This is a pure kinematic replay: no RL, no physics optimization. It loads the
already-computed retargeting output from
    data/retargeting/taco/mano2inspire_{rh,lh}/20230919_026@{right,left}.pkl
(opt_wrist_pos, opt_wrist_rot, opt_dof_pos) and directly overrides actor root
state + DOF state every frame, then captures an Isaac Gym camera sensor image.

Reused from the existing pipeline rather than built from scratch:
- The headless-camera-sensor pattern (CameraProperties + create_camera_sensor +
  set_camera_location + render_all_camera_sensors), from
  maniptrans_envs/lib/envs/core/vec_task.py's create_camera / set_camera.
- The exact bimanual camera framing (1280x720, cam_pos=(0.80,0,0.7),
  cam_target=(-1,0,0.3)) used for `record=True` runs in
  maniptrans_envs/lib/envs/tasks/dexhandmanip_bih.py's create_camera -- so this
  looks the same as prior wandb training-demo videos.
- The sim/asset/root-state/dof-state setup pattern from
  main/dataset/mano2dexhand.py's Mano2Dexhand.__init__/.fitting() (asset
  loading options, table transform, root_state/dof_state tensor layout) --
  minus the optimization loop, since we already have the converged result.

Must be run from the repo root: all data paths (data/taco/..., data/retargeting/...,
outputs/...) are resolved relative to CWD, not to this file's location.

Usage (from repo root): python scripts/taco/render_m3_video.py [seq_idx] [tag]
  seq_idx: index into taco_dataset_dexhand.SEQUENCES (default 0)
  tag: short name used in output filenames (default derived from seq_idx)
"""

from isaacgym import gymapi, gymtorch  # noqa: F401 -- must be first import in the whole process

import math
import os
import pickle
import sys

import cv2
import imageio
import numpy as np
import torch

from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory
import maniptrans_envs.lib.envs.dexhands  # noqa: F401 -- triggers hand auto-registration

from main.dataset.taco_dataset_dexhand import TACORightData, TACOLeftData, SEQUENCES
from main.dataset.transform import aa_to_rotmat, aa_to_quat

SEQ_IDX = int(sys.argv[1]) if len(sys.argv) > 1 else 0
SEQ_NAME = SEQUENCES[SEQ_IDX]["seq_name"]
_triplet_parts = [p.strip() for p in SEQUENCES[SEQ_IDX]["triplet"].strip("()").split(",")]
TOOL_NAME, TARGET_NAME = _triplet_parts[1], _triplet_parts[2]  # triplet = (action, tool, target)
TAG = sys.argv[2] if len(sys.argv) > 2 else f"t{SEQ_IDX}"
OUT_PATH = f"outputs/m3_{TAG}_preview.mp4"
OUT_PATH_LABELED = f"outputs/m3_{TAG}_preview_labeled.mp4"
FPS = 60  # matches the effective 60Hz (skip=2) rate our retargeted data is at

# Same fixed camera used for the bimanual record=True view (dexhandmanip_bih.py's
# create_camera). Static across the whole video, so we build one view/projection
# matrix and reuse it every frame to project wrist positions to screen space for
# the on-screen hand labels.
CAM_POS = np.array([0.80, -0.00, 0.7])
CAM_TARGET = np.array([-1, -0.00, 0.3])
CAM_UP = np.array([0.0, 0.0, 1.0])  # world up_axis = Z
CAM_HFOV_DEG = 69.4


def build_view_proj(width, height, near=0.01, far=100.0):
    # Standard OpenGL-style look-at + perspective, matching Isaac Gym's camera
    # sensor convention (camera looks down -Z in its own view space).
    forward = CAM_TARGET - CAM_POS
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, CAM_UP)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)

    view = np.eye(4)
    view[0, :3] = right
    view[1, :3] = true_up
    view[2, :3] = -forward
    view[:3, 3] = -view[:3, :3] @ CAM_POS

    aspect = width / height
    fov_h = math.radians(CAM_HFOV_DEG)
    fov_v = 2 * math.atan(math.tan(fov_h / 2) / aspect)
    f = 1.0 / math.tan(fov_v / 2)
    proj = np.array(
        [
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
            [0, 0, -1, 0],
        ]
    )
    return view, proj


def project_to_screen(point_xyz, view, proj, width, height):
    p = np.array([point_xyz[0], point_xyz[1], point_xyz[2], 1.0])
    clip = proj @ (view @ p)
    ndc = clip[:3] / clip[3]
    screen_x = (ndc[0] * 0.5 + 0.5) * width
    screen_y = (1.0 - (ndc[1] * 0.5 + 0.5)) * height
    return int(screen_x), int(screen_y)


def build_table_transf(device):
    # Reproduces mano2dexhand.py Mano2Dexhand.__init__ lines 166-175 exactly --
    # deterministic, same frame the retargeting optimization ran in.
    transf = torch.eye(4, dtype=torch.float32, device=device)
    transf[:3, :3] = aa_to_rotmat(torch.tensor([0.0, 0.0, -np.pi / 2], device=device)) @ aa_to_rotmat(
        torch.tensor([np.pi / 2, 0.0, 0.0], device=device)
    )
    table_pos_z = 0.4
    table_half_height = 0.015
    transf[:3, 3] = torch.tensor([0.0, 0.0, table_pos_z + table_half_height], device=device)
    return transf


def load_hand_asset(gym, sim, dexhand):
    asset_root, asset_file = os.path.split(dexhand.urdf_path)
    opts = gymapi.AssetOptions()
    opts.fix_base_link = False
    opts.disable_gravity = True
    opts.flip_visual_attachments = False
    opts.collapse_fixed_joints = False
    opts.default_dof_drive_mode = gymapi.DOF_MODE_POS
    return gym.load_asset(sim, asset_root, asset_file, opts)


def load_obj_asset(gym, sim, urdf_path):
    opts = gymapi.AssetOptions()
    opts.fix_base_link = False
    opts.disable_gravity = True
    opts.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
    opts.thickness = 0.001
    opts.vhacd_enabled = False
    return gym.load_asset(sim, *os.path.split(urdf_path), opts)


def main():
    device = "cuda:0"
    identity = torch.eye(4, dtype=torch.float32, device=device)

    dexhand_rh = DexHandFactory.create_hand("inspire", "right")
    dexhand_lh = DexHandFactory.create_hand("inspire", "left")

    # our loader, identity mujoco2gym_transf -- gives us obj_urdf_path, obj_trajectory
    # (untransformed), matching what taco_verify_m3.py already validated
    fdata_rh = TACORightData(mujoco2gym_transf=identity, device=device, dexhand=dexhand_rh)
    fdata_lh = TACOLeftData(mujoco2gym_transf=identity, device=device, dexhand=dexhand_lh)
    data_rh = fdata_rh[f"t{SEQ_IDX}"]
    data_lh = fdata_lh[f"t{SEQ_IDX}"]

    table_transf = build_table_transf(device)

    with open(f"data/retargeting/taco/mano2{str(dexhand_rh)}/{SEQ_NAME}@right.pkl", "rb") as f:
        opt_rh = pickle.load(f)
    with open(f"data/retargeting/taco/mano2{str(dexhand_lh)}/{SEQ_NAME}@left.pkl", "rb") as f:
        opt_lh = pickle.load(f)

    Tf = opt_rh["opt_wrist_pos"].shape[0]
    assert opt_lh["opt_wrist_pos"].shape[0] == Tf

    opt_rh_wrist_pos = torch.tensor(opt_rh["opt_wrist_pos"], dtype=torch.float32, device=device)
    opt_rh_wrist_quat = aa_to_quat(torch.tensor(opt_rh["opt_wrist_rot"], dtype=torch.float32, device=device))[:, [1, 2, 3, 0]]
    opt_rh_dof_pos = torch.tensor(opt_rh["opt_dof_pos"], dtype=torch.float32, device=device)

    opt_lh_wrist_pos = torch.tensor(opt_lh["opt_wrist_pos"], dtype=torch.float32, device=device)
    opt_lh_wrist_quat = aa_to_quat(torch.tensor(opt_lh["opt_wrist_rot"], dtype=torch.float32, device=device))[:, [1, 2, 3, 0]]
    opt_lh_dof_pos = torch.tensor(opt_lh["opt_dof_pos"], dtype=torch.float32, device=device)

    # object trajectories: transform our (identity-frame) trajectories into the
    # same table frame the retargeting optimization used, exactly like the
    # tip-error comparison in taco_verify_m3.py does for MANO targets.
    obj_traj_tool = table_transf[None] @ data_rh["obj_trajectory"]  # (Tf, 4, 4) -- brush (035)
    obj_traj_target = table_transf[None] @ data_lh["obj_trajectory"]  # (Tf, 4, 4) -- pan (057)

    def mat_to_quat_xyzw(rotmat):
        from main.dataset.transform import rotmat_to_quat

        return rotmat_to_quat(rotmat)[:, [1, 2, 3, 0]]

    obj_tool_pos = obj_traj_tool[:, :3, 3]
    obj_tool_quat = mat_to_quat_xyzw(obj_traj_tool[:, :3, :3])
    obj_target_pos = obj_traj_target[:, :3, 3]
    obj_target_quat = mat_to_quat_xyzw(obj_traj_target[:, :3, :3])

    # --- Isaac Gym sim setup (single env, no physics optimization needed) ---
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

    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)  # graphics_device_id=0 (not -1) -- needed for camera rendering
    assert sim is not None

    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)

    env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 1)

    rh_asset = load_hand_asset(gym, sim, dexhand_rh)
    lh_asset = load_hand_asset(gym, sim, dexhand_lh)
    tool_asset = load_obj_asset(gym, sim, data_rh["obj_urdf_path"])
    target_asset = load_obj_asset(gym, sim, data_lh["obj_urdf_path"])

    rh_actor = gym.create_actor(env, rh_asset, gymapi.Transform(), "dexhand_r", 0, 0)
    lh_actor = gym.create_actor(env, lh_asset, gymapi.Transform(), "dexhand_l", 0, 0)
    tool_actor = gym.create_actor(env, tool_asset, gymapi.Transform(), "tool", 0, 0)
    target_actor = gym.create_actor(env, target_asset, gymapi.Transform(), "target", 0, 0)

    # position-control DOF drive, high stiffness for exact kinematic tracking
    # (we also directly overwrite dof_state every frame below, belt-and-suspenders)
    for actor, dexhand in [(rh_actor, dexhand_rh), (lh_actor, dexhand_lh)]:
        props = gym.get_actor_dof_properties(env, actor)
        for i in range(len(props)):
            props["driveMode"][i] = gymapi.DOF_MODE_POS
            props["stiffness"][i] = 1e5
            props["damping"][i] = 50
        gym.set_actor_dof_properties(env, actor, props)

    # camera: exact bimanual record=True framing from dexhandmanip_bih.py's create_camera
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

    n_dof_rh = dexhand_rh.n_dofs
    n_dof_lh = dexhand_lh.n_dofs

    view, proj = build_view_proj(camera_cfg.width, camera_cfg.height)

    frames = []
    frames_labeled = []
    for t in range(Tf):
        root_state[rh_actor, 0:3] = opt_rh_wrist_pos[t]
        root_state[rh_actor, 3:7] = opt_rh_wrist_quat[t]
        root_state[rh_actor, 7:] = 0.0

        root_state[lh_actor, 0:3] = opt_lh_wrist_pos[t]
        root_state[lh_actor, 3:7] = opt_lh_wrist_quat[t]
        root_state[lh_actor, 7:] = 0.0

        root_state[tool_actor, 0:3] = obj_tool_pos[t]
        root_state[tool_actor, 3:7] = obj_tool_quat[t]
        root_state[tool_actor, 7:] = 0.0

        root_state[target_actor, 0:3] = obj_target_pos[t]
        root_state[target_actor, 3:7] = obj_target_quat[t]
        root_state[target_actor, 7:] = 0.0

        gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_state))

        dof_pos_t = torch.cat([opt_rh_dof_pos[t], opt_lh_dof_pos[t]])
        dof_state[:, 0] = dof_pos_t
        dof_state[:, 1] = 0.0
        gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_state))

        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.render_all_camera_sensors(sim)

        img = gym.get_camera_image(sim, env, camera, gymapi.IMAGE_COLOR)
        img = img.reshape(camera_cfg.height, camera_cfg.width, 4)[..., :3]
        frames.append(img.copy())

        # burned-in labels, positioned via actual camera projection of each
        # wrist's 3D position -- not a static guess -- so they track the hands
        # and unambiguously resolve which rendered actor is which.
        labeled = img.copy()
        rh_px = project_to_screen(opt_rh_wrist_pos[t].cpu().numpy(), view, proj, camera_cfg.width, camera_cfg.height)
        lh_px = project_to_screen(opt_lh_wrist_pos[t].cpu().numpy(), view, proj, camera_cfg.width, camera_cfg.height)
        # img/labeled are RGB arrays (Isaac Gym IMAGE_COLOR, RGBA with alpha dropped),
        # so cv2 color tuples here are (R, G, B), not cv2's usual BGR.
        RED = (255, 0, 0)
        GREEN = (0, 255, 0)
        cv2.putText(labeled, f"RIGHT HAND (tool/{TOOL_NAME})", (rh_px[0] - 90, rh_px[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2, cv2.LINE_AA)
        cv2.circle(labeled, rh_px, 5, RED, -1)
        cv2.putText(labeled, f"LEFT HAND (target/{TARGET_NAME})", (lh_px[0] - 90, lh_px[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2, cv2.LINE_AA)
        cv2.circle(labeled, lh_px, 5, GREEN, -1)
        frames_labeled.append(labeled)

        if (t + 1) % 100 == 0 or t == Tf - 1:
            print(f"rendered frame {t + 1}/{Tf}")

    gym.destroy_sim(sim)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    imageio.mimwrite(OUT_PATH, frames, fps=FPS, quality=8)
    print(f"Saved {OUT_PATH} ({Tf} frames @ {FPS}fps)")
    imageio.mimwrite(OUT_PATH_LABELED, frames_labeled, fps=FPS, quality=8)
    print(f"Saved {OUT_PATH_LABELED} ({Tf} frames @ {FPS}fps)")


if __name__ == "__main__":
    main()
