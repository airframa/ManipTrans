import math
import os
import pickle
from isaacgym import gymapi, gymtorch, gymutil
import logging

logging.getLogger("gymapi").setLevel(logging.CRITICAL)
logging.getLogger("gymtorch").setLevel(logging.CRITICAL)
logging.getLogger("gymutil").setLevel(logging.CRITICAL)

import numpy as np
import pytorch_kinematics as pk
import torch
import torch.nn.functional as F
import trimesh
from termcolor import cprint

from main.dataset.factory import ManipDataFactory
from main.dataset.transform import (
    aa_to_quat,
    aa_to_rot6d,
    aa_to_rotmat,
    quat_to_rotmat,
    rot6d_to_aa,
    rot6d_to_quat,
    rot6d_to_rotmat,
    rotmat_to_aa,
    rotmat_to_quat,
    rotmat_to_rot6d,
)
from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory
from main.dataset.collision_sdf import load_or_build_object_sdf, query_sdf, sample_link_points


def pack_data(data, dexhand):
    packed_data = {}
    for k in data[0].keys():
        if k == "mano_joints":
            mano_joints = []
            for d in data:
                mano_joints.append(
                    torch.concat(
                        [
                            d[k][dexhand.to_hand(j_name)[0]]
                            for j_name in dexhand.body_names
                            if dexhand.to_hand(j_name)[0] != "wrist"
                        ],
                        dim=-1,
                    )
                )
            packed_data[k] = torch.stack(mano_joints).squeeze()
        elif type(data[0][k]) == torch.Tensor:
            packed_data[k] = torch.stack([d[k] for d in data]).squeeze()
        elif type(data[0][k]) == np.ndarray:
            packed_data[k] = np.stack([d[k] for d in data]).squeeze()
        else:
            packed_data[k] = [d[k] for d in data]
    return packed_data


def soft_clamp(x, lower, upper):
    return lower + torch.sigmoid(4 / (upper - lower) * (x - (lower + upper) / 2)) * (upper - lower)


class Mano2Dexhand:
    def __init__(self, args, dexhand, obj_urdf_path):
        self.gym = gymapi.acquire_gym()
        self.sim_params = gymapi.SimParams()
        self.dexhand = dexhand

        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

        self.headless = args.headless
        if self.headless:
            self.graphics_device_id = -1

        assert args.physics_engine == gymapi.SIM_PHYSX

        self.sim_params.substeps = 1
        self.sim_params.physx.solver_type = 1
        self.sim_params.physx.num_position_iterations = 4
        self.sim_params.physx.num_velocity_iterations = 1
        self.sim_params.physx.num_threads = args.num_threads
        self.sim_params.physx.use_gpu = args.use_gpu

        self.sim_params.use_gpu_pipeline = args.use_gpu_pipeline
        self.sim_device = args.sim_device if args.use_gpu_pipeline else "cpu"

        self.sim = self.gym.create_sim(
            args.compute_device_id, args.graphics_device_id, args.physics_engine, self.sim_params
        )

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)
        if not self.headless:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())

        asset_root = os.path.split(self.dexhand.urdf_path)[0]
        asset_file = os.path.split(self.dexhand.urdf_path)[1]

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        # asset_options.use_mesh_materials = True
        dexhand_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        self.chain = pk.build_chain_from_urdf(open(os.path.join(asset_root, asset_file)).read())
        self.chain = self.chain.to(dtype=torch.float32, device=self.sim_device)

        dexhand_dof_stiffness = torch.tensor(
            [10] * self.dexhand.n_dofs,
            dtype=torch.float,
            device=self.sim_device,
        )
        dexhand_dof_damping = torch.tensor(
            [1] * self.dexhand.n_dofs,
            dtype=torch.float,
            device=self.sim_device,
        )
        self.limit_info = {}
        asset_rh_dof_props = self.gym.get_asset_dof_properties(dexhand_asset)
        self.limit_info["rh"] = {
            "lower": np.asarray(asset_rh_dof_props["lower"]).copy().astype(np.float32),
            "upper": np.asarray(asset_rh_dof_props["upper"]).copy().astype(np.float32),
        }

        self.num_dexhand_bodies = self.gym.get_asset_rigid_body_count(dexhand_asset)
        self.num_dexhand_dofs = self.gym.get_asset_dof_count(dexhand_asset)

        dexhand_dof_props = self.gym.get_asset_dof_properties(dexhand_asset)
        rigid_shape_rh_props_asset = self.gym.get_asset_rigid_shape_properties(dexhand_asset)
        for element in rigid_shape_rh_props_asset:
            element.friction = 0.0001
            element.rolling_friction = 0.0001
            element.torsion_friction = 0.0001
        self.gym.set_asset_rigid_shape_properties(dexhand_asset, rigid_shape_rh_props_asset)

        self.dexhand_dof_lower_limits = []
        self.dexhand_dof_upper_limits = []
        self._dexhand_effort_limits = []
        self._dexhand_dof_speed_limits = []
        for i in range(self.num_dexhand_dofs):
            dexhand_dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            dexhand_dof_props["stiffness"][i] = dexhand_dof_stiffness[i]
            dexhand_dof_props["damping"][i] = dexhand_dof_damping[i]

            self.dexhand_dof_lower_limits.append(dexhand_dof_props["lower"][i])
            self.dexhand_dof_upper_limits.append(dexhand_dof_props["upper"][i])
            self._dexhand_effort_limits.append(dexhand_dof_props["effort"][i])
            self._dexhand_dof_speed_limits.append(dexhand_dof_props["velocity"][i])

        self.dexhand_dof_lower_limits = torch.tensor(self.dexhand_dof_lower_limits, device=self.sim_device)
        self.dexhand_dof_upper_limits = torch.tensor(self.dexhand_dof_upper_limits, device=self.sim_device)
        self._dexhand_effort_limits = torch.tensor(self._dexhand_effort_limits, device=self.sim_device)
        self._dexhand_dof_speed_limits = torch.tensor(self._dexhand_dof_speed_limits, device=self.sim_device)
        default_dof_state = np.ones(self.num_dexhand_dofs, gymapi.DofState.dtype)
        default_dof_state["pos"] *= np.pi / 50
        default_dof_state["pos"][8] = 0.8
        default_dof_state["pos"][9] = 0.05
        self.dexhand_default_dof_pos = default_dof_state
        self.dexhand_default_pose = gymapi.Transform()
        self.dexhand_default_pose.p = gymapi.Vec3(0, 0, 0)
        self.dexhand_default_pose.r = gymapi.Quat(0, 0, 0, 1)

        table_width_offset = 0.2
        mujoco2gym_transf = np.eye(4)
        mujoco2gym_transf[:3, :3] = aa_to_rotmat(np.array([0, 0, -np.pi / 2])) @ aa_to_rotmat(
            np.array([np.pi / 2, 0, 0])
        )
        table_pos = gymapi.Vec3(-table_width_offset / 2, 0, 0.4)
        self.dexhand_pose = gymapi.Transform()
        table_half_height = 0.015
        self._table_surface_z = table_pos.z + table_half_height
        mujoco2gym_transf[:3, 3] = np.array([0, 0, self._table_surface_z])
        self.mujoco2gym_transf = torch.tensor(mujoco2gym_transf, device=self.sim_device, dtype=torch.float32)

        self.num_envs = args.num_envs
        num_per_row = int(math.sqrt(self.num_envs))
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_options = gymapi.AssetOptions()
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        asset_options.thickness = 0.001
        asset_options.fix_base_link = True
        asset_options.vhacd_enabled = False
        asset_options.disable_gravity = True
        asset_options.density = 200

        current_asset = self.gym.load_asset(self.sim, *os.path.split(obj_urdf_path), asset_options)

        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(current_asset)
        for element in rigid_shape_props_asset:
            element.friction = 0.00001
        self.gym.set_asset_rigid_shape_properties(current_asset, rigid_shape_props_asset)

        self.envs = []
        self.hand_idxs = []

        for i in range(self.num_envs):
            # Create env
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env)
            dexhand_actor = self.gym.create_actor(
                env,
                dexhand_asset,
                self.dexhand_default_pose,
                "dexhand",
                i,
                (1 if self.dexhand.self_collision else 0),
            )

            # Set initial DOF states
            self.gym.set_actor_dof_states(env, dexhand_actor, self.dexhand_default_dof_pos, gymapi.STATE_ALL)

            # Set DOF control properties
            self.gym.set_actor_dof_properties(env, dexhand_actor, dexhand_dof_props)

            self.obj_actor = self.gym.create_actor(env, current_asset, gymapi.Transform(), "manip_obj", i, 0)

            scene_asset_options = gymapi.AssetOptions()
            scene_asset_options.fix_base_link = True
            for joint_vis_id, joint_name in enumerate(self.dexhand.body_names):
                joint_name = self.dexhand.to_hand(joint_name)[0]
                joint_point = self.gym.create_sphere(self.sim, 0.005, scene_asset_options)
                a = self.gym.create_actor(
                    env, joint_point, gymapi.Transform(), f"mano_joint_{joint_vis_id}", self.num_envs + 1, 0b1
                )
                if "index" in joint_name:
                    inter_c = 70
                elif "middle" in joint_name:
                    inter_c = 130
                elif "ring" in joint_name:
                    inter_c = 190
                elif "pinky" in joint_name:
                    inter_c = 250
                elif "thumb" in joint_name:
                    inter_c = 10
                else:
                    inter_c = 0
                if "tip" in joint_name:
                    c = gymapi.Vec3(inter_c / 255, 200 / 255, 200 / 255)
                elif "proximal" in joint_name:
                    c = gymapi.Vec3(200 / 255, inter_c / 255, 200 / 255)
                elif "intermediate" in joint_name:
                    c = gymapi.Vec3(200 / 255, 200 / 255, inter_c / 255)
                else:
                    c = gymapi.Vec3(100 / 255, 150 / 255, 200 / 255)
                self.gym.set_rigid_body_color(env, a, 0, gymapi.MESH_VISUAL, c)

        env_ptr = self.envs[0]
        dexhand_handle = 0
        self.dexhand_handles = {
            k: self.gym.find_actor_rigid_body_handle(env_ptr, dexhand_handle, k) for k in self.dexhand.body_names
        }
        self.dexhand_dof_handles = {
            k: self.gym.find_actor_dof_handle(env_ptr, dexhand_handle, k) for k in self.dexhand.dof_names
        }
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        _net_cf = self.gym.acquire_net_contact_force_tensor(self.sim)

        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self._net_cf = gymtorch.wrap_tensor(_net_cf).view(self.num_envs, -1, 3)
        self.q = self._dof_state[..., 0]
        self.qd = self._dof_state[..., 1]
        self._base_state = self._root_state[:, 0, :]

        self.isaac2chain_order = [
            self.gym.get_actor_dof_names(env_ptr, dexhand_handle).index(j)
            for j in self.chain.get_joint_parameter_names()
        ]

        self.mano_joint_points = [
            self._root_state[:, self.gym.find_actor_handle(env_ptr, f"mano_joint_{i}"), :]
            for i in range(len(self.dexhand.body_names))
        ]

        if not self.headless:
            cam_pos = gymapi.Vec3(4, 3, 3)
            cam_target = gymapi.Vec3(-4, -3, 0)
            middle_env = self.envs[self.num_envs // 2 + num_per_row // 2]
            self.gym.viewer_camera_look_at(self.viewer, middle_env, cam_pos, cam_target)

        self.gym.prepare_sim(self.sim)

        # --- collision-penetration term setup (opt-in via fitting()'s collision_weight; see
        # scripts/taco/build_and_validate_sdf.py for the SDF design/validation this reuses).
        # Only wired up for TACO's urdfs/<id>/<id>.urdf layout -- inert no-op for GRAB/OakInk-V2/
        # FAVOR objects, whose URDFs live elsewhere and were never measured for this.
        self.obj_urdf_path = obj_urdf_path
        self._taco_obj_id = None
        norm_path = os.path.normpath(obj_urdf_path)
        parts = norm_path.split(os.sep)
        if "taco" in parts and "urdfs" in parts:
            self._taco_obj_id = parts[parts.index("urdfs") + 1]
        side_letter = self.dexhand.body_names[0][0]  # "R" or "L"
        assert side_letter in ("R", "L"), f"unexpected body_name prefix: {self.dexhand.body_names[0]}"
        self._collision_side_letter = side_letter
        inspire_root = os.path.split(self.dexhand.urdf_path)[0]
        self._collision_link_local_points = sample_link_points(
            inspire_root, side_letter, n_points_per_link=25, device=self.sim_device
        )
        # NOTE: dexhand.to_hand() is NOT usable here -- it's a lossy one-to-many mapping
        # (hand2dex_mapping["thumb_proximal"] = ["thumb_proximal", "thumb_proximal_base"], both
        # R_/L_-prefixed) so both "R_thumb_proximal" and "R_thumb_proximal_base" reverse-map to
        # the SAME generic key "thumb_proximal", and "R_hand_base_link" maps to "wrist", not
        # "hand_base_link". Strip the fixed 2-char R_/L_ prefix directly instead, which matches
        # collision_sdf.COLLISION_LINK_STL's keys exactly and keeps every link distinct.
        self._collision_link_body_names = [
            k for k in self.dexhand.body_names if k[2:] in self._collision_link_local_points
        ]
        assert len(self._collision_link_body_names) == len(self._collision_link_local_points), (
            f"expected all {len(self._collision_link_local_points)} collision links to be found in "
            f"dexhand.body_names, got {len(self._collision_link_body_names)}: {self._collision_link_body_names}"
        )
        self._obj_sdf = None  # lazily built in fitting(), which knows the target grasp region

    def set_force_vis(self, env_ptr, part_k, has_force):
        self.gym.set_rigid_body_color(
            env_ptr,
            0,
            self.dexhand_handles[part_k],
            gymapi.MESH_VISUAL,
            (
                gymapi.Vec3(
                    1.0,
                    0.6,
                    0.6,
                )
                if has_force
                else gymapi.Vec3(1.0, 1.0, 1.0)
            ),
        )

    def fitting(
        self,
        max_iter,
        obj_trajectory,
        target_wrist_pos,
        target_wrist_rot,
        target_mano_joints,
        collision_weight=0.0,
        tracking_weight_scale=1.0,
        init_wrist_pos=None,
        init_wrist_rot=None,
        init_dof_pos=None,
        checkpoint_path=None,
        checkpoint_every=200,
        resume=False,
    ):

        assert target_mano_joints.shape[0] == self.num_envs
        target_wrist_pos = (self.mujoco2gym_transf[:3, :3] @ target_wrist_pos.T).T + self.mujoco2gym_transf[:3, 3]
        target_wrist_rot = self.mujoco2gym_transf[:3, :3] @ aa_to_rotmat(target_wrist_rot)
        target_mano_joints = target_mano_joints.view(-1, 3)
        target_mano_joints = (self.mujoco2gym_transf[:3, :3] @ target_mano_joints.T).T + self.mujoco2gym_transf[:3, 3]
        target_mano_joints = target_mano_joints.view(self.num_envs, -1, 3)

        obj_trajectory = self.mujoco2gym_transf @ obj_trajectory

        middle_pos = (target_mano_joints[:, 3] + target_wrist_pos) / 2
        obj_pos = obj_trajectory[:, :3, 3]
        offset = middle_pos - obj_pos
        offset = offset / torch.norm(offset, dim=-1, keepdim=True) * 0.2

        if collision_weight > 0:
            assert self._taco_obj_id is not None, (
                f"collision_weight>0 requested but {self.obj_urdf_path} isn't a recognized TACO "
                "urdfs/<id>/<id>.urdf object -- the SDF collision term is only wired up for TACO."
            )
            # Always the FULL-OBJECT grid, not a local/narrow-band grid: fitting() optimizes ALL
            # frames of the sequence simultaneously (one env per frame), and the grasp region
            # can move tens of mm across the trajectory (measured ~87x34x46mm spread for t3's
            # plate) -- a single local grid centered on one frame silently misses every other
            # frame's query points (grid_sample's border-clamp then reports a safely-negative,
            # WRONG value instead of erroring, so this failed silently: collision_loss stayed
            # exactly 0.0 for the whole plate optimization until this was caught by comparing
            # dof_pos to an unmodified baseline run). Full-object grids at 0.4mm with
            # area-scaled sample density were validated in build_and_validate_sdf.py to recover
            # the dominant penetration point within <0.04mm even for the plate (24x24cm
            # footprint, 62.5M voxels, 250MB) -- tractable for every TACO object measured so far.
            self._obj_sdf, self._obj_sdf_lo, self._obj_sdf_hi = load_or_build_object_sdf(
                self._taco_obj_id, voxel_mm=0.4, device=self.sim_device
            )
            cprint(f"[collision term] obj={self._taco_obj_id} weight={collision_weight}", "yellow")

        opt_wrist_pos = torch.tensor(
            init_wrist_pos if init_wrist_pos is not None else (target_wrist_pos + offset),
            device=self.sim_device,
            dtype=torch.float32,
            requires_grad=True,
        )
        opt_wrist_rot = torch.tensor(
            init_wrist_rot if init_wrist_rot is not None else rotmat_to_rot6d(target_wrist_rot),
            device=self.sim_device,
            dtype=torch.float32,
            requires_grad=True,
        )
        opt_dof_pos = torch.tensor(
            init_dof_pos if init_dof_pos is not None else self.dexhand_default_dof_pos["pos"][None].repeat(self.num_envs, axis=0),
            device=self.sim_device,
            dtype=torch.float32,
            requires_grad=True,
        )
        opti = torch.optim.Adam(
            [{"params": [opt_wrist_pos, opt_wrist_rot], "lr": 0.0008}, {"params": [opt_dof_pos], "lr": 0.0004}]
        )

        start_iter = 0
        if resume and checkpoint_path is not None and os.path.exists(checkpoint_path):
            # Recovery for the recurring mid-run crash (native-level SIGSEGV, no Python
            # traceback via faulthandler, no accessible dmesg/gdb to root-cause further --
            # happens across collision_weight=0 and >0 alike, so it's not specific to the
            # collision term's added ops). Since gradients are per-env independent and Adam's
            # own state is per-element too (no cross-iteration coupling beyond its own moment
            # estimates), resuming from a periodic checkpoint reproduces the same trajectory a
            # single uninterrupted run would have taken, just possibly across multiple process
            # restarts.
            ckpt = torch.load(checkpoint_path, map_location=self.sim_device)
            with torch.no_grad():
                opt_wrist_pos.copy_(ckpt["opt_wrist_pos"])
                opt_wrist_rot.copy_(ckpt["opt_wrist_rot"])
                opt_dof_pos.copy_(ckpt["opt_dof_pos"])
            opti.load_state_dict(ckpt["optimizer"])
            start_iter = ckpt["iter"]
            cprint(f"[checkpoint] resumed from {checkpoint_path} at iter {start_iter}", "yellow")

        weight = []
        for k in self.dexhand.body_names:
            k = self.dexhand.to_hand(k)[0]
            if "tip" in k:
                if "index" in k:
                    weight.append(20)
                elif "middle" in k:
                    weight.append(10)
                elif "ring" in k:
                    weight.append(7)
                elif "pinky" in k:
                    weight.append(5)
                elif "thumb" in k:
                    weight.append(25)
                else:
                    raise ValueError
            elif "proximal" in k:
                weight.append(1)
            elif "intermediate" in k:
                weight.append(1)
            else:
                weight.append(1)
        weight = torch.tensor(weight, device=self.sim_device, dtype=torch.float32)
        iter = start_iter
        past_loss = 1e10
        while (self.headless and iter < max_iter) or (
            not self.headless and not self.gym.query_viewer_has_closed(self.viewer)
        ):
            iter += 1

            opt_wrist_quat = rot6d_to_quat(opt_wrist_rot)[:, [1, 2, 3, 0]]
            opt_wrist_rotmat = rot6d_to_rotmat(opt_wrist_rot)
            self._root_state[:, 0, :3] = opt_wrist_pos.detach()
            self._root_state[:, 0, 3:7] = opt_wrist_quat.detach()
            self._root_state[:, 0, 7:] = torch.zeros_like(self._root_state[:, 0, 7:])
            self._root_state[:, self.obj_actor, :3] = obj_trajectory[:, :3, 3]
            self._root_state[:, self.obj_actor, 3:7] = rotmat_to_quat(obj_trajectory[:, :3, :3])[:, [1, 2, 3, 0]]

            opt_dof_pos_clamped = torch.clamp(opt_dof_pos, self.dexhand_dof_lower_limits, self.dexhand_dof_upper_limits)

            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(opt_dof_pos_clamped))
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self._root_state))

            # Step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            if not self.headless:
                self.gym.step_graphics(self.sim)

            # Update jacobian and mass matrix
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_net_contact_force_tensor(self.sim)
            # Step rendering
            if not self.headless:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, False)
            self.gym.sync_frame_time(self.sim)

            isaac_joints = torch.stack(
                [self._rigid_body_state[:, self.dexhand_handles[k], :3] for k in self.dexhand.body_names],
                dim=1,
            )

            ret = self.chain.forward_kinematics(opt_dof_pos_clamped[:, self.isaac2chain_order])
            pk_joints = torch.stack([ret[k].get_matrix()[:, :3, 3] for k in self.dexhand.body_names], dim=1)
            pk_joints = (rot6d_to_rotmat(opt_wrist_rot) @ pk_joints.transpose(-1, -2)).transpose(
                -1, -2
            ) + opt_wrist_pos[:, None]

            target_joints = torch.cat([target_wrist_pos[:, None], target_mano_joints], dim=1)
            for k in range(len(self.mano_joint_points)):
                self.mano_joint_points[k][:, :3] = target_joints[:, k]
            tracking_loss = tracking_weight_scale * torch.mean(torch.norm(pk_joints - target_joints, dim=-1) * weight[None])

            collision_loss = torch.zeros((), device=self.sim_device)
            if collision_weight > 0:
                wrist_R = rot6d_to_rotmat(opt_wrist_rot)  # (nE, 3, 3)
                obj_R = obj_trajectory[:, :3, :3]  # (nE, 3, 3), fixed (not optimized)
                obj_t = obj_trajectory[:, :3, 3]  # (nE, 3)
                per_link_max_pen = []
                for body_name in self._collision_link_body_names:
                    link_key = body_name[2:]  # strip R_/L_ prefix directly -- see __init__ note on why not to_hand()
                    local_pts = self._collision_link_local_points[link_key]  # (nP, 3), fixed

                    chain_mat = ret[body_name].get_matrix()  # (nE, 4, 4)
                    chain_R = chain_mat[:, :3, :3]
                    chain_t = chain_mat[:, :3, 3]

                    # link-local -> chain frame -> wrist/world frame -> object-local frame
                    pts_chain = local_pts[None] @ chain_R.transpose(-1, -2) + chain_t[:, None, :]  # (nE,nP,3)
                    pts_world = (wrist_R @ pts_chain.transpose(-1, -2)).transpose(-1, -2) + opt_wrist_pos[:, None, :]
                    pts_obj_local = (pts_world - obj_t[:, None, :]) @ obj_R  # (nE,nP,3); obj_R orthonormal

                    sdf_vals = query_sdf(self._obj_sdf, self._obj_sdf_lo, self._obj_sdf_hi, pts_obj_local)  # (nE,nP)
                    per_link_max_pen.append(F.relu(sdf_vals).max(dim=-1).values)  # (nE,) one-sided hinge
                per_env_worst = torch.stack(per_link_max_pen, dim=-1).amax(dim=-1)  # (nE,): worst link per env
                # Reverted from top-k/softmax back to plain mean over envs: opt_wrist_pos/
                # opt_wrist_rot/opt_dof_pos are independent per-env parameters with no shared
                # term anywhere in this loop (no smoothness/temporal regularizer), so gradients
                # are per-env independent -- a mean cannot let the optimizer "trade" one frame's
                # quality for another's (d(mean of independent terms)/d(param_i) only involves
                # param_i, scaled by a constant 1/nE for every env alike). The earlier "frame 100
                # got sacrificed" diagnosis was wrong for that reason. top-k/softmax instead
                # concentrated ~all gradient onto whichever 1-few envs were currently worst,
                # starving the rest of gradient entirely (confirmed: frames 60/130 never
                # improved) and caused a reproducible mid-run segfault from the resulting
                # discontinuous env selection. Mean gives every frame its full, undiluted
                # per-env gradient direction every iteration.
                collision_loss = per_env_worst.mean()

            loss = tracking_loss + collision_weight * collision_loss

            if collision_weight > 0 and iter == 1:
                # Weight calibration diagnostic: report raw term values AND gradient magnitudes
                # (w.r.t. opt_dof_pos) BEFORE combining, so the collision term's influence can
                # be judged directly rather than guessed from the loss ratio alone.
                g_track = torch.autograd.grad(tracking_loss, opt_dof_pos, retain_graph=True)[0]
                if collision_loss.requires_grad:
                    g_coll = torch.autograd.grad(collision_loss, opt_dof_pos, retain_graph=True)[0]
                    g_coll_norm = g_coll.norm().item()
                else:
                    g_coll_norm = 0.0
                cprint(
                    f"[iter1 diag] tracking_loss={tracking_loss.item():.5f} (grad_norm={g_track.norm().item():.5f})  "
                    f"collision_loss={collision_loss.item():.5f} (grad_norm={g_coll_norm:.5f})  "
                    f"weighted_collision={(collision_weight*collision_loss).item():.5f}  "
                    f"weighted_collision_grad_norm={collision_weight*g_coll_norm:.5f}",
                    "cyan",
                )

            opti.zero_grad()
            loss.backward()
            opti.step()

            if checkpoint_path is not None and iter % checkpoint_every == 0:
                torch.save(
                    {
                        "iter": iter,
                        "opt_wrist_pos": opt_wrist_pos.detach(),
                        "opt_wrist_rot": opt_wrist_rot.detach(),
                        "opt_dof_pos": opt_dof_pos.detach(),
                        "optimizer": opti.state_dict(),
                    },
                    checkpoint_path,
                )

            if iter % 100 == 0:
                cprint(f"{iter} {loss.item()} (tracking={tracking_loss.item():.5f} collision={collision_loss.item():.5f})", "green")
                if iter > 1 and past_loss - loss.item() < 1e-5:
                    break
                past_loss = loss.item()

        to_dump = {
            "opt_wrist_pos": opt_wrist_pos.detach().cpu().numpy(),
            "opt_wrist_rot": rot6d_to_aa(opt_wrist_rot).detach().cpu().numpy(),
            "opt_dof_pos": opt_dof_pos_clamped.detach().cpu().numpy(),
            "opt_joints_pos": isaac_joints.detach().cpu().numpy(),
        }

        if not self.headless:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
        return to_dump


if __name__ == "__main__":
    _parser = gymutil.parse_arguments(
        description="Mano to Dexhand",
        headless=True,
        custom_parameters=[
            {
                "name": "--iter",
                "type": int,
                "default": 4000,
            },
            {
                "name": "--data_idx",
                "type": str,
                "default": "1906",
            },
            {
                "name": "--dexhand",
                "type": str,
                "default": "inspire",
            },
            {
                "name": "--side",
                "type": str,
                "default": "right",
            },
            {
                "name": "--collision_weight",
                "type": float,
                "default": 0.0,
            },
            {
                "name": "--warmstart_from",
                "type": str,
                "default": "",
            },
            {
                "name": "--checkpoint_path",
                "type": str,
                "default": "",
            },
            {
                "name": "--resume",
                "type": int,
                "default": 0,
            },
        ],
    )

    dexhand = DexHandFactory.create_hand(_parser.dexhand, _parser.side)

    def run(parser, idx):

        dataset_type = ManipDataFactory.dataset_type(idx)
        demo_d = ManipDataFactory.create_data(
            manipdata_type=dataset_type,
            side=parser.side,
            device="cuda:0",
            mujoco2gym_transf=torch.eye(4, device="cuda:0"),
            dexhand=dexhand,
            verbose=False,
        )

        demo_data = pack_data([demo_d[idx]], dexhand)

        parser.num_envs = demo_data["mano_joints"].shape[0]

        mano2inspire = Mano2Dexhand(parser, dexhand, demo_data["obj_urdf_path"][0])

        warmstart_kwargs = {}
        if parser.warmstart_from:
            # Warm-start opt_wrist_pos/opt_wrist_rot/opt_dof_pos from an existing (e.g.
            # pre-collision-term) retargeting pkl instead of the hardcoded default pose --
            # confirmed via diag_warmstart_test.py that this alone resolves the collision term's
            # tunneling-through-thin-objects failure mode (a one-sided hinge from a generic
            # default pose can push a link straight through a thin object to the far side, where
            # the SDF reads "clear" even though the swept path interpenetrated; starting already
            # correctly-sided next to a near-feasible solution avoids ever needing to cross).
            with open(parser.warmstart_from, "rb") as f:
                warmstart = pickle.load(f)
            warmstart_kwargs["init_wrist_pos"] = warmstart["opt_wrist_pos"]
            warmstart_kwargs["init_wrist_rot"] = (
                aa_to_rot6d(torch.tensor(warmstart["opt_wrist_rot"], device="cuda:0", dtype=torch.float32)).cpu().numpy()
            )
            warmstart_kwargs["init_dof_pos"] = warmstart["opt_dof_pos"]
            cprint(f"[warmstart] initializing from {parser.warmstart_from}", "yellow")

        to_dump = mano2inspire.fitting(
            parser.iter,
            demo_data["obj_trajectory"],
            demo_data["wrist_pos"],
            demo_data["wrist_rot"],
            demo_data["mano_joints"].view(parser.num_envs, -1, 3),
            collision_weight=parser.collision_weight,
            checkpoint_path=parser.checkpoint_path or None,
            resume=bool(parser.resume),
            **warmstart_kwargs,
        )

        if dataset_type == "oakink2":
            dump_path = f"data/retargeting/OakInk-v2/mano2{str(dexhand)}/{os.path.split(demo_data['data_path'][0])[-1].replace('.pkl', f'@{idx[-1]}.pkl')}"
        elif dataset_type == "favor":
            dump_path = (
                f"data/retargeting/favor_pass1/mano2{str(dexhand)}/{os.path.split(demo_data['data_path'][0])[-1]}"
            )
        elif dataset_type == "grabdemo":
            dump_path = f"data/retargeting/grab_demo/mano2{str(dexhand)}/{os.path.split(demo_data['data_path'][0])[-1].replace('.npy', '.pkl')}"
        elif dataset_type == "taco":
            dump_path = f"data/retargeting/taco/mano2{str(dexhand)}/{os.path.split(demo_data['data_path'][0])[-1]}.pkl"
        elif dataset_type == "grabdemo2":
            dump_path = f"data/retargeting/grab_demo_long/mano2{str(dexhand)}/{os.path.split(demo_data['data_path'][0])[-1].replace('.npy', '.pkl')}"
        elif dataset_type == "oakink2_mirrored":
            dump_path = f"data/retargeting/OakInk-v2-mirrored/mano2{str(dexhand)}/{os.path.split(demo_data['data_path'][0])[-1].replace('.pkl', f'@{idx[-1]}.pkl')}"
        elif dataset_type == "favor_mirrored":
            dump_path = f"data/retargeting/favor_pass1-mirrored/mano2{str(dexhand)}/{os.path.split(demo_data['data_path'][0])[-1]}"
        else:
            raise ValueError("Unsupported dataset type")

        os.makedirs(os.path.dirname(dump_path), exist_ok=True)
        with open(dump_path, "wb") as f:
            pickle.dump(to_dump, f)
        cprint(f"[DONE] saved {dump_path}", "magenta")

    run(_parser, _parser.data_idx)
