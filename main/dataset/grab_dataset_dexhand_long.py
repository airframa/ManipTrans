"""
Second, longer GRAB baseline for the M3 methodology check (Part 2): the same
underlying GRAB demo motion (102) at its full 108-frame length, rather than
the 60-frame trimmed clip grab_dataset_dexhand.py uses. Already present on
disk (data/grab_demo/102/102_sv_dict_st_0_ed_108.npy, extracted from the
existing grab.zip) -- no new data downloaded.

This is a new file, not a modification of grab_dataset_dexhand.py (CLAUDE.md:
do not modify existing GRAB / OakInk-V2 / FAVOR loaders). Logic is duplicated
from GrabDemoDexHand almost verbatim, pointed at the longer npy file and
registered under a distinct ManipDataFactory key/index prefix ("h0") so it
doesn't collide with "g0".
"""

import os
from functools import lru_cache

import numpy as np
import torch
import trimesh
from pytorch3d.structures import Meshes

from manotorch.manolayer import ManoLayer

from main.dataset.transform import aa_to_rotmat
from .base import ManipData
from .decorators import register_manipdata


@register_manipdata("grabdemo2_rh")
class GrabDemoLongDexHand(ManipData):
    def __init__(
        self,
        *,
        data_dir: str = "data/grab_demo/102",
        split: str = "all",
        skip: int = 1,
        device="cuda:0",
        mujoco2gym_transf=None,
        max_seq_len=int(1e10),
        dexhand=None,
        **kwargs,
    ):
        super().__init__(
            data_dir=data_dir,
            split=split,
            skip=skip,
            device=device,
            mujoco2gym_transf=mujoco2gym_transf,
            max_seq_len=max_seq_len,
            dexhand=dexhand,
            **kwargs,
        )
        self.manolayer = ManoLayer(
            rot_mode="axisang",
            side="right",
            center_idx=None,
            mano_assets_root="data/mano_v1_2",
            use_pca=False,
            flat_hand_mean=True,
        ).to(device)

        self.data_pathes = [os.path.join(self.data_dir, "102_sv_dict_st_0_ed_108.npy")]

        self.device = device

        # identical to grab_dataset_dexhand.py's transf_offset -- same underlying
        # motion capture source/convention, just a different-length clip of it
        transf_offset = np.eye(4)
        transf_offset[:3, :3] = aa_to_rotmat(np.array([-np.pi / 2, 0, 0])) @ aa_to_rotmat(np.array([0, 0, np.pi / 2]))
        transf_offset[:3, 3] = np.array([0.0, 0.018, 0.0])

        self.transf_offset = torch.tensor(transf_offset, dtype=torch.float32, device=mujoco2gym_transf.device)

        self.mujoco2gym_transf = mujoco2gym_transf @ self.transf_offset
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data_pathes)

    @lru_cache(maxsize=None)
    def __getitem__(self, idx):
        assert idx == "h0", "Only one long-clip sequence available -- index must be 'h0'"
        idx = int(idx[1:])
        assert self.mujoco2gym_transf is not None

        data = np.load(self.data_pathes[idx], allow_pickle=True).item()
        obj_mesh = trimesh.load(os.path.join(self.data_dir, "102_obj.obj"), process=False)

        length = len(data["object_global_orient"])
        obj_pose = np.eye(4)[None].repeat(length, axis=0)
        obj_pose[:, :3, :3] = aa_to_rotmat(data["object_global_orient"]).transpose(0, 2, 1)
        obj_pose[:, :3, 3] = data["object_transl"]
        obj_pose = torch.tensor(obj_pose, device=self.device)
        hand_rot = torch.tensor(data["rhand_global_orient_gt"], device=self.device)
        mano_out_verts = torch.tensor(data["rhand_verts"], device=self.device)
        mano_out_joints = torch.matmul(self.manolayer.th_J_regressor, mano_out_verts)

        wrist_pos = mano_out_joints.detach()[:, 0]
        middle_pos = mano_out_joints.detach()[:, 4]
        wrist_pos = wrist_pos - (middle_pos - wrist_pos) * 0.25

        mano_joints = {
            "index_proximal": mano_out_joints.detach()[:, 1],
            "index_intermediate": mano_out_joints.detach()[:, 2],
            "index_distal": mano_out_joints.detach()[:, 3],
            "index_tip": mano_out_verts[:, 353].detach(),
            "middle_proximal": mano_out_joints.detach()[:, 4],
            "middle_intermediate": mano_out_joints.detach()[:, 5],
            "middle_distal": mano_out_joints.detach()[:, 6],
            "middle_tip": mano_out_verts[:, 467].detach(),
            "pinky_proximal": mano_out_joints.detach()[:, 7],
            "pinky_intermediate": mano_out_joints.detach()[:, 8],
            "pinky_distal": mano_out_joints.detach()[:, 9],
            "pinky_tip": mano_out_verts[:, 695].detach(),
            "ring_proximal": mano_out_joints.detach()[:, 10],
            "ring_intermediate": mano_out_joints.detach()[:, 11],
            "ring_distal": mano_out_joints.detach()[:, 12],
            "ring_tip": mano_out_verts[:, 576].detach(),
            "thumb_proximal": mano_out_joints.detach()[:, 13],
            "thumb_intermediate": mano_out_joints.detach()[:, 14],
            "thumb_distal": mano_out_joints.detach()[:, 15],
            "thumb_tip": mano_out_verts[:, 766].detach(),
        }

        inspire_rot_offset = self.dexhand.relative_rotation
        wrist_rot = aa_to_rotmat(hand_rot) @ torch.tensor(
            np.repeat(inspire_rot_offset[None], length, axis=0), device=self.device
        )

        mesh = Meshes(
            verts=torch.from_numpy(obj_mesh.vertices[None, ...]).float(),
            faces=torch.from_numpy(obj_mesh.faces[None, ...]).float(),
        )
        rs_verts_obj = self.random_sampling_pc(mesh)

        data = {
            "data_path": self.data_pathes[idx],
            "obj_id": "-1",
            "obj_verts": rs_verts_obj,
            "obj_urdf_path": os.path.join(self.data_dir, "102_obj.urdf"),
            "obj_trajectory": torch.tensor(
                np.stack(obj_pose[:: self.skip].cpu()), device=self.device, dtype=torch.float
            ),
            "scene_objs": [],
            "wrist_pos": wrist_pos,
            "wrist_rot": wrist_rot,
            "mano_joints": mano_joints,
        }

        self.process_data(data, idx, rs_verts_obj)
        opt_path = f"data/retargeting/grab_demo_long/mano2{str(self.dexhand)}/102_sv_dict_st_0_ed_108.pkl"

        self.load_retargeted_data(data, opt_path)

        return data


if __name__ == "__main__":
    raise RuntimeError(
        "Run via a standalone entry script with isaacgym imported first -- see "
        "scripts/taco/ for the established pattern (this is a pre-existing repo quirk, "
        "not something introduced here)."
    )
