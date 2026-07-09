"""
TACO -> ManipTrans loader. Supports a small hardcoded registry of sequences
(see SEQUENCES below), each with independently-confirmed right=tool/left=target
hand assignment (wrist-to-object-trajectory distance heuristic; zero or
near-zero crossovers -- see CLAUDE.md progress log).

Registered with ManipDataFactory as "taco_rh"/"taco_lh". Index format for
ManipDataFactory.dataset_type() / mano2dexhand.py's --data_idx: "t0", "t1", ...
-- an index into SEQUENCES.
"""

import os
import pickle
from functools import lru_cache

import numpy as np
import torch
import trimesh
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from manotorch.manolayer import ManoLayer

from main.dataset.transform import aa_to_rotmat, rotmat_to_aa
from .base import ManipData
from .decorators import register_manipdata

TACO_ROOT = "data/taco"

# Verified in the M2 grounding session: same indices work for both hands
# (MANO_LEFT.pkl / MANO_RIGHT.pkl are exact x-mirror images at these vertices).
FINGERTIP_VERTEX_IDS = {
    "index": 353,
    "middle": 467,
    "pinky": 695,
    "ring": 576,
    "thumb": 766,
}

NATIVE_FPS = 30
TARGET_FPS = 120  # matches base.py's hardcoded dt = 1/(120/skip)

# Registry of validated sequences. index -> "t{index}" for ManipDataFactory /
# mano2dexhand.py's --data_idx. Each entry's right=tool/left=target assignment
# was confirmed via the wrist-to-object-trajectory distance heuristic before
# being added here (see CLAUDE.md M2 progress log for entry 0; M3 follow-up
# session for entries 1-3).
SEQUENCES = [
    # 0: original PoC. 292 frames, right-to-tool crossings=0/583 (post-resample).
    dict(triplet="(brush, brush, pan)", seq_name="20230919_026", tool_id="035", target_id="057"),
    # 1: hammer/helmet. 85 frames, right-to-tool crossings=0/85, left-to-target crossings=0/85.
    dict(triplet="(hit, hammer, helmet)", seq_name="20231002_063", tool_id="139", target_id="039"),
    # 2: spoon/bowl. 108 frames, right-to-tool crossings=0/108, left-to-target crossings=17/108 (some ambiguity).
    dict(triplet="(put in, spoon, bowl)", seq_name="20231104_179", tool_id="198", target_id="180"),
    # 3: knife/plate. 106 frames, right-to-tool crossings=0/106, left-to-target crossings=0/106.
    dict(triplet="(scrape off, knife, plate)", seq_name="20231020_232", tool_id="063", target_id="166"),
]


def _resample_translation(trans: np.ndarray, times_src: np.ndarray, times_dst: np.ndarray) -> np.ndarray:
    # trans: (T, 3), linear interpolation per component
    return np.stack([np.interp(times_dst, times_src, trans[:, i]) for i in range(3)], axis=-1).astype(np.float32)


def _resample_rotmat(rotmats: np.ndarray, times_src: np.ndarray, times_dst: np.ndarray) -> np.ndarray:
    # rotmats: (T, 3, 3), slerp
    slerp = Slerp(times_src, R.from_matrix(rotmats))
    return slerp(times_dst).as_matrix().astype(np.float32)


def _resample_axis_angle(aa: np.ndarray, times_src: np.ndarray, times_dst: np.ndarray) -> np.ndarray:
    # aa: (T, 3) global orientation -- go through rotation matrices so we slerp, not lerp
    rotmats = aa_to_rotmat(aa)
    resampled = _resample_rotmat(rotmats, times_src, times_dst)
    return rotmat_to_aa(resampled).astype(np.float32)


def _lerp_componentwise(x: np.ndarray, times_src: np.ndarray, times_dst: np.ndarray) -> np.ndarray:
    # x: (T, D) -- simplification for the 45-dim finger pose: linear interp per axis-angle
    # component rather than per-joint slerp. Adjacent-frame finger rotations are small,
    # so this is a reasonable approximation for a PoC loader; revisit if retargeting
    # error concentrates in fast finger motions.
    D = x.shape[1]
    return np.stack([np.interp(times_dst, times_src, x[:, d]) for d in range(D)], axis=-1).astype(np.float32)


# TACO-specific table-height correction (mirrors grab_dataset_dexhand.py's transf_offset
# pattern -- a translation baked into self.mujoco2gym_transf, applied in process_data()
# before Mano2Dexhand ever sees the data). Rotation is left as identity: the M3
# coordinate-frame investigation found TACO's raw axis/handedness convention already
# matches what Mano2Dexhand.fitting()'s table transform expects (same as OakInk-V2,
# no correction needed there either) -- only the absolute height was off.
#
# Mano2Dexhand's table transform maps raw Y -> final Z (verified empirically), so a
# translation along raw Y is what shifts height in the frame the optimizer sees, exactly
# how GRAB's own transf_offset only touches its Y-component ([0.0, 0.018, 0.0]).
#
# Derived from data (scripts/taco/check_coord_frames.py + M4 pre-flight session): over
# the full PoC sequence, the tool (brush) object's z-height (post table-transform, no
# correction) ranges min=0.2637 to max=0.3914, and the target (pan) ranges min=0.2963 to
# max=0.3906. Table surface sits at z=0.415 (table_pos_z=0.4 + table_half_height=0.015,
# both mano2dexhand.py and dexhandmanip_bih.py). Taking each object's resting-height
# (trajectory minimum, i.e. when it's presumably on/near the table) and averaging the
# two deltas needed to bring that minimum up to the table surface:
#   delta_tool   = 0.415 - 0.2637 = 0.1513
#   delta_target = 0.415 - 0.2963 = 0.1187
#   TACO_HEIGHT_DELTA = mean(0.1513, 0.1187) = 0.135
TACO_HEIGHT_DELTA = 0.135


class TACODataBase(ManipData):
    def __init__(
        self,
        *,
        side: str,
        data_dir: str = TACO_ROOT,
        split: str = "all",
        skip: int = 2,  # TACO resampled to 120Hz equivalent, skip=2 -> effective 60Hz, matches base.py's dt assumption
        device="cuda:0",
        mujoco2gym_transf=None,
        max_seq_len=int(1e10),
        dexhand=None,
        **kwargs,
    ):
        assert side in ("right", "left"), f"side must be 'right' or 'left', got {side}"
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
        self.side = side

        # center_idx=0 (NOT None): TACO's hand_trans is defined as the wrist's world
        # position, matching TACO's own hand_pose_loader.py convention (which recenters
        # the wrist to the origin before adding trans). center_idx=None left manotorch's
        # get_rotation_center() offset (~10cm, shape-dependent) uncompensated, displacing
        # every joint/vertex ~10cm and inflating tips_distance 5-10x -- root-caused during
        # M4 debugging (see CLAUDE.md). grab_dataset_dexhand.py's center_idx=None is not
        # a counterexample: that loader never runs the MANO forward pass for vertices (it
        # loads precomputed absolute rhand_verts and only uses th_J_regressor, which
        # center_idx doesn't affect), so its setting was never actually exercised.
        self.manolayer = ManoLayer(
            rot_mode="axisang",
            side=side,
            center_idx=0,
            mano_assets_root="data/mano_v1_2",
            use_pca=False,
            flat_hand_mean=True,
        ).to(device)

        # table-height correction -- see TACO_HEIGHT_DELTA derivation above.
        # Same pattern as grab_dataset_dexhand.py's transf_offset; shared identically
        # across both hands (TACORightData/TACOLeftData) since it's one physical scene.
        transf_offset = np.eye(4)
        transf_offset[:3, 3] = np.array([0.0, TACO_HEIGHT_DELTA, 0.0])
        self.transf_offset = torch.tensor(transf_offset, dtype=torch.float32, device=mujoco2gym_transf.device)
        self.mujoco2gym_transf = mujoco2gym_transf @ self.transf_offset

        self.data_pathes = list(range(len(SEQUENCES)))

    def __len__(self):
        return len(self.data_pathes)

    @lru_cache(maxsize=None)
    def __getitem__(self, idx):
        # index format: "t0", "t1", ... (ManipDataFactory.dataset_type() dispatch prefix)
        # or a plain int -- both index into SEQUENCES.
        if isinstance(idx, str) and idx.startswith("t"):
            idx = int(idx[1:])
        else:
            idx = int(idx)
        seq = SEQUENCES[idx]
        seq_triplet, seq_name, tool_id, target_id = seq["triplet"], seq["seq_name"], seq["tool_id"], seq["target_id"]
        obj_id = tool_id if self.side == "right" else target_id

        hand_dir = os.path.join(self.data_dir, "Hand_Poses", seq_triplet, seq_name)
        obj_dir = os.path.join(self.data_dir, "Object_Poses", seq_triplet, seq_name)

        hand_pkl = "right_hand.pkl" if self.side == "right" else "left_hand.pkl"
        shape_pkl = "right_hand_shape.pkl" if self.side == "right" else "left_hand_shape.pkl"

        with open(os.path.join(hand_dir, hand_pkl), "rb") as f:
            hand_data = pickle.load(f)
        with open(os.path.join(hand_dir, shape_pkl), "rb") as f:
            beta = pickle.load(f)["hand_shape"].numpy().astype(np.float32)  # (10,)

        frame_keys = sorted(hand_data.keys())
        T = len(frame_keys)
        hand_pose = np.stack([hand_data[k]["hand_pose"].numpy() for k in frame_keys]).astype(np.float32)  # (T, 48)
        hand_trans = np.stack([hand_data[k]["hand_trans"].numpy() for k in frame_keys]).astype(np.float32)  # (T, 3)

        obj_file = f"tool_{tool_id}.npy" if self.side == "right" else f"target_{target_id}.npy"
        obj_traj_src = np.load(os.path.join(obj_dir, obj_file)).astype(np.float32)  # (T, 4, 4)
        assert obj_traj_src.shape[0] == T, "hand/object frame count mismatch"

        # --- resample 30Hz -> 120Hz-equivalent grid ---
        times_src = np.arange(T) / NATIVE_FPS
        times_dst = np.arange(0, times_src[-1] + 1e-9, 1.0 / TARGET_FPS)

        global_orient = _resample_axis_angle(hand_pose[:, :3], times_src, times_dst)  # (T2, 3)
        finger_pose = _lerp_componentwise(hand_pose[:, 3:], times_src, times_dst)  # (T2, 45)
        hand_trans_rs = _resample_translation(hand_trans, times_src, times_dst)  # (T2, 3)
        obj_rotmat_rs = _resample_rotmat(obj_traj_src[:, :3, :3], times_src, times_dst)  # (T2, 3, 3)
        obj_trans_rs = _resample_translation(obj_traj_src[:, :3, 3], times_src, times_dst)  # (T2, 3)

        # --- apply skip (120Hz -> effective 60Hz, matching base.py's dt=1/(120/skip)) ---
        sl = slice(None, None, self.skip)
        global_orient = global_orient[sl]
        finger_pose = finger_pose[sl]
        hand_trans_rs = hand_trans_rs[sl]
        obj_rotmat_rs = obj_rotmat_rs[sl]
        obj_trans_rs = obj_trans_rs[sl]

        Tf = global_orient.shape[0]
        full_pose = np.concatenate([global_orient, finger_pose], axis=1)  # (Tf, 48)

        pose_t = torch.tensor(full_pose, dtype=torch.float32, device=self.device)
        beta_t = torch.tensor(beta, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(Tf, 1)
        trans_t = torch.tensor(hand_trans_rs, dtype=torch.float32, device=self.device)

        mano_out = self.manolayer(pose_t, beta_t)
        verts = mano_out.verts + trans_t[:, None, :]
        joints = mano_out.joints + trans_t[:, None, :]

        wrist_pos = joints[:, 0].detach()
        middle_pos = joints[:, 4].detach()
        wrist_pos = wrist_pos - (middle_pos - wrist_pos) * 0.25  # standard wrist-pullback hack, matches other loaders

        mano_joints = {
            "index_proximal": joints[:, 1].detach(),
            "index_intermediate": joints[:, 2].detach(),
            "index_distal": joints[:, 3].detach(),
            "index_tip": verts[:, FINGERTIP_VERTEX_IDS["index"]].detach(),
            "middle_proximal": joints[:, 4].detach(),
            "middle_intermediate": joints[:, 5].detach(),
            "middle_distal": joints[:, 6].detach(),
            "middle_tip": verts[:, FINGERTIP_VERTEX_IDS["middle"]].detach(),
            "pinky_proximal": joints[:, 7].detach(),
            "pinky_intermediate": joints[:, 8].detach(),
            "pinky_distal": joints[:, 9].detach(),
            "pinky_tip": verts[:, FINGERTIP_VERTEX_IDS["pinky"]].detach(),
            "ring_proximal": joints[:, 10].detach(),
            "ring_intermediate": joints[:, 11].detach(),
            "ring_distal": joints[:, 12].detach(),
            "ring_tip": verts[:, FINGERTIP_VERTEX_IDS["ring"]].detach(),
            "thumb_proximal": joints[:, 13].detach(),
            "thumb_intermediate": joints[:, 14].detach(),
            "thumb_distal": joints[:, 15].detach(),
            "thumb_tip": verts[:, FINGERTIP_VERTEX_IDS["thumb"]].detach(),
        }

        global_orient_rotmat = aa_to_rotmat(torch.tensor(global_orient, dtype=torch.float32, device=self.device))
        inspire_rot_offset = self.dexhand.relative_rotation
        wrist_rot = global_orient_rotmat @ torch.tensor(
            np.repeat(inspire_rot_offset[None], Tf, axis=0), dtype=torch.float32, device=self.device
        )

        obj_trajectory = torch.eye(4, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(Tf, 1, 1)
        obj_trajectory[:, :3, :3] = torch.tensor(obj_rotmat_rs, dtype=torch.float32, device=self.device)
        obj_trajectory[:, :3, 3] = torch.tensor(obj_trans_rs, dtype=torch.float32, device=self.device)

        obj_mesh_path = os.path.join(TACO_ROOT, "object_models_released", f"{obj_id}_cm.obj")
        obj_mesh = trimesh.load(obj_mesh_path, process=False, force="mesh", skip_materials=True)
        obj_mesh_verts_m = (obj_mesh.vertices * 0.01).astype(np.float32)  # cm -> m

        mesh = Meshes(
            verts=torch.from_numpy(obj_mesh_verts_m[None, ...]),
            faces=torch.from_numpy(obj_mesh.faces[None, ...].astype(np.float32)),
        )
        rs_verts_obj = self.random_sampling_pc(mesh)

        data = {
            "data_path": f"{seq_triplet}/{seq_name}@{self.side}",
            "obj_id": obj_id,
            "obj_verts": rs_verts_obj,
            "obj_urdf_path": os.path.join(TACO_ROOT, "urdfs", obj_id, f"{obj_id}.urdf"),
            "obj_trajectory": obj_trajectory,
            "scene_objs": [],
            "wrist_pos": wrist_pos,
            "wrist_rot": wrist_rot,
            "mano_joints": mano_joints,
        }

        self.process_data(data, idx, rs_verts_obj)

        opt_path = f"data/retargeting/taco/mano2{str(self.dexhand)}/{seq_name}@{self.side}.pkl"
        self.load_retargeted_data(data, opt_path)

        return data


@register_manipdata("taco_rh")
class TACORightData(TACODataBase):
    def __init__(self, **kwargs):
        kwargs.pop("side", None)
        super().__init__(side="right", **kwargs)


@register_manipdata("taco_lh")
class TACOLeftData(TACODataBase):
    def __init__(self, **kwargs):
        kwargs.pop("side", None)
        super().__init__(side="left", **kwargs)


def run_verification():
    # Deferred imports: constructing a real DexHand pulls in maniptrans_envs.lib.envs,
    # which imports isaacgym -- and isaacgym must be imported before torch anywhere in
    # the process. Callers must import isaacgym before importing/calling this function
    # (see taco_verify_m2.py at the repo root).
    from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory
    import maniptrans_envs.lib.envs.dexhands  # noqa: F401 -- triggers hand auto-registration

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = "cuda:0"
    mujoco2gym_transf = torch.eye(4, dtype=torch.float32, device=device)

    dexhand_rh = DexHandFactory.create_hand("inspire", "right")
    dexhand_lh = DexHandFactory.create_hand("inspire", "left")

    fdata_rh = TACORightData(mujoco2gym_transf=mujoco2gym_transf, device=device, dexhand=dexhand_rh)
    fdata_lh = TACOLeftData(mujoco2gym_transf=mujoco2gym_transf, device=device, dexhand=dexhand_lh)

    data_rh = fdata_rh[0]
    data_lh = fdata_lh[0]

    print("=" * 20, "RIGHT (tool=brush 035)", "=" * 20)
    for k, v in data_rh.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:28s} shape={tuple(v.shape)!s:20s} dtype={v.dtype} device={v.device}")
        elif isinstance(v, dict):
            print(f"  {k:28s} dict with {len(v)} keys")
            for kk, vv in v.items():
                print(f"      {kk:24s} shape={tuple(vv.shape)!s:20s} dtype={vv.dtype} device={vv.device}")
        else:
            print(f"  {k:28s} {type(v)} = {v}")

    Tf = data_rh["obj_trajectory"].shape[0]

    # --- shape assertions vs. the ManipData interface documented in CLAUDE.md ---
    assert data_rh["obj_verts"].shape == (1000, 3)
    assert data_rh["obj_trajectory"].shape == (Tf, 4, 4)
    assert data_rh["wrist_pos"].shape == (Tf, 3)
    assert data_rh["wrist_rot"].shape == (Tf, 3, 3) or data_rh["wrist_rot"].shape == (Tf, 3)  # rotmat pre-conversion
    assert len(data_rh["mano_joints"]) == 20
    for v in data_rh["mano_joints"].values():
        assert v.shape == (Tf, 3)
    assert isinstance(data_rh["obj_id"], str)
    assert isinstance(data_rh["obj_urdf_path"], str) and os.path.exists(data_rh["obj_urdf_path"])
    print(f"\nAll shape assertions passed. Tf={Tf} frames at effective 60Hz.")

    # --- sanity-check PNG: fingertips + object point cloud at a few timesteps ---
    tip_keys = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]
    n_show = 4
    t_idxs = np.linspace(0, Tf - 1, n_show).astype(int)

    fig = plt.figure(figsize=(20, 5))
    for i, t in enumerate(t_idxs):
        ax = fig.add_subplot(1, n_show, i + 1, projection="3d")

        obj_verts_t = (
            data_rh["obj_trajectory"][t, :3, :3] @ data_rh["obj_verts"].T
        ).T + data_rh["obj_trajectory"][t, :3, 3]
        obj_verts_t = obj_verts_t.detach().cpu().numpy()
        ax.scatter(obj_verts_t[:, 0], obj_verts_t[:, 1], obj_verts_t[:, 2], s=1, alpha=0.15, color="tab:orange", label="tool (brush)")

        target_verts_t = (
            data_lh["obj_trajectory"][t, :3, :3] @ data_lh["obj_verts"].T
        ).T + data_lh["obj_trajectory"][t, :3, 3]
        target_verts_t = target_verts_t.detach().cpu().numpy()
        ax.scatter(target_verts_t[:, 0], target_verts_t[:, 1], target_verts_t[:, 2], s=1, alpha=0.15, color="tab:blue", label="target (pan)")

        rh_tips = np.stack([data_rh["mano_joints"][k][t].detach().cpu().numpy() for k in tip_keys])
        lh_tips = np.stack([data_lh["mano_joints"][k][t].detach().cpu().numpy() for k in tip_keys])
        rh_wrist = data_rh["wrist_pos"][t].detach().cpu().numpy()
        lh_wrist = data_lh["wrist_pos"][t].detach().cpu().numpy()

        ax.scatter(rh_tips[:, 0], rh_tips[:, 1], rh_tips[:, 2], s=40, color="red", marker="^", label="RH fingertips")
        ax.scatter(*rh_wrist, s=60, color="darkred", marker="x", label="RH wrist")
        ax.scatter(lh_tips[:, 0], lh_tips[:, 1], lh_tips[:, 2], s=40, color="green", marker="^", label="LH fingertips")
        ax.scatter(*lh_wrist, s=60, color="darkgreen", marker="x", label="LH wrist")

        ax.set_title(f"frame {t}/{Tf}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        if i == 0:
            ax.legend(fontsize=6, loc="upper left")

    out_png = "data/taco/m2_sanity_check.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Saved sanity-check PNG to {out_png}")

    return data_rh, data_lh


def run_distance_check():
    from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory
    import maniptrans_envs.lib.envs.dexhands  # noqa: F401

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = "cuda:0"
    mujoco2gym_transf = torch.eye(4, dtype=torch.float32, device=device)

    dexhand_rh = DexHandFactory.create_hand("inspire", "right")
    dexhand_lh = DexHandFactory.create_hand("inspire", "left")

    fdata_rh = TACORightData(mujoco2gym_transf=mujoco2gym_transf, device=device, dexhand=dexhand_rh)
    fdata_lh = TACOLeftData(mujoco2gym_transf=mujoco2gym_transf, device=device, dexhand=dexhand_lh)

    data_rh = fdata_rh[0]
    data_lh = fdata_lh[0]

    rh_wrist = data_rh["wrist_pos"]  # (Tf, 3) -- right hand
    lh_wrist = data_lh["wrist_pos"]  # (Tf, 3) -- left hand
    tool_pos = data_rh["obj_trajectory"][:, :3, 3]  # (Tf, 3) -- brush (035)
    target_pos = data_lh["obj_trajectory"][:, :3, 3]  # (Tf, 3) -- pan (057)

    Tf = rh_wrist.shape[0]
    assert lh_wrist.shape[0] == Tf and tool_pos.shape[0] == Tf and target_pos.shape[0] == Tf

    right_to_tool = (rh_wrist - tool_pos).norm(dim=-1).detach().cpu().numpy()
    right_to_target = (rh_wrist - target_pos).norm(dim=-1).detach().cpu().numpy()
    left_to_tool = (lh_wrist - tool_pos).norm(dim=-1).detach().cpu().numpy()
    left_to_target = (lh_wrist - target_pos).norm(dim=-1).detach().cpu().numpy()

    frames = np.arange(Tf)

    plt.figure(figsize=(12, 5))
    plt.plot(frames, right_to_tool, label="right_wrist - tool (brush)", color="red", linestyle="-")
    plt.plot(frames, right_to_target, label="right_wrist - target (pan)", color="red", linestyle="--")
    plt.plot(frames, left_to_tool, label="left_wrist - tool (brush)", color="green", linestyle="--")
    plt.plot(frames, left_to_target, label="left_wrist - target (pan)", color="green", linestyle="-")
    plt.xlabel("frame index")
    plt.ylabel("distance (m)")
    plt.title("Wrist-to-object distance over the sequence (effective 60Hz)")
    plt.legend()
    plt.grid(alpha=0.3)

    out_png = "data/taco/m2_distance_check.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Saved distance-check PNG to {out_png}")

    n_cross_right = int(np.sum(right_to_tool > right_to_target))
    n_cross_left = int(np.sum(left_to_target > left_to_tool))
    print(f"Frames where right_to_tool > right_to_target: {n_cross_right}/{Tf}")
    print(f"Frames where left_to_target > left_to_tool:   {n_cross_left}/{Tf}")
    print(f"right_to_tool:   mean={right_to_tool.mean():.4f}  max={right_to_tool.max():.4f}")
    print(f"right_to_target: mean={right_to_target.mean():.4f}  min={right_to_target.min():.4f}")
    print(f"left_to_target:  mean={left_to_target.mean():.4f}  max={left_to_target.max():.4f}")
    print(f"left_to_tool:    mean={left_to_tool.mean():.4f}  min={left_to_tool.min():.4f}")


if __name__ == "__main__":
    raise RuntimeError(
        "Run via `python taco_verify_m2.py` from the repo root instead -- isaacgym must be "
        "imported before torch anywhere in the process, and importing this module directly "
        "triggers main/dataset/__init__.py's auto-registration of every loader in this "
        "directory (including mano2dexhand.py) before we get a chance to import isaacgym first."
    )
