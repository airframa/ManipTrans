"""
M3 verification: kinematic retargeting tracking error.

mano2dexhand.py's fitting loop already dumps "opt_joints_pos" -- the actual
simulated Inspire Hand body positions (in the fitting's internal table-frame
coordinate system) at the point the optimization stopped. We reuse that
directly rather than re-running any simulation or forward-kinematics pass:
just transform our MANO fingertip targets into the same table frame (a fixed,
deterministic transform -- reproduced here from mano2dexhand.py's
Mano2Dexhand.__init__, not re-derived per sequence) and compare.

Must be run from the repo root: all data paths (data/taco/..., data/retargeting/...)
are resolved relative to CWD, not to this file's location.

Usage (from repo root): python scripts/taco/taco_verify_m3.py [seq_idx] [tag]
  seq_idx: index into taco_dataset_dexhand.SEQUENCES (default 0)
  tag: short name used in output filenames (default derived from seq_idx)
"""

from isaacgym import gymapi  # noqa: F401 -- must be first import in the whole process

import pickle
import sys

import numpy as np
import torch

from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory
import maniptrans_envs.lib.envs.dexhands  # noqa: F401 -- triggers hand auto-registration

from main.dataset.taco_dataset_dexhand import TACORightData, TACOLeftData, SEQUENCES
from main.dataset.transform import aa_to_rotmat

TIP_NAMES = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]

SEQ_IDX = int(sys.argv[1]) if len(sys.argv) > 1 else 0
SEQ_NAME = SEQUENCES[SEQ_IDX]["seq_name"]
TAG = sys.argv[2] if len(sys.argv) > 2 else f"t{SEQ_IDX}"


def build_table_transf(device):
    # Reproduces mano2dexhand.py Mano2Dexhand.__init__ lines 166-175 exactly.
    # Deterministic (no sequence-specific values) -- safe to recompute here
    # rather than spin up a full Isaac Gym sim just to read it back.
    transf = torch.eye(4, dtype=torch.float32, device=device)
    transf[:3, :3] = aa_to_rotmat(torch.tensor([0.0, 0.0, -np.pi / 2], device=device)) @ aa_to_rotmat(
        torch.tensor([np.pi / 2, 0.0, 0.0], device=device)
    )
    table_pos_z = 0.4
    table_half_height = 0.015
    transf[:3, 3] = torch.tensor([0.0, 0.0, table_pos_z + table_half_height], device=device)
    return transf


def run():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = "cuda:0"
    mujoco2gym_transf_identity = torch.eye(4, dtype=torch.float32, device=device)

    dexhand_rh = DexHandFactory.create_hand("inspire", "right")
    dexhand_lh = DexHandFactory.create_hand("inspire", "left")

    fdata_rh = TACORightData(mujoco2gym_transf=mujoco2gym_transf_identity, device=device, dexhand=dexhand_rh)
    fdata_lh = TACOLeftData(mujoco2gym_transf=mujoco2gym_transf_identity, device=device, dexhand=dexhand_lh)

    data_rh = fdata_rh[f"t{SEQ_IDX}"]
    data_lh = fdata_lh[f"t{SEQ_IDX}"]

    table_transf = build_table_transf(device)

    results = {}
    for side, data, dexhand in [("right", data_rh, dexhand_rh), ("left", data_lh, dexhand_lh)]:
        pkl_path = f"data/retargeting/taco/mano2{str(dexhand)}/{SEQ_NAME}@{side}.pkl"
        with open(pkl_path, "rb") as f:
            opt = pickle.load(f)
        opt_joints_pos = torch.tensor(opt["opt_joints_pos"], device=device, dtype=torch.float32)  # (Tf, n_body, 3)

        Tf = opt_joints_pos.shape[0]
        assert Tf == data["wrist_pos"].shape[0], f"{side}: Tf mismatch {Tf} vs {data['wrist_pos'].shape[0]}"

        # dexhand.body_names already includes the side prefix (e.g. "R_thumb_tip")
        tip_body_idx = {name: dexhand.body_names.index([b for b in dexhand.body_names if b.endswith(name)][0]) for name in TIP_NAMES}

        target_tips = {}
        inspire_tips = {}
        tip_errors = {}
        for name in TIP_NAMES:
            mano_tip = data["mano_joints"][name]  # (Tf, 3), identity mujoco2gym frame
            target_tip = (table_transf[:3, :3] @ mano_tip.T).T + table_transf[:3, 3]
            inspire_tip = opt_joints_pos[:, tip_body_idx[name]]  # (Tf, 3), already in table frame
            err = (inspire_tip - target_tip).norm(dim=-1)  # (Tf,)

            target_tips[name] = target_tip
            inspire_tips[name] = inspire_tip
            tip_errors[name] = err.detach().cpu().numpy()

        all_err = np.stack(list(tip_errors.values()), axis=1)  # (Tf, 5)
        mean_overall = all_err.mean()
        max_overall = all_err.max()
        worst_frame = int(np.unravel_index(np.argmax(all_err), all_err.shape)[0])
        per_frame_mean = all_err.mean(axis=1)  # (Tf,)
        worst5 = np.argsort(per_frame_mean)[-5:][::-1]

        print(f"=== {side} (dexhand={dexhand}) ===")
        for name in TIP_NAMES:
            print(f"  {name:12s} mean={tip_errors[name].mean()*100:6.2f}cm  max={tip_errors[name].max()*100:6.2f}cm")
        print(f"  OVERALL: mean={mean_overall*100:.2f}cm  max={max_overall*100:.2f}cm  worst_single_frame={worst_frame}")
        print(f"  Worst 5 frames by mean-across-fingers error: {worst5.tolist()} -> {[round(per_frame_mean[i]*100,2) for i in worst5]} cm")

        results[side] = dict(
            tip_errors=tip_errors,
            target_tips=target_tips,
            inspire_tips=inspire_tips,
            per_frame_mean=per_frame_mean,
            Tf=Tf,
        )

    # --- error-over-time line plot ---
    plt.figure(figsize=(12, 5))
    for side, color in [("right", "red"), ("left", "green")]:
        plt.plot(results[side]["per_frame_mean"] * 100, label=f"{side} mean tip error (cm)", color=color)
    plt.axhline(1.0, color="black", linestyle=":", label="1cm target")
    plt.xlabel("frame index")
    plt.ylabel("error (cm)")
    plt.title(f"M3 [{TAG}]: kinematic retargeting fingertip tracking error (mean across 5 tips)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_err_path = f"outputs/m3_{TAG}_tracking_error.png"
    plt.savefig(out_err_path, dpi=150)
    print(f"Saved {out_err_path}")

    # --- overlay scatter: MANO fingertips (targets) vs retargeted Inspire fingertips ---
    n_show = 4
    Tf_rh = results["right"]["Tf"]
    t_idxs = np.linspace(0, Tf_rh - 1, n_show).astype(int)

    fig = plt.figure(figsize=(20, 10))
    for row, side in enumerate(["right", "left"]):
        for i, t in enumerate(t_idxs):
            ax = fig.add_subplot(2, n_show, row * n_show + i + 1, projection="3d")
            target_pts = np.stack([results[side]["target_tips"][name][t].detach().cpu().numpy() for name in TIP_NAMES])
            inspire_pts = np.stack([results[side]["inspire_tips"][name][t].detach().cpu().numpy() for name in TIP_NAMES])

            ax.scatter(target_pts[:, 0], target_pts[:, 1], target_pts[:, 2], s=60, color="blue", marker="o", label="MANO target")
            ax.scatter(inspire_pts[:, 0], inspire_pts[:, 1], inspire_pts[:, 2], s=60, color="orange", marker="^", label="Inspire retargeted")
            for j in range(len(TIP_NAMES)):
                ax.plot(
                    [target_pts[j, 0], inspire_pts[j, 0]],
                    [target_pts[j, 1], inspire_pts[j, 1]],
                    [target_pts[j, 2], inspire_pts[j, 2]],
                    color="gray",
                    linewidth=0.7,
                )
            err_cm = results[side]["tip_errors"][TIP_NAMES[0]][t] * 100  # just for title context
            ax.set_title(f"{side} frame {t}/{results[side]['Tf']}\nmean err={results[side]['per_frame_mean'][t]*100:.2f}cm")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            if row == 0 and i == 0:
                ax.legend(fontsize=6, loc="upper left")

    plt.tight_layout()
    out_overlay_path = f"outputs/m3_{TAG}_overlay.png"
    plt.savefig(out_overlay_path, dpi=150)
    print(f"Saved {out_overlay_path}")

    return results


if __name__ == "__main__":
    run()
