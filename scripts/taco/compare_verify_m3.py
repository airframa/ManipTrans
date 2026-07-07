"""
M3 follow-up: tracking-error comparison across multiple TACO sequences plus a
GRAB baseline, all at iter=5000 (our validated choice), to check whether TACO's
~1-2cm mean error with occasional spikes is typical for this kinematic-
retargeting stage in general, or unusually bad.

Reuses the same metric as taco_verify_m3.py: mano2dexhand.py's fitting loop
dumps "opt_joints_pos" (actual simulated Inspire Hand body positions in the
fitting's internal table-frame); we transform each dataset's MANO fingertip
targets into that same fixed table frame and compare.

Must be run from the repo root: all data paths (data/taco/..., data/retargeting/...)
are resolved relative to CWD, not to this file's location.

Usage (from repo root): python scripts/taco/compare_verify_m3.py
"""

from isaacgym import gymapi  # noqa: F401 -- must be first import in the whole process

import pickle

import numpy as np
import torch

from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory
import maniptrans_envs.lib.envs.dexhands  # noqa: F401 -- triggers hand auto-registration

from main.dataset.taco_dataset_dexhand import TACORightData, TACOLeftData, SEQUENCES
from main.dataset.grab_dataset_dexhand import GrabDemoDexHand
from main.dataset.grab_dataset_dexhand_long import GrabDemoLongDexHand
from main.dataset.transform import aa_to_rotmat

TIP_NAMES = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]


def build_table_transf(device):
    # Reproduces mano2dexhand.py Mano2Dexhand.__init__ lines 166-175 exactly.
    # Deterministic (no sequence-specific values) -- same for every dataset.
    transf = torch.eye(4, dtype=torch.float32, device=device)
    transf[:3, :3] = aa_to_rotmat(torch.tensor([0.0, 0.0, -np.pi / 2], device=device)) @ aa_to_rotmat(
        torch.tensor([np.pi / 2, 0.0, 0.0], device=device)
    )
    table_pos_z = 0.4
    table_half_height = 0.015
    transf[:3, 3] = torch.tensor([0.0, 0.0, table_pos_z + table_half_height], device=device)
    return transf


def compute_metrics(data, dexhand, pkl_path, table_transf, device):
    with open(pkl_path, "rb") as f:
        opt = pickle.load(f)
    opt_joints_pos = torch.tensor(opt["opt_joints_pos"], device=device, dtype=torch.float32)  # (Tf, n_body, 3)

    Tf = opt_joints_pos.shape[0]
    assert Tf == data["wrist_pos"].shape[0], f"Tf mismatch {Tf} vs {data['wrist_pos'].shape[0]}"

    tip_body_idx = {name: dexhand.body_names.index([b for b in dexhand.body_names if b.endswith(name)][0]) for name in TIP_NAMES}

    tip_errors = {}
    for name in TIP_NAMES:
        mano_tip = data["mano_joints"][name]  # (Tf, 3), identity mujoco2gym frame
        target_tip = (table_transf[:3, :3] @ mano_tip.T).T + table_transf[:3, 3]
        inspire_tip = opt_joints_pos[:, tip_body_idx[name]]  # (Tf, 3), already in table frame
        err = (inspire_tip - target_tip).norm(dim=-1)  # (Tf,)
        tip_errors[name] = err.detach().cpu().numpy()

    all_err = np.stack(list(tip_errors.values()), axis=1)  # (Tf, 5)
    per_frame_mean = all_err.mean(axis=1)
    worst_frame = int(np.argmax(per_frame_mean))

    return dict(
        Tf=Tf,
        mean=float(all_err.mean()),
        max=float(all_err.max()),
        worst_frame=worst_frame,
        worst_frame_err=float(per_frame_mean[worst_frame]),
        per_tip_mean={name: float(v.mean()) for name, v in tip_errors.items()},
    )


def run():
    device = "cuda:0"
    identity = torch.eye(4, dtype=torch.float32, device=device)

    dexhand_rh = DexHandFactory.create_hand("inspire", "right")
    dexhand_lh = DexHandFactory.create_hand("inspire", "left")

    table_transf = build_table_transf(device)

    results = []

    # --- TACO: all 4 sequences, both hands ---
    fdata_rh = TACORightData(mujoco2gym_transf=identity, device=device, dexhand=dexhand_rh)
    fdata_lh = TACOLeftData(mujoco2gym_transf=identity, device=device, dexhand=dexhand_lh)

    for idx, seq in enumerate(SEQUENCES):
        tag = f"taco[{idx}] {seq['triplet']}/{seq['seq_name']}"
        for side, fdata, dexhand in [("right", fdata_rh, dexhand_rh), ("left", fdata_lh, dexhand_lh)]:
            data = fdata[f"t{idx}"]
            pkl_path = f"data/retargeting/taco/mano2{str(dexhand)}/{seq['seq_name']}@{side}.pkl"
            m = compute_metrics(data, dexhand, pkl_path, table_transf, device)
            m["dataset"] = tag
            m["side"] = side
            results.append(m)

    # --- GRAB baseline: right hand only ---
    fdata_g = GrabDemoDexHand(mujoco2gym_transf=identity, device=device, dexhand=dexhand_rh)
    data_g = fdata_g["g0"]
    pkl_path_g = f"data/retargeting/grab_demo/mano2{str(dexhand_rh)}/102_sv_dict.pkl"
    m = compute_metrics(data_g, dexhand_rh, pkl_path_g, table_transf, device)
    m["dataset"] = "grab[g0] 102_sv_dict"
    m["side"] = "right"
    results.append(m)

    # --- GRAB baseline #2: same underlying motion, full 108-frame length ---
    fdata_g2 = GrabDemoLongDexHand(mujoco2gym_transf=identity, device=device, dexhand=dexhand_rh)
    data_g2 = fdata_g2["h0"]
    pkl_path_g2 = f"data/retargeting/grab_demo_long/mano2{str(dexhand_rh)}/102_sv_dict_st_0_ed_108.pkl"
    m = compute_metrics(data_g2, dexhand_rh, pkl_path_g2, table_transf, device)
    m["dataset"] = "grab[h0] 102_sv_dict_st_0_ed_108"
    m["side"] = "right"
    results.append(m)

    # --- report ---
    print(f"\n{'dataset':45s} {'side':6s} {'Tf':>5s} {'mean(cm)':>9s} {'max(cm)':>8s} {'worst_frame':>11s} {'worst_err(cm)':>13s}")
    print("-" * 100)
    for r in results:
        print(
            f"{r['dataset']:45s} {r['side']:6s} {r['Tf']:5d} "
            f"{r['mean']*100:9.2f} {r['max']*100:8.2f} {r['worst_frame']:11d} {r['worst_frame_err']*100:13.2f}"
        )

    return results


if __name__ == "__main__":
    run()
