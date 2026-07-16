"""
Definitive check: instantiate the REAL TACORightData/TACOLeftData classes
(as the actual training env does, including the real mujoco2gym_transf --
NOT torch.eye(4) like the M2/M3 verification scripts used) and dump
obj_angular_velocity directly, to reconcile a discrepancy: the earlier
live-env reset-state diagnostic showed |ang_vel|=25.04 rad/s at seq_idx=100
for the RH-side (tool/knife) object of t3, but a standalone reproduction of
the resample+skip+compute_angular_velocity pipeline (with mujoco2gym_transf
= identity) found the sequence's max angular velocity is only ~4.88 rad/s
anywhere in the whole sequence, nowhere near frame 100.

Must be run from the repo root: python scripts/taco/diag_real_dataset_velocity.py
"""

from isaacgym import gymapi  # noqa: F401 -- must be first import in the whole process

import numpy as np
import torch

from main.dataset.taco_dataset_dexhand import TACORightData, SEQUENCES
from main.dataset.transform import aa_to_rotmat

device = "cuda:0"

# Reproduce the REAL env's mujoco2gym_transf exactly (dexhandmanip_bih.py:257-262)
mujoco2gym_transf = np.eye(4)
mujoco2gym_transf[:3, :3] = aa_to_rotmat(np.array([0, 0, -np.pi / 2])) @ aa_to_rotmat(np.array([np.pi / 2, 0, 0]))
mujoco2gym_transf[:3, 3] = np.array([0, 0, 0.415])  # table_surface_z, matches env
mujoco2gym_transf_t = torch.tensor(mujoco2gym_transf, dtype=torch.float32, device=device)

from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory
import maniptrans_envs.lib.envs.dexhands  # noqa: F401 -- triggers hand auto-registration

dexhand_rh = DexHandFactory.create_hand("inspire", "right")

fdata_rh = TACORightData(mujoco2gym_transf=mujoco2gym_transf_t, device=device, dexhand=dexhand_rh)
data_rh = fdata_rh["t3"]

ang_vel = data_rh["obj_angular_velocity"]  # (Tf, 3)
mag = ang_vel.norm(dim=-1).detach().cpu().numpy()

print(f"Tf = {mag.shape[0]}")
print(f"mag[95:106] (around frame 100): {[round(float(x), 3) for x in mag[95:106]]}")
print(f"global max = {mag.max():.3f} rad/s at frame {int(np.argmax(mag))}")
print(f"mag at exactly frame 100: {mag[100]:.4f} rad/s")

lin_vel = data_rh["obj_velocity"]
lin_mag = lin_vel.norm(dim=-1).detach().cpu().numpy()
print(f"\nlinear vel mag[95:106]: {[round(float(x), 4) for x in lin_mag[95:106]]}")
print(f"linear vel at frame 100: {lin_mag[100]:.4f} m/s")

print(f"\nobj_trajectory dtype/shape: {data_rh['obj_trajectory'].shape}")
print("obj_trajectory rotation at frames 99,100,101:")
for f in [99, 100, 101]:
    print(data_rh["obj_trajectory"][f, :3, :3].detach().cpu().numpy())
