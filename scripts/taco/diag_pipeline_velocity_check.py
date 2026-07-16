"""
Follow-up to diag_raw_velocity_check.py: raw 30Hz data tops out at 4.88-6.73
rad/s (t3/t2 tool), but the live pipeline (TACODataBase -> process_data)
was observed producing 25-31 rad/s at the same sequence. Runs the ACTUAL
resample + skip + compute_angular_velocity code path (imported directly
from taco_dataset_dexhand.py / base.py, no full Isaac Gym env needed) to
find exactly where the amplification is introduced.

Must be run from the repo root: python scripts/taco/diag_pipeline_velocity_check.py
"""

from isaacgym import gymapi  # noqa: F401 -- must be first import; main.dataset auto-imports mano2dexhand.py which needs isaacgym imported first

import os

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from main.dataset.taco_dataset_dexhand import (
    TACO_ROOT,
    NATIVE_FPS,
    TARGET_FPS,
    SEQUENCES,
    _resample_rotmat,
)
from main.dataset.base import ManipData

SEQ_IDX = 3  # t3 knife/plate
SKIP = 2

seq = SEQUENCES[SEQ_IDX]
obj_file = f"tool_{seq['tool_id']}.npy"
path = os.path.join(TACO_ROOT, "Object_Poses", seq["triplet"], seq["seq_name"], obj_file)
obj_traj_src = np.load(path).astype(np.float32)
T = obj_traj_src.shape[0]
print(f"=== {seq['seq_name']} tool, raw T={T} ===")

times_src = np.arange(T) / NATIVE_FPS
times_dst = np.arange(0, times_src[-1] + 1e-9, 1.0 / TARGET_FPS)
print(f"times_src (raw, 30Hz): {len(times_src)} pts, spacing={times_src[1]-times_src[0]:.5f}s")
print(f"times_dst (resampled, 120Hz): {len(times_dst)} pts, spacing={times_dst[1]-times_dst[0]:.5f}s")

obj_rotmat_rs = _resample_rotmat(obj_traj_src[:, :3, :3], times_src, times_dst)  # (T2,3,3) @ 120Hz
print(f"obj_rotmat_rs shape (post-slerp, pre-skip, 120Hz): {obj_rotmat_rs.shape}")

sl = slice(None, None, SKIP)
obj_rotmat_final = obj_rotmat_rs[sl]  # (Tf, 3, 3) @ effective 60Hz
print(f"obj_rotmat_final shape (post-skip, effective 60Hz): {obj_rotmat_final.shape}")

# (A) naive finite difference directly on the 120Hz PRE-skip resampled array, dt=1/120
diff_r_120 = obj_rotmat_rs[1:] @ np.transpose(obj_rotmat_rs[:-1], (0, 2, 1))
w_120 = np.linalg.norm(R.from_matrix(diff_r_120).as_rotvec(), axis=-1) / (1.0 / TARGET_FPS)
print(f"\n(A) angular velocity on 120Hz resampled array (dt=1/120, pre-skip): max={w_120.max():.3f} rad/s at frame {int(np.argmax(w_120))}")

# (B) naive finite difference on the post-skip 60Hz-effective array, dt=1/(120/skip)=1/60 --
#     this exactly matches base.py's compute_angular_velocity dt argument.
diff_r_60 = obj_rotmat_final[1:] @ np.transpose(obj_rotmat_final[:-1], (0, 2, 1))
w_60 = np.linalg.norm(R.from_matrix(diff_r_60).as_rotvec(), axis=-1) / (1.0 / (TARGET_FPS / SKIP))
print(f"(B) angular velocity on post-skip 60Hz-effective array (dt=1/60, matches base.py): max={w_60.max():.3f} rad/s at frame {int(np.argmax(w_60))}")

# (C) run the ACTUAL base.py compute_angular_velocity (numpy gradient + gaussian filter) for
#     a byte-for-byte match to what TACODataBase really produces.
device = "cuda:0" if torch.cuda.is_available() else "cpu"
rotmat_t = torch.tensor(obj_rotmat_final[:, None, :, :], dtype=torch.float32, device=device)
angvel_real = ManipData.compute_angular_velocity(rotmat_t, 1.0 / (TARGET_FPS / SKIP), guassian_filter=True).squeeze(1)
angvel_real_mag = angvel_real.norm(dim=-1).cpu().numpy()
print(f"(C) ACTUAL base.py compute_angular_velocity (with gaussian_filter1d): max={angvel_real_mag.max():.3f} rad/s at frame {int(np.argmax(angvel_real_mag))}")

# also without the gaussian filter, to isolate its effect
angvel_nofilter = ManipData.compute_angular_velocity(rotmat_t, 1.0 / (TARGET_FPS / SKIP), guassian_filter=False).squeeze(1)
angvel_nofilter_mag = angvel_nofilter.norm(dim=-1).cpu().numpy()
print(f"(C') same, guassian_filter=False: max={angvel_nofilter_mag.max():.3f} rad/s at frame {int(np.argmax(angvel_nofilter_mag))}")

print(f"\nFor reference, RAW 30Hz naive finite-difference max was 4.88 rad/s (from diag_raw_velocity_check.py).")
print(f"Frame index mapping: post-skip frame f corresponds to raw frame ~= f * {SKIP} / ({TARGET_FPS}/{NATIVE_FPS}) = f/2")

top5 = np.argsort(angvel_real_mag)[-8:][::-1]
print(f"\ntop-8 frames by ACTUAL pipeline angular velocity: {[(int(i), round(float(angvel_real_mag[i]),2)) for i in top5]}")
print(f"corresponding raw-frame estimate: {[round(int(i)/2,1) for i in top5]}")
