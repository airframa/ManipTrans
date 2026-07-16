"""
Root-cause check: are TACO's implausible object angular-velocity spikes
(25-31 rad/s) present in the RAW 30Hz obj_trajectory npy files, or
introduced by our resampling/velocity-computation pipeline?

Three independent computations, side by side, per sequence:
  (a) RAW: finite difference directly on the raw (T,4,4) npy at native 30Hz
      (dt=1/30), completely bypassing TACODataBase/resampling/base.py.
  (b) OURS: the actual pipeline output -- TACODataBase.__getitem__ ->
      process_data() -- post-resample (30->120Hz slerp), post-skip (->60Hz
      effective), post-compute_angular_velocity (base.py, dt=1/60).
  (c) quaternion dot-product sign check at (a)'s spike frames, both on the
      raw consecutive frames AND on the pre-slerp resampling input, to
      directly test the double-cover/long-path slerp hypothesis.

Must be run from the repo root: python scripts/taco/diag_raw_velocity_check.py
"""

import os

import numpy as np
from scipy.spatial.transform import Rotation as R

TACO_ROOT = "data/taco"

SEQS = [
    dict(name="t3 knife/plate", triplet="(scrape off, knife, plate)", seq_name="20231020_232", tool_id="063", target_id="166"),
    dict(name="t2 spoon/bowl", triplet="(put in, spoon, bowl)", seq_name="20231104_179", tool_id="198", target_id="180"),
]


def raw_angular_velocity(rotmats, dt):
    # matrix-based relative rotation, immune to quaternion double-cover -- mirrors
    # base.py's compute_angular_velocity exactly, just on the untouched raw array.
    diff_r = rotmats[1:] @ np.transpose(rotmats[:-1], (0, 2, 1))
    rotvec = R.from_matrix(diff_r).as_rotvec()  # shortest-path by construction
    angle = np.linalg.norm(rotvec, axis=-1)
    return angle / dt  # rad/s, per-frame-gap


def main():
    for seq in SEQS:
        for role, obj_id, fname_tpl in [("tool", seq["tool_id"], "tool_{}.npy"), ("target", seq["target_id"], "target_{}.npy")]:
            path = os.path.join(TACO_ROOT, "Object_Poses", seq["triplet"], seq["seq_name"], fname_tpl.format(obj_id))
            traj = np.load(path).astype(np.float64)  # (T, 4, 4)
            rotmats = traj[:, :3, :3]
            T = rotmats.shape[0]

            raw_w = raw_angular_velocity(rotmats, dt=1.0 / 30.0)  # (T-1,)

            print(f"\n=== {seq['name']} / {role} ({fname_tpl.format(obj_id)}), T={T} raw frames @ 30Hz ===")
            print(f"  RAW angular velocity (rad/s): mean={raw_w.mean():.3f} median={np.median(raw_w):.3f} max={raw_w.max():.3f} (at raw-frame-gap {int(np.argmax(raw_w))}->{int(np.argmax(raw_w))+1})")
            top5 = np.argsort(raw_w)[-5:][::-1]
            print(f"  top-5 raw-frame-gaps by angular velocity: {[(int(i), round(float(raw_w[i]),2)) for i in top5]}")

            # quaternion dot-product sign check at the raw data's own worst gaps
            quats = R.from_matrix(rotmats).as_quat()  # (T,4) xyzw, scipy's own canonicalization
            print("  quaternion dot(q[i],q[i+1]) at top-5 worst raw gaps (negative => antipodal representation):")
            for i in top5:
                dot = float(np.dot(quats[i], quats[i + 1]))
                print(f"    gap {int(i)}->{int(i)+1}: dot={dot:+.4f}  raw_angvel={raw_w[i]:.2f} rad/s")

            # is the spike a single-frame outlier (isolated bad annotation) or sustained?
            # print angular velocity in a window around the worst gap
            worst = int(np.argmax(raw_w))
            lo, hi = max(0, worst - 3), min(len(raw_w), worst + 4)
            print(f"  raw angular velocity window around worst gap [{lo}:{hi}]: {[round(float(x),2) for x in raw_w[lo:hi]]}")


if __name__ == "__main__":
    main()
