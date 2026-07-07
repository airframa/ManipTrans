# CLAUDE.md — TACO → Inspire Hand via ManipTrans

## Project

Retarget the **TACO** bimanual tool-use dataset onto the **Inspire Hand** dexterous embodiment using the **ManipTrans** pipeline. The output is a DexManipNet-style dataset: for each TACO sequence, a physically feasible Inspire Hand trajectory (joint angles + wrist 6-DoF over time) that reproduces the human bimanual tool-use behavior in Isaac Gym.

This is a data-generation task — we are *not* modifying ManipTrans's algorithm. We are writing a TACO-specific `ManipData` adapter and generating URDFs / preprocessed trajectories so that ManipTrans's existing training pipeline can consume TACO.

## Repositories and paths

Working setup: we work directly inside the ManipTrans repo on the `taco-retargeting` branch. Generated data (URDFs, trajectories, checkpoints) goes under `data/` or `outputs/` and must be `.gitignore`d.

- **ManipTrans repo (also working dir):** `/home/fmb/projects/ManipTrans`
- **Branch:** `taco-retargeting`
- **TACO data root:** `/home/fmb/projects/ManipTrans/data/taco`
  - `data/taco/Hand_Poses/` — extracted from `Hand_Poses.zip`
  - `data/taco/Object_Poses/` — extracted from `Object_Poses.zip`
  - `data/taco/object_models_released/` — extracted from `Object_Models.zip` (note: not `Object_Models/`)
- **URDFs output:** `/home/fmb/projects/ManipTrans/data/taco/urdfs/`
- **Retargeted trajectories output:** `/home/fmb/projects/ManipTrans/outputs/inspire_trajectories/`
- **Conda env:** `maniptrans` (already set up and verified working)

## Key interfaces

### What ManipTrans consumes (per-sample dict from `ManipData`)

**Verified from actual source code** (`main/dataset/base.py`), not documentation.

Fields the subclass supplies directly (all `torch.Tensor` on `self.device`, not numpy):

| Field | Actual shape | Notes |
|---|---|---|
| `data_path` | `str` | sequence identifier |
| `obj_id` | `str` | |
| `obj_verts` | `(1000, 3)` | **Sampled** via `random_sampling_pc()` (pytorch3d `sample_points_from_meshes`, seeded RNG). Fixed at 1000. Not raw mesh vertices. |
| `obj_urdf_path` | `str` | |
| `obj_trajectory` | `(T, 4, 4)` | **Homogeneous SE(3) matrices only.** Never (T,7). |
| `wrist_pos` | `(T, 3)` | meters, world frame |
| `wrist_rot` | `(T, 3, 3)` | **Rotation matrices, not axis-angle/quat.** Base class immediately does `rotmat_to_aa(...)`. Must have the dexhand's `relative_rotation` offset already baked in: `wrist_rot = mano_wrist_rotmat @ dexhand.relative_rotation` (every existing loader does this). |
| `mano_joints` | `dict` of 20 `(T, 3)` tensors | **Not** a single `(T, 21, 3)` array. Keys: `{finger}_{proximal,intermediate,distal,tip}` for index/middle/ring/pinky/thumb. Tips are **reselected mesh vertices** (specific vertex indices), not the joint-regressor tips. |
| `scene_objs` | `list` | usually `[]` |

Auto-computed by `process_data()` (call at end of `__getitem__`):
- Applies `mujoco2gym_transf` (identity in current configs)
- Computes `obj_velocity`, `obj_angular_velocity`, `wrist_velocity`, `wrist_angular_velocity`, `mano_joints_velocity`, `tips_distance` (chamfer distance from fingertips to sampled object points)
- **Velocity math hardcodes** `dt = 1/(120/skip)` — assumes 120 Hz source data. **Not** configurable. Any loader must resample to hit this assumption (see FPS note below).

Auto-computed by `load_retargeted_data()`:
- Adds `opt_wrist_pos`, `opt_wrist_rot`, `opt_dof_pos` (+velocities) from a prior kinematic-retargeting pass (`mano2dexhand.py`) if present on disk. Placeholders otherwise.

### Bimanual pipeline structure

Each `ManipData` subclass is **single-hand, single-object** — registered as `<name>_rh` and `<name>_lh` separately via `ManipDataFactory`. "Bimanual" is handled by the training env (`dexhandmanip_bih.py`) which instantiates both a right-hand and left-hand dataset and spawns two independent object actors. **Two objects (tool + target) is already supported** — one object per hand.

Existing `ManipData` subclasses live under `main/dataset/`. `grab_dataset_dexhand.py` is our closest analog for TACO (raw MANO forward pass with `manotorch.ManoLayer`), **not** OakInk-V2 (which uses a custom SMPLX layer).

### What TACO provides (per sequence)

**On-disk layout** (verified by inspecting `(brush, brush, pan)/20230919_026`):

```
data/taco/
├── Hand_Poses/
│   └── (action, tool, target)/                 ← NB: triplet = (action, tool, target), not (tool, action, object)
│       └── YYYYMMDD_NNN/                       ← session ID
│           ├── left_hand.pkl                   ← dict keyed by "00001", "00002"...
│           ├── right_hand.pkl                  ←   each frame: {'hand_pose': (48,), 'hand_trans': (3,)}
│           ├── left_hand_shape.pkl             ←   {'hand_shape': (10,) β vector}
│           └── right_hand_shape.pkl
├── Object_Poses/
│   └── (action, tool, target)/
│       └── YYYYMMDD_NNN/
│           ├── tool_<id>.npy                   ← (T, 4, 4) SE(3), ALREADY in meters
│           └── target_<id>.npy                 ← (T, 4, 4) SE(3), ALREADY in meters
└── object_models_released/
    ├── <id>_cm.obj                             ← genuinely in cm — needs ×0.01 to reach meters
    └── ...
```

**Key facts (all verified by inspection):**

- **Hand pose is 48-dim axis-angle:** 3 (global orient) + 45 (finger pose). No separate `global_orient` key — it's the first 3 elements of `hand_pose`. Matches MANO with `use_pca=False`.
- **`hand_trans` is in meters** (verified range ≈ 0.10–0.80).
- **Object trajectories are ALREADY in meters** (verified range ≈ -0.11 to 0.62). Only the **meshes** need cm→m conversion.
- **Shape β is per-sequence, per-hand** — left and right can differ (per-person-per-hand).
- **Frame counts match exactly** across left hand, right hand, tool, and target within a sequence.
- **Sequence lengths vary widely** — 292 frames observed in one sample, hand pkls up to ~17.5MB. Some are much shorter.
- **Triplet folder names contain parentheses, commas, spaces** — e.g., `(brush, brush, pan)`. Use quoted paths and `glob`, never trust unquoted shell expansion.
- **Object id cross-reference:** `tool_035.npy` ↔ `object_models_released/035_cm.obj`.
- **Native FPS confirmed at 30 Hz** — see Confirmed conventions section below.

## Gaps to bridge (implementation checklist)

- [ ] **URDF generation:** COACD convex decomposition on all TACO object meshes. Reuse `coacd_process.py` from ManipTrans. Output layout: mirror `coacd_object_preview/align_ds/<id>/scan.urdf` — one `<link name="base">` with one visual/collision pointing at the COACD merged mesh. Isaac Gym's `convex_decomposition_from_submeshes=True` handles the split at load time. ~206 unique objects.
- [ ] **Unit conversion:** **Only meshes** need cm→m (scale vertices ×0.01 before COACD). Object trajectories are already in meters — do NOT scale them.
- [ ] **MANO forward pass:** Use `manotorch.manolayer.ManoLayer(rot_mode="axisang", use_pca=False, flat_hand_mean=True, mano_assets_root="data/mano_v1_2", side=side)`. Feed `hand_pose[:3]` as global orient, `hand_pose[3:]` as finger pose, `hand_trans` as translation, and the per-sequence `hand_shape` β broadcast to all frames.
- [ ] **Fingertip reselection:** vertex indices `{index: 353, middle: 467, pinky: 695, ring: 576, thumb: 766}` — same for both hands.
- [ ] **Wrist pos/rot construction:** Copy the existing loader pattern exactly: `wrist_pos = mano_joint_0` pulled back by the standard 0.25× offset; `wrist_rot = mano_wrist_rotmat @ dexhand.relative_rotation`.
- [ ] **FPS resampling:** Target **120 Hz** with `skip=2` (not 60 Hz) to match the hardcoded velocity assumption. Linear interp for translations, **slerp** for rotations, on both hand and object trajectories.
- [ ] **Bimanual + two-object handling:** Register `taco_rh` and `taco_lh` in `main/dataset/factory.py`. Each returns single-hand + single-object data. The bimanual env wires them together.
- [ ] **Tool ↔ hand assignment:** Not derivable from filenames alone — needs a per-sequence heuristic (e.g., which object stays closer to which wrist over the trajectory) or manual annotation. For PoC, pick manually.
- [ ] **`TACOData` class:** Two subclasses (`TACORightData`, `TACOLeftData`) or one with `side` param — mimic `grab_dataset_dexhand.py`.
- [ ] **Coordinate frames:** Verify TACO's world-frame axis/up convention against ManipTrans's assumptions before trusting anything. Visual debug required.

## Conventions and gotchas

- **All fields are `torch.Tensor` on `self.device`**, not numpy arrays. The base class assumes this.
- **All units in meters** in the ManipTrans pipeline. TACO trajectories are already meters; only meshes need conversion.
- **Rotations enter as (T, 3, 3) matrices**, not quaternions or axis-angle. The base class converts to axis-angle internally.
- **Target 120 Hz with `skip=2`**, not 60 Hz — the base class velocity math hardcodes `dt = 1/(120/skip)`.
- **Slerp, not lerp, for rotations** during FPS resampling.
- **Sanity-check every new sequence visually** before running the RL pipeline. Silent frame-alignment, unit, or MANO-mean bugs will burn hours of GPU time otherwise.
- **Fixed random seeds** for anything reproducible (COACD may be non-deterministic — that's fine).
- **Working inside the ManipTrans repo** on the `taco-retargeting` branch. New files (TACOData class, preprocessing scripts) can be added, ideally under a new `taco/` subfolder or clearly named files. Do not modify existing GRAB / OakInk-V2 / FAVOR loaders. Generated data (URDFs, checkpoints, trajectories) must be `.gitignore`d.
- **Chosen PoC sequence:** `(brush, brush, pan)/20230919_026` — 292 frames, simple geometry, clean single-behavior motion, all frame counts aligned. Triplet = (action, tool, target).
- **Finger-pose FPS resampling is componentwise linear interpolation on axis-angle values, NOT per-joint slerp** (global wrist orientation and object rotation DO use proper slerp). This is a deliberate simplification on the assumption that adjacent-frame finger rotations are small — flagged in-code. Revisit if retargeting error concentrates in fast finger motions.
- **`isaacgym` must be imported before `torch`, anywhere in the process.** `main/dataset/__init__.py` auto-imports every file in that directory, including `mano2dexhand.py`, which needs `isaacgym` imported first. Any script touching `main.dataset` AND needing a real `DexHand` instance (for `relative_rotation`/`n_dofs`) must be a standalone entry script with `isaacgym` imported first (see `taco_verify_m2.py`) — cannot be invoked as `-m main.dataset...`. This is a pre-existing repo quirk, not something we introduced.

## Progress log

**M1 — URDF generation: COMPLETE.**
- `data/taco/meshes_m/035.obj`, `data/taco/meshes_m/057.obj` — cm→m scaled meshes (verified bounding boxes match expected: brush 0.325×0.084×0.083 m, pan 0.240×0.190×0.142 m)
- `data/taco/urdfs/035/035.urdf` (+ `collision.obj`, 11 COACD parts) — brush, visually verified in MeshLab
- `data/taco/urdfs/057/057.urdf` (+ `collision.obj`, 32 COACD parts, capped) — pan, visually verified in MeshLab
- COACD params used (matching OakInk-V2 documented defaults): `--max-convex-hull 32 --seed 1 -mi 2000 -md 5 -t 0.07`
- Both verified loading in Isaac Gym with `convex_decomposition_from_submeshes=True`, correct rigid body/shape counts, no errors
- All of `data/taco/` already covered by `.gitignore`

**M2 — TACOData loader for one sequence: COMPLETE and fully verified.**
- `main/dataset/taco_dataset_dexhand.py` — `TACODataBase` (mirrors `grab_dataset_dexhand.py`) + `TACORightData`/`TACOLeftData` subclasses, plus `run_verification()` and `run_distance_check()`.
- `scripts/taco/taco_verify_m2.py`, `scripts/taco/taco_distance_check.py` — standalone entry scripts (isaacgym import-order gotcha applies, see below). Must be run from repo root (e.g. `python scripts/taco/taco_verify_m2.py`).
- **Verified:** `Tf=583` frames (292 @ 30Hz → 120Hz-equivalent → `skip=2`), all fields match the `ManipData` interface. Hand/object assignment confirmed with zero crossovers across all 583 frames: right hand → tool (brush), left hand → target (pan).
- **Note carried into M3:** frames ~500–583 show a sharp increase in right-wrist-to-object distance — plausibly the hand releasing the brush at the end of the motion.

**M3 — Register + kinematic retargeting: COMPLETE, validated across multiple sequences + GRAB baseline.**
- Registered `taco_rh`/`taco_lh` in `factory.py`; `taco_dataset_dexhand.py` generalized to a `SEQUENCES` registry (index 0 = original PoC, regression-checked unchanged behavior).
- All verification/comparison scripts (`taco_verify_m2.py`, `taco_verify_m3.py`, `taco_distance_check.py`, `compare_verify_m3.py`) live under `scripts/taco/` — moved out of repo root for tidiness. All are CWD-relative (no `__file__` path tricks) — must be run from the repo root, e.g. `python scripts/taco/taco_verify_m3.py`.
- Convergence check (iter 2000 vs. 5000, PoC sequence): worst frames unchanged, only marginal (~10-20%) error reduction — confirms spikes are genuine kinematic infeasibility, not under-optimization.
- **Multi-sequence validation (iter=5000, all runs):**

| Sequence | side | Tf | mean (cm) | max (cm) | worst frame |
|---|---|---|---|---|---|
| brush/pan 20230919_026 (PoC) | right (tool) | 583 | 1.32 | 21.43 | 564 |
| | left (target) | 583 | 2.19 | 9.66 | 338 |
| hammer/helmet 20231002_063 | right (tool) | 169 | 1.64 | 9.34 | 92 |
| | left (target) | 169 | 0.92 | 1.86 | 79 |
| spoon/bowl 20231104_179 | right (tool) | 215 | 0.58 | 3.09 | 26 |
| | left (target) | 215 | 1.15 | 13.86 | 210 |
| knife/plate 20231020_232 | right (tool) | 211 | 0.50 | 2.51 | 187 |
| | left (target) | 211 | 1.74 | 14.08 | 112 |
| **GRAB 102 (baseline)** | right | 60 | **0.71** | **1.64** | 13 |

- **Conclusion:** TACO's typical-case (mean) error is often as good as GRAB's (several combos 0.5–0.6cm). But every TACO sequence tested has at least one spike (max 1.86–21.43cm) that GRAB's demo (max 1.64cm, fully converged, no spikes) doesn't exhibit. GRAB hit the plateau-break condition at iter~4600; no TACO sequence did within 5000 iters (consistent with more-iterations giving only marginal gains, not resolution, per the convergence check).
- Worst frames cluster near sequence end in 3/4 cases (tool release/regrasp motion, consistent with the M2 distance anomaly) — but NOT universally: knife/plate's worst frame (112/211) is mid-sequence, indicating some sequences have hard grasp transitions mid-motion too.
- **Accepted as baseline.** These are genuine kinematic infeasibilities that the RL residual policy is designed to absorb — supported by same-pipeline evidence (GRAB baseline), not just assumption. Proceeding to M4 on the original PoC sequence.

**M4 — planning complete, NOT launched. Two methodological concerns raised by supervisor, under investigation before launch.**

**M4 plan (validated, ready once concerns below are resolved):**
- Entry point: `main/rl/train.py` (Hydra), task config `main/cfg/task/ResDexHand.yaml`, RL config `main/cfg/rl_train/ResDexHandPPO.yaml`.
- Proposed command:
  ```
  python main/rl/train.py task=ResDexHand dexhand=inspire side=BiH headless=true num_envs=4096 \
    learning_rate=2e-4 test=false randomStateInit=true dataIndices=[t0] \
    rh_base_model_checkpoint=assets/imitator_ckp/imitator_rh_inspire.pth \
    lh_base_model_checkpoint=assets/imitator_ckp/imitator_lh_inspire.pth \
    early_stop_epochs=1000 actionsMovingAverage=0.4 experiment=cross_taco_t0_inspire
  ```
- **Bug caught: README/config default checkpoint paths are wrong.** Both say `assets/imitator_rh_inspire.pth` / `assets/imitator_lh_inspire.pth`, but the real files live at `assets/imitator_ckp/imitator_rh_inspire.pth` / `assets/imitator_ckp/imitator_lh_inspire.pth`. Must override explicitly or training fails at startup.
- Key hyperparams: `num_envs=4096`, `horizon_length=32`, `minibatch_size=1024`, `mini_epochs=5`, `gamma=0.99`, `learning_rate=2e-4`, `early_stop_epochs=1000` (minimum-epoch floor before a slope-based plateau check can trigger, not a patience counter).
- Reward: weighted sum of exponential tracking terms (wrist pos/rot, per-finger-tip position with per-finger weights, proximal/intermediate joints, object pos/rot, velocities, fingertip contact force, power penalties) + a tightening schedule that progressively shrinks tolerance.
- Estimated cost (from a real prior identical-config run on this hardware, `runs/cross_20aed@0_inspire__06-30-10-25-22/`, OakInk-V2 20aed@0, side=BiH, num_envs=4096): **~15.1 sec/epoch, ~4–4.5 hours wall-clock** for early_stop_epochs=1000, single GPU. That reference run converged to ~0.96–0.98 success rate, reward plateaued ~epoch 700+. GPU memory unmeasured historically — recommend watching `nvidia-smi` for the first few minutes of the real run.

**Concerns raised before launch (supervisor), under investigation:**
1. **Coordinate frame consistency:** is TACO's world frame (origin, up-axis, handedness) actually consistent with what `mano2dexhand.py` assumes / what GRAB already satisfies? Previously logged as "looks plausible, no issue found" (soft check only) — now being rigorously verified with a direct frame comparison between GRAB and TACO's first-frame poses.
2. **GRAB baseline too thin:** our only GRAB comparison (sequence 102) is Tf=60 — much shorter than our TACO sequences (169–583 frames). Running at least one longer GRAB sequence (checking `data/grab_demo/grab.zip` for additional sequences first) before trusting the GRAB-vs-TACO comparison.

**M4 launch is on hold pending resolution of both.**

## Phased plan

**Phase 1 — Single-sequence proof of concept** (current milestone)
1. Pick one TACO sequence with a simple tool (e.g., a screwdriver-style triplet).
2. Load and inspect: MANO params, object trajectories, meshes. Visualize alignment.
3. COACD-decompose that sequence's tool + target objects → URDFs.
4. Write minimal `TACOData` that returns the ManipTrans dict for this one sequence.
5. Run `mano2dexhand.py` (kinematic retargeting) → visually verify Inspire Hand tracks MANO fingertips. This produces the reference trajectory only, not the final output.
6. Run the full ManipTrans residual RL policy on that sequence → verify convergence and that the rollout in Isaac Gym looks sensible. **This is the actual deliverable format.**

**Phase 2 — Scale to full dataset**
- Batch URDF generation across all objects.
- Extend `TACOData` to iterate over all sequences.
- Register properly in `factory.py`.
- Job-array the RL training across the 10× 4090s.

**Phase 3 — Validation and packaging**
- Per-sequence tracking-error metrics.
- Failure-mode analysis (which triplets fail — expect thin/small objects to be hard).
- Package outputs in DexManipNet-compatible format.

## How to verify things work

- **Kinematic retargeting sanity:** Overlay Inspire Hand fingertips on MANO fingertips in a 3D viewer. Mean fingertip error < 1cm is a reasonable target.
- **Object trajectory sanity:** Play the object mesh along its trajectory alongside the hand — the tool should stay approximately in the grasping hand, not float away.
- **RL convergence:** Watch the reward curves. If they plateau at low values, the reference trajectory has kinematic infeasibilities (contact penetrations, unreachable poses) — this is expected for some sequences and is what the residual policy is meant to fix.
- **Rollout visual check:** After training, roll out the policy in Isaac Gym and record video. This is the ultimate test.

## Confirmed direction

**Full RL residual policy** (confirmed with supervisor). "Action" means trained Isaac Gym policies via the full ManipTrans pipeline — not just kinematic retargeting from `mano2dexhand.py`. Kinematic retargeting is only the intermediate step that produces the reference trajectory the residual policy trains against.

Implication: compute-heavy. Each sequence requires an RL run (parallel envs × thousands of epochs). Plan job scheduling on the 10× 4090s accordingly, and start with a small subset before scaling.

## Confirmed conventions (resolved from local repos + empirical checks)

- **`flat_hand_mean = True`**. TACO uses `manopth.ManoLayer(..., use_pca=False, ncomps=45, center_idx=0)` in `dataset_utils/hand_pose_loader.py` — manopth's default is `flat_hand_mean=True`. manotorch and manopth read the same `hands_mean` from the MANO asset file and apply the same formula, so `manotorch.ManoLayer(flat_hand_mean=True)` reproduces TACO's decoding exactly. Empirically verified on frame 00150 of the PoC sequence: `True` gives a natural brushing grip, `False` gives an over-curled clenched hand (double-applied mean).
- **TACO native FPS = 30**. Confirmed in `dataset_utils/visualization.py:60` (`duration=(1000/(30//sampling_rate))`) and in the README's ffmpeg command (`fps=30`). Resample 30 → 120 Hz with `skip=2`.
- **Fingertip vertex indices work for both hands**: `{index: 353, middle: 467, pinky: 695, ring: 576, thumb: 766}`. `MANO_LEFT.pkl` and `MANO_RIGHT.pkl` share topology and are exact x-mirror images at these 5 vertices — verified numerically. Do NOT switch to manotorch's canonical tip list (444/445 asymmetry between hands is a topology quirk that would introduce a subtle bug).
- **MANO layer args:** `manotorch.manolayer.ManoLayer(rot_mode="axisang", side=side, mano_assets_root="data/mano_v1_2", use_pca=False, flat_hand_mean=True)`.

## Open questions

**Resolved:**

1. **`center_idx = None`** — chosen for structural consistency with `grab_dataset_dexhand.py` and the rest of ManipTrans, deviating from TACO's own `center_idx=0` convention. Confirmed in M2.
2. **Hand/object assignment for PoC sequence:** right hand → tool (035, brush), left hand → target (057, pan). Confirmed with zero crossovers across all 583 frames (see Progress log, M2).

**Non-blocking (handle when scaling beyond PoC):**

3. TACO world-frame axis/up convention — looks plausible from M2's sanity checks, no issue found so far.
4. General per-sequence tool→hand assignment heuristic (wrist-distance approach validated for this one sequence; confirm it generalizes before batch-processing many sequences).

## Scope questions (awaiting supervisor input)

- Full 2317-sequence TACO v1, or start with pre-released 244-sequence subset?
- Retarget everything, or filter to specific TACO test splits (S1–S4)?

## Working style

- Discuss non-trivial design decisions before writing large amounts of code.
- For any new script, add a `--dry-run` or single-sequence mode before running on the full dataset.
- Log everything: TensorBoard for RL, structured JSON for preprocessing statistics.
- When something looks wrong in a rendered visualization, stop and debug — don't paper over it.