# Cube-Subset Hand-Eye Pose Sampling — Design

**Date:** 2026-09-21  
**Status:** Approved  
**Context:** Replace A-style (17 separate rotation/translation) offsets in `easy_handeye2_franka_auto` with a cube corner+center × EE-tilt pool, randomly subsampled for calibration quality vs run length.

## Problem

A-style poses apply either an EE rotation or a base translation, never both. For eye-on-base calibration, combining translation and orientation variation at each sample improves excitation of the hand-eye solve. Running the full combinatorial set is slow and often hits FOV/joint limits.

## Goals

- Build a pose pool: **cube center + 8 corners** (base-frame translations) × **±X/±Y/±Z EE tilts** = **54** poses.
- Select a **uniform random subset** (default **15**) with a **reproducible seed**.
- Replace A-style as the only offset generator (no dual-mode).
- On motion or sample failure: **skip that pose and continue** (do not abort the whole run).
- Keep existing motion stack (`RobotPoseSource.go_to` / `FrankaInterface.reset_to_start_pose`) and TF/sampling path unchanged.

## Non-goals

- Stratified or curated pose selection.
- FOV / collision / reachability prechecks.
- Base-frame (global) tilts.
- Keeping A-style as a CLI option.

## Pose math

Given free-drive home `T_home` (4×4, base←EE):

**Translations (base):** half-size `d` meters

| Index | Offset |
|-------|--------|
| 0 | `(0, 0, 0)` center |
| 1–8 | all `(±d, ±d, ±d)` corners |

**Rotations (EE):** angle `α` radians; right-multiply onto home orientation (`q_home * q_delta`):

| Index | Delta |
|-------|--------|
| 0–5 | ±α about EE X, Y, Z |

**Pool:** for each of 9 translations and 6 rotations, pose `T` has:

- `T[:3, 3] = home[:3, 3] + t_base`
- `T[:3, :3]` from `q_home * q_delta` (position from translation; orientation independent of which corner)

**Subset:** `random.Random(seed).sample(pool, n_poses)` without replacement.  
If `n_poses > 54` or `n_poses < 1` → raise `ValueError`.  
If `n_poses == 54` → return full pool in sample order (still shuffled by `sample`).

## CLI

| Flag | Default | Notes |
|------|---------|--------|
| `--rotation-delta-degrees` | cal: 25 / motion-test: 15 | EE tilt magnitude |
| `--cube-half-size-meters` | cal: 0.05 / motion-test: 0.05 | replaces `--translation-delta-meters` |
| `--n-poses` | 15 | subset size |
| `--seed` | 0 | RNG seed |
| `--first-n` | 0 | optional truncate *after* subset (bring-up) |

Remove: `--translation-delta-meters`, `--translations-only`.

## Failure policy

- Motion/`go_to` exception → log error, **continue** to next pose.
- `take_sample` does not grow list → warn, continue.
- After loop: compute/save iff `successful_samples >= max(min_samples, 3)`.
- `--return-home` runs if home was captured (even if some poses failed).

## Files

- Modify: `handeye_offsets.py` — replace `compute_poses_around_state` with `compute_cube_poses`.
- Modify: `handeye_auto_calibrate.py`, `handeye_motion_test.py`, tests, README.
- Design/plan docs under `docs/superpowers/`.

## Success criteria

- Unit tests: pool=54 before subsample; same seed → same poses; corners at `(±d,±d,±d)`; EE right-multiply tilts; invalid `n_poses` errors.
- Motion-test and auto-calibrate use the new generator and skip-on-failure policy.
- README documents the new flags.
