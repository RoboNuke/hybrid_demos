# robonuke_project_template

The starting point for a research project built on
[robonuke_rl_core](https://github.com/RoboNuke/robonuke_rl_core). 
## Start a project

1. GitHub: **Use this template -> Create a new repository** -> name the new repo after
   the project. This copies the files into a fresh repo with no shared history — it is
   not a fork, so the new project never shows up as one.
2. Rename `my_project/` to your project's name and fix the imports in `scripts/*.py`
   (`setup()` is the only place they appear).
3. Make sure the package is installed (once, for all projects):

   ```bash
   conda activate general
   pip install -e ~/robonuke_rl_core
   ```

   The package expects an Isaac Lab environment (built against Isaac Lab 0.47.1 /
   Isaac Sim 5.1.0 / Python 3.11) and brings its other dependencies itself.

## What lives where

| piece | where | registered by |
| --- | --- | --- |
| environments (env + task cfg + `gym.register`) | `my_project/tasks/` | importing the module in `setup()` |
| config sections (`@dataclass` + `validate`) | `my_project/cfg.py` | `register_section` in `setup()` |
| auxiliary losses (`AuxLoss` subclasses) | `my_project/losses.py` | `@register_loss`, imported in `setup()` |
| configs (`base:` chains, override only what differs; layout below) | `configs/` | `--config` / `--eval_config` |
| cluster resources (the `hpc` section: account, partitions, walltime, image) | `configs/base/hpc.yaml` | chained under every experiment |
| SLURM submitters (thin callers of `robonuke_rl_core.hpc`) | `launchers/` | — |
| model architectures (`nn.Module` + `<Arch>ModelCfg`) | `my_project/models/` | `register_architecture` in `setup()` |
| overlays | same pattern: register in `setup()` | package CLAUDE.md recipes |
| analysis notebooks (cells: imports, variables, download, preprocessing, then one plot / table each; downloads cached in the git-ignored `data_analysis/cache/`) | `data_analysis/` | — |

The demo task (`Isaac-Forge-DemoPegInsert-Direct-v0`) subclasses Isaac Lab's Forge peg
insert unchanged — "forge" in the task name is what lets the package's Forge wrappers
(controller, fragile peg, efficient reset, contact sensing) apply. Replace its `pass`
bodies with your observations, rewards and resets.

## Configs

* `configs/base/` -- the env: one file per task (`fb_one_leg.yaml`: the task id and its physics, everything
  independent of the training method; `fb_one_leg_rel.yaml`: the same with the interface relative to the
  end-effector, only its task id differs; `fb_one_leg_delta.yaml`: the same with only the action relative
  to the end-effector), chained onto `hpc.yaml` (the cluster's resources). Base one on it
  and you get a known task with known physics.
* `configs/eval/` -- only what one test changes on top of the trained run's own config (more friction, a
  recording: `one_leg_phases.yaml`; the residual watcher's 100 envs: `residual_watch_furniture.yaml`), chained onto `hpc.yaml` for the cluster launcher's resources.
* `configs/experiments/<group>/` -- one folder per experiment group: `_foundation.yaml` (based on a
  `base/` task) holds what every training of the group shares -- the seed, the wandb entity, every
  learning parameter -- and one launch config per training (`bc_<group>.yaml`, `residual_<group>.yaml`)
  sets its learner, model, wandb project and group, and env count. The leading `_` keeps
  `launch_train.py <folder>` from submitting the foundation.

## Run

```bash
# train the BC policy, then a residual on it (configs/experiments/resip_replica/)
python scripts/train_bc.py --config configs/experiments/resip_replica/bc_resip_replica.yaml --headless
python scripts/make_demo_transitions.py --config configs/experiments/resip_replica/residual_resip_replica.yaml
python scripts/train.py --config configs/experiments/resip_replica/residual_resip_replica.yaml --headless
# the same BC on the interface relative to the end-effector (configs/experiments/resip_replica_rel/)
python scripts/train_bc.py --config configs/experiments/resip_replica_rel/bc_resip_replica_rel.yaml --headless
# the same BC with only the action relative to the end-effector (configs/experiments/policy_dec_replica/)
python scripts/train_bc.py --config configs/experiments/policy_dec_replica/bc_policy_dec_replica.yaml --headless
# the same BC with the chunk relative to the EE pose at the observation (configs/experiments/policy_dec_replica_chunk/)
python scripts/train_bc.py --config configs/experiments/policy_dec_replica_chunk/bc_policy_dec_replica_chunk.yaml --headless
# ... with the action scaled per dim, zero-centred (configs/experiments/policy_dec_replica_chunk_zero_centered_norm/)
python scripts/train_bc.py --config configs/experiments/policy_dec_replica_chunk_zero_centered_norm/bc_policy_dec_replica_chunk_zero_centered_norm.yaml --headless

# evaluate / record a trained run (downloads policy + config from the wandb run's files)
python scripts/eval.py --run hur/hybrid_demo_furniture/<run_name> \
    --eval_config configs/eval/one_leg_phases.yaml --headless

# watch it live (j pause / k reset / l quit), or inspect the reset distribution
python scripts/debug.py --run hur/hybrid_demo_furniture/<run_name>
python scripts/debug.py --run hur/hybrid_demo_furniture/<run_name> --resets --hold_seconds 2
```

## Run on the cluster

The launchers submit SLURM + Apptainer jobs from a login node — a light python is enough
(omegaconf + pyyaml; wandb only for `launch_eval`'s run query). Resources come from the
`hpc` config section (`configs/base/hpc.yaml`, overridable per experiment or on the CLI);
the field reference is the package README. Names derive from wandb: `--project` and
`--group_prefix` are required, group = `{group_prefix}_{config_stem}` (sweeps append
`_{label}-{value}`), the SLURM job name is the group, runs are `{group}_a{i}`.

```bash
# one job per launch config in the group's folder (the _foundation.yaml is skipped)
python launchers/launch_train.py configs/experiments/resip_replica --project hybrid_demo_furniture \
    --group_prefix oct21

# one job per swept value (group oct21_residual_resip_replica_lr-1.0e-4, ...)
python launchers/launch_sweep.py configs/experiments/resip_replica/residual_resip_replica.yaml \
    --project hybrid_demo_furniture_residual --group_prefix oct21 \
    --sweep_param sac.actor_lr --label lr --value 1.0e-4 --value 3.0e-4

# one eval job per (trained run x eval config), selected by group
python launchers/launch_eval.py --eval_config configs/eval/one_leg_phases.yaml \
    --project hybrid_demo_furniture --group oct21_bc_resip_replica
```

`--dry_run` on any of them prints the exact `sbatch` commands and submits nothing.
`WANDB_API_KEY` is read from your login shell (never a config field); without it the job
logs offline instead of crashing.

## Config sections this project adds

| field | type | default | what it does |
| --- | --- | --- | --- |
| `demo.reward_scale` | float | `1.0` | example field of the example section — replace with your project's |

## The Forge demo recorder's fields

`scripts/record_demos.py` (an Xbox-style controller, `my_project/demos/`) and `scripts/replay_demos.py`
record and replay Forge peg-insert demos into a LeRobot-style dataset.

### The policy interface (observation and action)

**Observation** (24 dims) = `task.cfg.obs_order`, then the previous action: `fingertip_pos` (Forge's noisy
fingertip estimate, 3), `fingertip_quat` (noisy, 4), `hole_estimate` (the noisy hole estimate,
3, a term `my_project/obs_terms.py` adds), `ee_linvel` (3), `ee_angvel` (3), `ft_force` (3),
`force_threshold` (1), previous action (what the env applied after its action smoothing, as a
world pose `[x, y, z, yaw]` in the action's units, 4; at an episode's first step, the pose Forge
seeds its smoothing with). Positions are in the env (robot base) frame.

**Action** (`action.world_pose`, 4 dims): the fingertip's world pose `[x, y, z, yaw]`,
converted into Forge's 7-d action by `my_project/wrappers.py` (roll / pitch 0, success
prediction -1: Forge's reward code indexes it, so it stays in Forge's internal action).

| field | type | default | what it does |
| --- | --- | --- | --- |
| `action.observe_previous_action` | bool | `true` | with `action.world_pose`: keep the previous action (the smoothed world pose the env applied, 4) at the end of the policy observation; `false` drops it, and Forge's own 7, entirely |
| `action.bolt_relative` | bool | `false` | with `action.world_pose`: `[x, y, z]` is relative to the env's bolt (hole) estimate, Forge's own action frame, instead of the env origin; yaw stays absolute |
| `action.world_pose` | bool | `false` | the policy's action is the fingertip's world pose `[x, y, z, yaw]` (m, rad, env frame) instead of Forge's normalized 7-d action; the observation's previous action becomes the smoothed action the env applied, as a world pose (replacing Forge's 7-d previous action) |

### Fields of the `record` section

Read only by the recorder (`my_project/cfg.py:RecordCfg`).

| field | type | default | what it does |
| --- | --- | --- | --- |
| `record.dataset_dir` | str \| null | `null` | dataset to create or append to; required to record |
| `record.input` | str | `gamepad` | `gamepad` (GUI), `expert` (the ground-truth oracle below) or `search_expert` (noisy observations only, spiral search; below); the experts run headless |
| `record.max_speed_pos` | float | `0.03` | fingertip speed at full stick, m/s |
| `record.max_speed_yaw` | float | `0.5` | yaw rate at full stick, rad/s |
| `record.fine_scale` | float | `0.25` | speed multiplier while RB is held |
| `record.leash_pos` | float | `0.01` | the setpoint stays within this of the fingertip, m per axis |
| `record.leash_yaw` | float | `0.15` | the yaw setpoint stays within this of the fingertip's yaw, rad |
| `record.stick_dead_zone` | float | `0.1` | stick deflection below this reads as 0 |
| `record.axis_signs` | list[float] | `[1, 1, 1, 1]` | sign per command axis (vx, vy, vz, yaw), to flip a direction that feels inverted |
| `record.success_hold_steps` | int | `5` | save once the success check has held this many steps; `0`: only A saves |
| `record.save_timeouts` | bool | `false` | keep episodes that hit the time limit (marked unsuccessful) |
| `record.realtime` | bool | `true` | pace the loop to the control rate in wall-clock time |
| `record.num_episodes` | int | `0` | stop after this many saved episodes; `0`: until START |

### Fields of the scripted expert (`record.input: expert`, `my_project/demos/expert.py`)

| field | type | default | what it does |
| --- | --- | --- | --- |
| `record.expert.surface_radius_min` | float | `0.010` | nearest the peg first lands to the hole axis, m (the opening is 8.1 mm across, the peg 8 mm) |
| `record.expert.surface_radius_max` | float | `0.014` | farthest, m (the block's top is 40 mm across) |
| `record.expert.approach_clearance` | float | `0.010` | the peg's bottom flies this far above the hole top before descending, m |
| `record.expert.descend_speed` | float | `0.02` | lowering speed until contact, m/s |
| `record.expert.contact_force` | float | `1.5` | contact once the force's magnitude passes this, N |
| `record.expert.slide_speed` | float | `0.01` | speed of the slide onto the hole axis, m/s |
| `record.expert.push_depth` | float | `0.003` | the height target stays this far below the fingertip while sliding and inserting, m |
| `record.expert.insert_push_depth` | float \| null | `null` | the same lead once the peg is in the hole (`null`: `push_depth`); sets the insertion speed and may exceed the sliding push, which must stay gentle or friction stops the slide short of the hole, m |
| `record.expert.drop_depth` | float | `0.002` | inserting once the peg's bottom is this far below the hole top, m |

### Fields of the searching expert (`record.input: search_expert`, `my_project/demos/search_expert.py`)

| field | type | default | what it does |
| --- | --- | --- | --- |
| `record.search_expert.approach_height` | float | `0.045` | fingertip height above the believed hole top for the approach, m |
| `record.search_expert.descend_speed` | float | `0.02` | lowering speed until contact, m/s |
| `record.search_expert.contact_force` | float | `3.0` | contact once the 3-step mean of the noisy force's magnitude passes this, N |
| `record.search_expert.search_radius` | float | `0.005` | the spiral restarts past this radius around the believed hole centre, m |
| `record.search_expert.search_pitch` | float | `0.0008` | distance between spiral turns, m |
| `record.search_expert.search_speed` | float | `0.01` | speed along the spiral, m/s |
| `record.search_expert.push_depth` | float | `0.003` | the height target stays this far below the fingertip while searching, m |
| `record.search_expert.drop_depth` | float | `0.002` | inserting once the fingertip is this far below the contact height, m |
| `record.search_expert.insert_push_depth` | float | `0.003` | the same lead while inserting, m |

## FurnitureBench one_leg (resip_replica)

`configs/base/fb_one_leg.yaml`: `Isaac-Furniture-OneLeg-FBAbs-ResiP-v0` from
[Robonuke_Furniture_Bench](https://github.com/RoboNuke/Robonuke_Furniture_Bench) (`pip install -e`
that checkout, plus `pip install "pyarrow<21"` in the Isaac Sim env, whose NumPy is 1.x; the
scripts' `setup()` imports `furniture_lab.tasks`): ResiP's sim (10 Hz, differential IK into joint PD,
absolute EE targets with rot6d, action 10, the 63-wide observation in the robot-base frame) with the
task's own defaults, except what `physic_aligned_evalHappy_a0` trained with (1000-step episodes, the top's
SDF at 1024, the position gripper with self-collisions, Isaac Lab's IK Jacobian). The env is
manager-based, so the two env families differ underneath: `my_project/families.py` (which family a task
name is, its observation layout, its action layout) and `my_project/envhooks.py` (how a wrapper reaches a
Direct env vs. a manager-based one) hold the differences, and the residual, the history and the demo
conversion run on both. The Forge-only pieces stay Forge-only: the package's controller / fragile /
contact / task_metrics wrappers, `action.world_pose`, `hybrid`, `episode.reset_on_success` (the furniture
env ends an episode on assembly itself).

**Demos** come from that repo's collector (`scripts/collect_scripted.py`, FurnitureBench's pkl format;
the command for `data/physic_aligned` is in `bc_resip_replica.yaml`) and are converted, with no Isaac
Sim, into the recorder's dataset layout (`scripts/import_furniture_demos.py`). Per frame:
`observation.state` (49) and `action` (8) in FurnitureBench's interface, plus ResiP's derived exactly
from the same frames (`furniture_lab.demos.fb_abs_frames`): `observation.state_abs` and `action.abs`
(10, the absolute target the delta commanded: `[ee_pos + dpos, rot6d(R_ee R_dq), gripper]`). Raw
material beside them: `observation.joint_positions`, `observation.next_state` /
`observation.next_state_abs`, `reward` (the env's sparse assembly reward), `skill`. For the task relative
to the end-effector (`--task_name Isaac-Furniture-OneLeg-FBRel-ResiP-v0`) also `observation.state_rel`
(55: every pose relative to the EE pose, in its frame; no EE pose), `action.rel` (10: `action.abs`'s
target relative to that frame's EE pose, `[dpos, drot, gripper]` in the EE's frame, the pose part in
[-1, 1]: `dpos` in units of 0.12 m, `drot` a rot6d minus the identity's, the few longer leads -- the
post-assembly lift -- clipped as the task's term clips them) and
`observation.next_state_rel`, built from the ResiP `observation.state_abs` (63) by
`furniture_lab.demos.ee_relative_states` / `ee_relative_actions`, the NumPy twins of the task's terms
(the command for `data/physic_aligned_rel`, the same 50 demos, is in `resip_replica_rel/_foundation.yaml`).
For the ResiP observation with the action relative to the end-effector (`--task_name
Isaac-Furniture-OneLeg-FBDelta-ResiP-v0`) every FBAbs-ResiP column, plus `action.delta` (7: `action.abs`'s
target relative to that frame's EE pose, `[dpos, drot, gripper]` in the EE's frame, the pose part in
[-1, 1]: `dpos` in units of 0.12 m, `drot` an axis-angle in rad, clipped as the task's term clips it;
`furniture_lab.demos.ee_delta_actions`; the command for `data/physic_aligned_delta` is in
`policy_dec_replica/_foundation.yaml`).
`meta/episodes.jsonl` carries each demo's success and source file; re-running appends only new
files. `--include_failures` also imports `fails/*.pkl` (success false). Imported demos carry no
recording session (no env snapshot), so a residual run's `--check_in_sim` replay is Forge-only.

**Relative to the end-effector** (`configs/experiments/resip_replica_rel/`, on `base/fb_one_leg_rel.yaml`:
`Isaac-Furniture-OneLeg-FBRel-ResiP-v0`, the same sim and physics): `bc_resip_replica_rel.yaml` is
`bc_resip_replica.yaml` on `data/physic_aligned_rel`'s `observation.state_rel` / `action.rel`, for 150k
gradient steps (the cosine schedule spans them). Its action is used as it is (`bc.normalize_action:
false`): it is already in [-1, 1], so a zero output -- the warm start's buffer at an episode's start
included -- holds the EE instead of decoding to the demos' mid-range motion. `my_project/families.py`'s
`is_rel` names the interface; its action has the absolute target's dims (`[pos 3, rot6d 6, gripper]`,
the rot6d minus the identity's), so the action layout and the BC metric groups are the FBAbs ones.

**The action relative to the end-effector** (`configs/experiments/policy_dec_replica/`, on
`base/fb_one_leg_delta.yaml`: `Isaac-Furniture-OneLeg-FBDelta-ResiP-v0`, the same sim, physics and
observation): `bc_policy_dec_replica.yaml` is `bc_resip_replica_rel.yaml` on the ResiP observation
(`observation.state_abs`, 63, min-max scaled as evalHappy's) and `data/physic_aligned_delta`'s
`action.delta` (7, its rotation a 3-d axis-angle), used as it is (`bc.normalize_action: false`), for
150k gradient steps. `my_project/families.py`'s `is_delta` names the interface: the FBAbs ResiP
observation layout, the action layout `[pos 3, axis_angle 3, gripper]` and the BC metric groups
`pos` / `rot` / `gripper` on it.

**The chunk relative to the EE pose at the observation** (`configs/experiments/policy_dec_replica_chunk/`,
on `base/fb_one_leg.yaml`: the FBAbs ResiP task itself): `bc_policy_dec_replica_chunk.yaml` is
`bc_resip_replica.yaml` on the same demos and columns (`data/physic_aligned`, `observation.state_abs` /
`action.abs`) with `model.actor.action_frame: ee_at_obs`: the network predicts every step of its chunk
in m and rad in the EE frame of the observation it acts on (not of the pose each step ends up at), used
as it is (`bc.normalize_action: false`) and unclipped (`model.actor.clip_sample: false`), for 150k
gradient steps. The env gets the absolute targets those make with that pose.
`configs/experiments/policy_dec_replica_chunk_zero_centered_norm/` is the same with the action scaled per
dim by `max(|min|, |max|)` of its labels (`bc.symmetric_action`; zero stays the observation's pose) and
clipped to [-1, 1] as the reference clips (`model.actor.clip_sample` at its default).

**Training** (`configs/experiments/resip_replica/`): `bc_resip_replica.yaml` is ResiP's released
one_leg recipe (batch 1024, warmup 2000, per-dim scaling, 512 rollouts every 10 epochs whatever the
test loss, warm start, 16 DDIM steps) on `data/physic_aligned`; `residual_resip_replica.yaml` is
residual SAC on the frozen `physic_aligned_evalHappy_a0`, named by `residual.base_project` /
`base_run_name` and downloaded from wandb, logging to its own project
(`hybrid_demo_furniture_residual`). Its demo transitions read the importer's
`observation.state_abs` / `action.abs` / `observation.next_state_abs`. Each demo ends where the
training env would have ended it: the env's own assembled test (`furniture.success_mode: soft`) runs
on the recorded part poses after every step, the first success is the demo's terminal step, and the
recording's rows after it are dropped (the scripted expert holds its own check for 5 steps, then
releases and lifts: 9 rows per one_leg demo). The reward is what the env's reward terms pay, with no
`step_dt`, which the package divides out of a manager-based env's reward (a success pays 1). A file
converted before this (`meta.furniture_format` below 2) is refused.
`residual_serl_g099_qmin.yaml` is the same residual with `sac.discount_factor` 0.99 and the actor on
the min over the critics (`sac.actor_q_reduction: min`), in its own group with its own demo file;
`residual_serl_g099_qmin_std01_fixed_alpha.yaml` is that one with a small fixed exploration (the actor's
std held in [0.08, 0.10] by `model.actor.min_log_std` / `max_log_std`) and a fixed temperature
(`sac.learn_entropy: false`, alpha 0.01); `residual_serl_g0995_qmin_std01_fixed_alpha.yaml` is that one
with `sac.discount_factor` 0.995, so a success still pays the states hundreds of steps before it;
`residual_serl_g0995_qmin_std01_fixed_alpha_scale002.yaml` is that one with the residual's authority at 2%
of the base's range (`residual.action_scale_fraction: 0.02`, at most ~3 mm per dim), logging every 1000
steps with `episode/success` the mean of the last 10 episodes (`trainer.success_window: 10`):

```bash
python scripts/train_bc.py --config configs/experiments/resip_replica/bc_resip_replica.yaml --headless
python scripts/make_demo_transitions.py --config configs/experiments/resip_replica/residual_resip_replica.yaml
python scripts/train.py --config configs/experiments/resip_replica/residual_resip_replica.yaml --headless
```

To watch or record a checkpoint: `scripts/debug.py --run <entity>/<project>/<run>` (live viewer), or
`scripts/eval.py --run <entity>/<project>/<run> --eval_config configs/eval/one_leg_phases.yaml --headless`
(the mp4s, summary and trace under the run's `videos/one_leg_phases_<timestamp>/`, beside its
`checkpoints/`), with one_leg's task phases tracked (Robonuke_Furniture_Bench `furniture.track_phases`)
and drawn by the `one_leg` overlay (`my_project/overlays.py`: a box per phase reached with its tests as
measurement/goal, frozen once passed, the tracking error, a force bar, and the phase timeline with every
phase's start marked from the first frame), the wrench estimate in the per-step parquet and the policy's
chunk as spheres.

**ResiP's own checkpoint.** `scripts/eval_resip_checkpoint.py` (`my_project/bc/resip_checkpoint.py`)
rolls out ResiP's released one_leg BC weights (`s3://iai-robust-rearrangement/checkpoints/bc/one_leg/low/actor_chkpt.pt`)
in this port's ResiP task, acted as their `evaluate_model.py` acts it (4 DDIM steps, warm start
from timestep 50): each step the 63-wide observation is rebuilt in their 58-wide layout (robot
state with the EE z in their 15 mm lower base frame; the 5 base-frame parts and the front
obstacle -- its x, y with its fixed z, and THEIR body's rotation (their sim builds it rotated 90 deg
about z; ours is an unrotated box of the same shape; the checkpoint's own constant training value
when its normalizer shows one) -- as `[pos, quat xyzw]` in the tag frame, w >= 0 as their sim writes
it), and the action's z moved back.
Near their ~54% means the port is faithful; far below means the port caps BC. `scripts/check_resip_checkpoint.py` (`my_project/bc/resip_offline_check.py`, no sim) checks the
loading itself: the checkpoint run on its own training zarr, in their layout and frames, against the recorded actions. The `--config`
gives the task, envs, episode length and `eval.num_rollouts` only. `--save_rollouts <file.parquet>`
writes every rollout step (the port observation, the action sent, rewards, metrics; `run_eval`'s state
table) for offline analysis, e.g. the failure-stage sort.
`scripts/eval_resip_from_demos.py` (`my_project/bc/resip_demo_starts.py`) rolls the same checkpoint out from
the initial states of ResiP's own 50 demos (`--zarr`, `--trials` rollouts each: the arm IK-driven to
the demo's first EE pose, fingers, obstacles and parts written as recorded), so the starts are
exactly the policy's training ones; success is the soft assembled latch, a part leaving the limits a
failure; `--save_rollouts` writes the same state table.

**ResiP's own demos.** ResiP publishes its 50 one_leg low teleop demos only processed, as a
robust-rearrangement `.zarr`; `scripts/import_resip_zarr.py` (`my_project/demos/resip_zarr.py`,
needs `pip install zarr`) writes them in the same absolute rot6d columns (their state has our
terms, order and rot6d convention; the parts' quaternions become rot6d; `action/pos` is our
`action.abs`). There are no joint positions in the zarr, so they cannot be replayed from their
start state:

```bash
aws s3 sync --no-sign-request s3://iai-robust-rearrangement/data/processed/diffik/sim/one_leg/teleop/low/ ~/resip_data/one_leg_teleop_low/
python scripts/import_resip_zarr.py --zarr ~/resip_data/one_leg_teleop_low/<name>.zarr --out data/furniture_one_leg_resip_teleop
```

## The forge-v1 nut-demo importer's fields

`scripts/import_forge_v1.py` (`my_project/demos/forge_v1.py`) replays the
[forge-v1](https://huggingface.co/datasets/angledusgar/forge-v1) NutThread demos in Forge's nut-thread env
and keeps the ones our success check accepts, in the recorder's format.

| field | type | default | meaning |
|---|---|---|---|
| `forge_v1.source_dir` | str | — | the downloaded forge-v1 dataset |
| `forge_v1.dataset_dir` | str | — | where the successful replays are written |
| `forge_v1.num_demos` | int | `50` | stop after this many successful replays |
| `forge_v1.contact_force` | float | `2.0` | force (N) marking a recorded episode's first contact |
| `forge_v1.success_hold_steps` | int | `5` | keep a replay once success has held this many steps |

## Residual RL on the diffusion policy

A SAC residual trains on top of a frozen BC policy with the package's own SAC loop (train, eval and
debug all build the same env; `configs/experiments/resip_replica/residual_resip_replica.yaml`):

```bash
python scripts/train.py --config configs/experiments/resip_replica/residual_resip_replica.yaml --headless
```

Every step the env (`my_project/residual.py`, applied through the package's
`register_env_wrapper` hook) runs the frozen base: each env keeps its own chunk (a new one every
`model.actor.action_horizon` steps or when its episode restarts, warm-started as in BC). The
agent observes `[normalize(obs), normalize(a_base)]`, the base's 24-d observation and this
step's base action, both with the **base policy's** min-max normalizer (28 dims; the critic
state gets `normalize(a_base)` appended), except that an observation column whose stored range is
below `MinMaxNormalizer.NARROW_RANGE` (1e-4) reads as `[mid - 1, mid + 1]`, ~0: a base fit before
that rule stretched float noise there over [-1, 1] (the furniture base: 7 columns, resting parts and
the obstacle, 7.5e-8 .. 9.3e-6), and an online value just outside it mapped far beyond. The base
itself still reads its stats as trained. A demo file records the rule (`meta.agent_obs_narrow_range`)
and one without it is refused. Its action `r` (4, in [-1, 1]) is sent as
`a_base + residual.action_scale * r`. `residual/position_mm` and `residual/yaw_deg` log the
correction's size per agent. The base's architecture comes from its BC run's resolved config,
its weights and normalizer from the checkpoint, and the env's widths are checked against it.
`residual.base_action_horizon` and `residual.base_inference_steps` override the base run's executed
horizon and DDIM steps (null: the run's own); `residual.normalize_observation: false` gives the agent
(and the critic) the raw observation instead of the base-normalized, widened one, and
`residual.observe_base_action: false` leaves `normalize(a_base)` out of both (Policy Decorator's
`actor_input: obs`). Both flags are part of a demo file's fingerprint.

**Chunk-level residual (Policy Decorator).** `residual.steps_per_chunk: k` > 1 makes the agent act
once every `k` env steps with a `(k x A)`-wide action, step-major (row `j` is sub-step `j`'s residual;
`A` is the per-step width: 10 on the absolute frame, 6 with `action_frame: ee`). The wrapper's `step`
runs the `k` env steps itself: sub-step `j` sends the base's own next action plus `scale * r_j`
(the base hands its chunk out one action per env step, as always), the `k` rewards are summed, the
dones ORed, and the observation after the last sub-step returned. The base re-plans only on a chunk
boundary: its executed `action_horizon` must be a multiple of `k` (checked when the base loads;
Policy Decorator's recipe sets `base_action_horizon: 4`, `base_inference_steps: 4`). So at the chunk
boundary the base's next `k` actions are known: the critic state always holds those `k` actions
normalized, step-major (`k x 10` on the absolute frame: the very rows the residual rows are added to;
for `k = 1` today's single `normalize(a_base)`), the agent's observation too with
`observe_base_action: true`, and `critic_action: full` gives the critics the chunk's executed actions,
`normalize(a_base_j) + gain * r_j` per row, concatenated step-major (`k x 10`, the agent's action
width). For an env whose episode ends inside the chunk, the rows past its end describe actions its
episode never took (the sub-steps below). Every env steps
together, so an env whose episode ends at sub-step `j < k - 1` cannot stop there: Isaac Lab resets
it, it runs the chunk's remaining sub-steps on the base alone (zero residual), those sub-steps'
rewards, dones and metrics are not counted for it, its base is made to re-plan at the next chunk
boundary (so the new episode's chunks line up again), and the agent's first observation of the new
episode is the one after sub-step `k - 1`. `residual/*` metrics are per-sub-step means; the env's
`episode_metrics_to_log` are taken from the sub-step each episode closed on; other `infos` entries
are the last sub-step's. The package counts agent steps: `trainer.total_timesteps` and an eval
episode's length are in agent steps (`k` sim steps each); `scripts/watch_residual_eval.py` runs
`ceil(steps / k)` agent steps per round, reads success / engagement / force after each one and
counts an episode's length in sim steps. The history (`history.*`, the outermost wrapper) runs on
every env step, so at a chunk boundary it holds the previous `history.length` env steps (a
one-env-step stride, the chunk's own sub-steps included). Limitations, each refused with an error:
demo transitions (`scripts/make_demo_transitions.py`: one transition per demo step; a chunk's would
need the demo's `k` consecutive residuals against the base's chunk, the summed reward and
chunk-aligned pieces, and Policy Decorator trains without demos), `hybrid.enabled`, and
`critic_action: full` with `action_frame: ee` (the EE frame turns between sub-steps, so the executed
offsets after the first are not determined by the chunk's observation; the absolute frame is fine).

**The critic state is the residual's own.** The wrapper always publishes a critic observation group
(it creates one for an env that has none, such as FurnitureBench): the env's critic state or the
base's normalized observation, then the base action(s) normalized (the chunk's `k` with
`steps_per_chunk: k`), then the EE offset (`accumulate`) and the history. It does not change with
`residual.normalize_observation` / `observe_base_action`, which shape only what the actor sees, and
the learner hands it to the critic-action transform, so the critics always see what was executed.

```yaml
residual:
  enabled: true
  steps_per_chunk: 4
  base_action_horizon: 4
  base_inference_steps: 4
  normalize_observation: false
  observe_base_action: false
  critic_action: full
```

**Watching a residual run learn.** `scripts/watch_residual_eval.py` runs beside training in
its own Isaac process, with the eval conditions it is given (`--eval_config`, required). It evaluates the
**base alone** first, at startup (step 0: a zero residual), then every new numbered checkpoint
as training writes it, each with the same seed (the same 100 starting conditions), the SAC policy
deterministic. It logs `eval/success_rate`, `eval/engaged_rate`, `eval/force_mean`,
`eval/force_max_mean` and histograms of per-episode max and mean contact force to its own wandb
run (`<run>_eval`, same group) at the checkpoint's step, and appends each result to
`<run dir>/eval_watch/results.jsonl`. Each round shows a tqdm bar on stderr: env steps out of the
episode's length (FurnitureBench: with the envs still running and the successes so far).

```bash
python scripts/watch_residual_eval.py --run <residual run dir> --eval_config <eval config>
# e.g. the resip_replica residual, 100 envs:
python scripts/watch_residual_eval.py \
  --run runs/hybrid_demo_furniture_residual/residual_resip_replica/residual_resip_replica_a0 \
  --eval_config configs/eval/residual_watch_furniture.yaml
```

On a FurnitureBench residual run each env's first episode counts, and it logs
`eval/success_rate`, `eval/out_of_bounds_rate`, `eval/length_mean` and `eval/length_hist`
(no force or engagement on that env).

**Deterministic vs sampled.** The watcher evaluates the SAC policy's **mean** action by
default. Training only ever values **sampled** actions (the entropy target keeps the policy's
spread wide), and the mean alone can be a much worse policy: 15–40 points lower on these runs.
`--stochastic` evaluates with sampled actions, as training acts, into
`<run>/eval_watch_stochastic/` and its own wandb run; `--once` evaluates the checkpoints on
disk and exits (a stopped run). `scripts/publish_stochastic_eval.py <run>` adds those results
to the run's deterministic eval wandb run as `eval_stochastic/*`, on its own x-axis
`eval_stochastic/step` (the checkpoint step), so both curves sit in one run; re-runnable.

```bash
python scripts/watch_residual_eval.py --run <residual run dir> --eval_config <eval config> --stochastic --once
python scripts/publish_stochastic_eval.py <residual run dir>
```

| field | type | default | what it does |
| --- | --- | --- | --- |
| `residual.enabled` | bool | `false` | train a residual on a frozen diffusion policy (needs `action.world_pose`; sac or ppo) |
| `residual.base_run` | str \| null | `null` | the base policy's BC run as a local directory (`checkpoints/` inside, resolved config there or one level up). When enabled, set this or `base_project` + `base_run_name`, not both |
| `residual.base_project` | str \| null | `null` | the base policy's BC run on wandb: its project. Must differ from this run's `wandb.project` (residual runs log to their own project) |
| `residual.base_run_name` | str \| null | `null` | ... and its run name (or run id). The run's `resolved_config.yaml` and `base_checkpoint` are downloaded fresh whenever an env is built; the `[residual] base ...` log line names the checkpoint's step |
| `residual.base_entity` | str \| null | `null` | ... under this wandb entity; null: this run's `wandb.entity` |
| `residual.base_checkpoint` | str | `actor_chkpt_best_success_rate.pt` | the base's checkpoint file (in `<base_run>/checkpoints/`, or among the wandb run's files) |
| `residual.action_scale` | list[float] \| null | `null` | largest correction per step, `[x, y, z, yaw]` in m and rad; `null`: `residual.action_scale_fraction` of the base's normalized action range per dim |
| `residual.action_scale_fraction` | float | `0.1` | with `action_scale: null`, the correction bound as a fraction of the base's normalized action range: `fraction * (max - min) / 2` of its training actions per dim (robust-rearrangement's `action_scale: 0.1`; Policy Decorator's alpha, 0.1–0.3 typical) |
| `residual.demo_actions_beyond_scale` | str | `store` | demo transitions (`scripts/make_demo_transitions.py`) whose residual action `(a_demo - a_base) / scale` is beyond the policy's [-1, 1] in any dim: `store` them exactly (the critic takes the action as an input) or `drop` them. Never clipped. Part of the demo file's fingerprint |
| `residual.critic_action` | str | `residual` | what SAC's critics take as the action: `residual`, the agent's `r` in [-1, 1] (the base action reaches them through their critic state), or `full`, the executed pose action normalized as the base's, `normalize(a_base) + scale / half_range * r` = `normalize(a_base + scale * r)` (`my_project/residual.py` `residual_critic_action`, registered in `scripts/train.py`'s `setup()`), built from the critic state, never from the agent's observation. With `steps_per_chunk: k`, the chunk's `k` executed actions, `normalize(a_base_j) + gain * r_j` per row, step-major (`k x A`; not with `action_frame: ee`). A demo transition's is then exactly `normalize(a_demo)`, however far beyond the residual's reach. The actor still outputs `r`. SAC only |
| `residual.action_frame` | str | `absolute` | what the agent's action is (`my_project/residual_ee.py`). `absolute`: `scale * r` added to every dim of the base action (the fields above). `ee` (furniture FBAbs tasks; no `hybrid`, no `residual.action_scale`): `r = [r_pos 3, r_rot 3]` (6; the gripper is the base's), a correction of the base's target in the end-effector's frame -- `ee_pos_scale * R_ee r_pos` along the EE's axes (`R_ee` the observation's), a rotation `Exp(ee_rot_scale_deg * r_rot)` about the target's own axes -- re-sent every step. The env still gets the 10-d absolute target; the agent is announced a 6-d action in [-1, 1]. `critic_action: full` gives the critics the executed correction -- the offset after the step, in the EE frame over its reach, 6 like `r`; with `normalize(a_base)` in their input it fixes the target sent -- rebuilt from the observation with the same map, and `scripts/make_demo_transitions.py` stores the `r` that reproduces each demo action `target` (furniture FBAbs tasks, `my_project/residual_target.py`): Policy Decorator's meaning of r (ManiSkill2 `pd_ee_delta_pose` with `res_scale` 0.1), `r = [r_pos 3, r_rot 3, r_grip 1]` per step: `p_base + R_base (ee_pos_scale r_pos)`, `R_base Exp(ee_rot_scale r_rot)` (the base target's own axes), the gripper `+ scale_grip r_grip` (the absolute frame's gripper scale); `critic_action: full` rebuilds the chunk's executed targets (`k x 10` wide: the critics are built at that width) |
| `residual.ee_pos_scale` | float | `0.01` | m per unit `r_pos` per step (`action_frame: ee` / `target`; Policy Decorator: 0.01) |
| `residual.ee_rot_scale_deg` | float | `5.0` | deg per unit `r_rot` per step (`action_frame: ee` / `target`; Policy Decorator: 0.01 rad = 0.573) |
| `residual.accumulate` | bool | `false` | `action_frame: ee`: each step's correction adds to an offset that goes back to zero whenever the base starts a new chunk (every `model.actor.action_horizon` steps, and at an episode's start), instead of being re-sent every step; the offset before the step (`R_ee^T o` and `log(O)`, each over its largest reach in one chunk) is appended to the agent's observation and the critic's state after `normalize(a_base)` |
| `residual.steps_per_chunk` | int | `1` | env steps per agent action (Policy Decorator's residual chunk). `1`: the agent acts every env step. `k > 1`: once every `k` env steps with a `(k x A)`-wide action in [-1, 1], step-major; the wrapper runs the `k` env steps (each the base's own next action plus its row), sums the rewards, ORs the dones, returns the last observation; `residual/*` metrics are per-sub-step means. The base's executed `action_horizon` must be a multiple of `k`, so the observed base action is the chunk's `k` actions (step-major) and `critic_action: full` is the chunk's `k` executed actions. No `hybrid`, no demo transitions. See "Chunk-level residual" above |
| `residual.base_action_horizon` | int \| null | `null` | override the base's executed `model.actor.action_horizon` (Policy Decorator: 4); `null`: the base run's own. At most the base's `horizon`; with `steps_per_chunk` a multiple of it. Also applied by `scripts/make_demo_transitions.py` |
| `residual.base_inference_steps` | int \| null | `null` | override the base's DDIM `model.actor.inference_steps` (Policy Decorator: 4); `null`: the base run's own. At most the base's `num_diffusion_iters` |
| `residual.normalize_observation` | bool | `true` | the **agent** gets the base's observation normalized with the base's normalizer, its near-constant columns widened; `false`: the raw observation as the env produces it (Policy Decorator's). The base action part, when observed, stays `normalize(a_base)`. The critic state is not affected (`residual.critic_state`: always normalized). Part of the demo file's fingerprint |
| `residual.observe_base_action` | bool | `true` | append `normalize(a_base)` (with `steps_per_chunk: k`, the chunk's `k` base actions normalized, step-major; and, with `accumulate`, the offset) to the **agent's** observation; `false`: the agent sees the observation alone (Policy Decorator's `actor_input: obs`). The critic state always holds the base action(s), so `critic_action: full` works either way. Part of the demo file's fingerprint |
| `residual.critic_state` | str | `residual` | what the SAC critics observe. `residual`: the residual's critic state (`critic_state`: the env's, or the normalized base observation, then the base action(s) normalized). `observation`: no critic state, the critics see the agent's own observation (Policy Decorator's critic); `critic_action: full` rebuilds the executed action from the base action(s) the wrapper publishes every agent step as `infos["residual_base"]` / `["next_residual_base"]`, stored per transition as the transform's `side_inputs` and never a critic input. Furniture envs only; not with `action_frame: ee` or `sac.demo_transitions`. Eval and debug register the critic-action builder too, so the critics load at their width |
| `residual.ee_relative_actions` | bool | `false` | Policy Decorator's command form (`my_project/residual_target.py` `EERelative`) for the base action the agent observes (`observe_base_action`) and the executed action the critics take (`critic_action: full`): each of the chunk's absolute targets against the EE pose of the chunk's observation, in the EE's frame, `[R_ee^T (p - p_ee) / ee_relative_pos_unit, log(R_ee^T R) / ee_relative_rot_unit, normalize(gripper)]`, 7 per step (ManiSkill2 `pd_ee_delta_pose`, `frame: ee`), so both are the same `k x 7` form. The critic state stays the observation without the base action (a critic group is created for it). Needs `action_frame: target` and `critic_state: observation` |
| `residual.ee_relative_pos_unit` | float | `0.1` | m per unit of the EE-relative position (ManiSkill2 `pd_ee_delta_pose`: 0.1) |
| `residual.ee_relative_rot_unit` | float | `0.1` | rad per unit of the EE-relative rotation vector (ManiSkill2 `pd_ee_delta_pose`: 0.1) |
| `residual.active_from_phase` | str \| null | `null` | one_leg's task phase (`furniture.track_phases`: `corner`, `grasp`, `insert`, `screw`) from which the residual acts: an agent step whose observation is in an earlier phase (or at an episode-end code) runs the base alone, the agent's action zeroed, train and eval alike; `infos["residual_active"]` (1 / 0 per env) reports it (`sac.executed_action_mask: residual_active` stores the executed zeros; `sac.store_labels` keeps only the phases it acts in out of the memory's other phases). `insert`: only once the leg is grasped. Null: every step |

**The critic is asymmetric.** Forge gives the env a critic state (`task.cfg.state_order`: true
peg and hole poses, joint positions, gains, ...), so SAC's critics train on that 65-d state
while the actor sees the 24-d observation. That is privileged sim information; a real-robot run
needs a critic built from what the robot can measure.

### Demo transitions

**Demo transitions** (`my_project/residual_demos.py`) are built **offline from the recorded
frames**: no env, no replay, so the observations are exactly the ones the base was trained on.
For each demo, transitions `t -> t+1` over its frames (the last frame's next state was never
recorded):

* observations: `residual_observation` of the recorded policy observation and the base's own
  action (the frozen base run on the recorded observations, its chunk clock starting at the
  demo's first frame, `residual.base_action_horizon` / `base_inference_steps` applied, the
  `normalize_observation` / `observe_base_action` flags honoured), plus the `HistoryBuffer` when
  `history.enabled` — the same code the env wrappers run; critic states: the residual wrapper's
  critic state (`residual.critic_state`): the recorded critic state or, for an env without a critic
  group (FurnitureBench), the base's observation normalized (widened), then the normalized base
  action, the EE offset and the history, whatever the agent observes (at the flags' defaults that
  equals `observations` column for column). The file records the layout (`meta.critic_state_format`
  3) and an older file is refused at env build;
* actions: `(a_demo - a_base) / scale`, exactly — never clipped. Where it is beyond the squashed
  policy's [-1, 1] (the demo more than one scale from the base), `residual.demo_actions_beyond_scale`
  stores the transition as it is (default) or drops it; the script prints how often, per dim;
* rewards: Forge's peg-insert reward of each step, recomputed from the recorded true state
  after it (`forge_rewards`: Factory's own reward and success methods on a stand-in, Forge's
  terms line for line). The asset action penalty (scale 0.001) uses the next frame's
  fingertip where Forge uses its last physics substep's; the success-prediction term is 0,
  which `delay_until_ratio > 1` guarantees (refused otherwise);
* terminated / truncated: false (a demo ends inside the episode limit, which is checked).

`--check_in_sim N` additionally replays the session's first `N` demos through the
experiment's training env and compares the env's reward with the offline one, per step and per
term. Measured on the search demos: at most 1.4e-5 per step, all of it the approximated asset
penalty. **A restored start replays exactly only when the reset before it recreates the
recorded one** (`restore` cannot reset PhysX's internal state): the session's seed, plain
resets, recording order, and the RNG drawn as the recorder drew it. `scripts/replay_demos.py`
is exact from the session start in order, but one demo replayed alone drifts (episode 2:
26 mm, and it fails). The check builds its env first for that and skips any demo whose peg
leaves the recording; with the residual's base drawing from the same RNG, the first demo is
the exact one. The file is written to `sac.demo_transitions` with a fingerprint
(task cfg up to cosmetic fields, `action`, `residual`, `history`, `hybrid`); building the env for a run
whose fingerprint differs raises, naming the differences, and so does a file made before the agent's
observation widened the base's near-constant columns (`meta.agent_obs_narrow_range`, below) or, without
a critic group, before its states were the agent's observations (`meta.critic_state_format`). A chunked
residual (`residual.steps_per_chunk` > 1) has no demo transitions: the conversion refuses it. Only training is checked: a config
rebuilt from a run (eval, debug, the watcher) never reads the file, and its `task.cfg` is the
live env's, holding the action terms the env derived at `gym.make`. A new observation layout or base is
a re-conversion, not new demos:

```bash
python scripts/make_demo_transitions.py --config configs/experiments/resip_replica/residual_resip_replica.yaml
```

`--dataset` overrides the demos (default: the base policy's `bc.dataset`), `--out` the file.
A recording session whose task config differs from the experiment's beyond the episode length,
seed, render interval and viewer is refused, as is a demo longer than the experiment's episode.
Two per-episode randomization ranges may be recorded narrower than the experiment's (every
value no larger): `obs_rand.fixed_asset_pos` and `task.held_asset_pos_noise`. Each recorded
episode is then one the experiment could also draw; the nut demos have the bolt estimate exact
and the nut centred, while their residual run randomizes both.

| field | type | default | what it does |
| --- | --- | --- | --- |
| `history.enabled` | bool | `false` | append the agent's observation history (`my_project/wrappers.py`, `ObservationHistory`), outermost, after the residual: the base's input never changes |
| `history.length` | int | `8` | previous steps kept per term (15 Hz: ~0.5 s), oldest first; at an episode's first step the current value repeated |
| `history.terms` | list[str] | `[fingertip_pos, ft_force]` | `task.cfg.obs_order` terms to keep, read at their positions in the policy observation (Forge's `fingertip_pos` is the noisy estimate, `ft_force` the noisy 3-d force: Forge observes no torque). Appended to the critic state too, when the env has one |

### Ending episodes on success

`episode.reset_on_success` (`my_project/wrappers.py`, `ResetOnSuccess`) ends an env's episode
the step it first meets the task's own success check, reported like the time limit
(terminated and truncated), so SAC bootstraps through it: the objective is unchanged, the rest
of a solved episode is just not simulated. The step's reward and success are counted before the
reset. Applied to every env the package builds (train, eval, debug), never to the demo
recorder, replay or BC rollouts.

| field | type | default | what it does |
| --- | --- | --- | --- |
| `episode.reset_on_success` | bool | `false` | reset an env the step it succeeds (Factory's `_get_curr_successes` at `success_threshold`, with the wrist rotation for nut threading), counted as a time-out. One env only (`task.cfg.scene.num_envs: 1`): Factory's reset assumes every env resets together |

### Hybrid force/position control under the world pose

`hybrid.enabled` (`my_project/cfg.py`, `HybridCfg`; `HybridController` in
`my_project/wrappers.py`) attaches the package's controller (`ForgeControllerWrapper`,
`tau = J^T [(I - S)(K e - D v) + S K_f (f_d - f)]`) innermost, and the world pose and the
residual pass its blocks through untouched. The package's `controller` section cannot do this
(it wants the selection right after Forge's 7-wide action) and must stay off. The agent's
action is `[x, y, z, yaw | selection | force target]`: one selection bit per `force_axes` axis
(`> 0` is force control) and one force target (`[-1, 1]` times `force_target_bounds`). Gains
are constant: Forge's own pose gains, `default_force_gains` for force. A reset env goes back to
position control with a zero force target (Factory's reset closes the gripper through the
controller). With the residual on, the base gives only the pose; demo transitions
(`scripts/make_demo_transitions.py`) choose position on every axis with a zero force target,
as the demos ran. Applied to every env the package builds, never to the recorder, replay or BC
rollouts.

| field | type | default | what it does |
| --- | --- | --- | --- |
| `hybrid.enabled` | bool | `false` | Forge: attach the controller under the world pose; needs `action.world_pose`, `controller.enabled: false`, and `model.actor.bernoulli_action_dims` exactly the selection block (`[4, 5, 6]` for 3 axes). Furniture: the agent's action carries the blocks of the hybrid task id, which must be `Isaac-Furniture-OneLeg-FBAbs-ResiP-Hybrid-v0` (and that id needs it on); the selection block is `[10, 11, 12]` |
| `hybrid.force_axes` | list[int] | `[1, 1, 1, 0, 0, 0]` | Forge only: `[x, y, z, Rx, Ry, Rz]` axes the force branch may take (the package's `controller.force_axes`); the selection and force blocks are as wide as its sum. Furniture: `task.cfg.hybrid.force_axes` |
| `hybrid.force_target_bounds` | list[float] | `[50, 50, 50, 5, 5, 5]` | Forge only: the force action in [-1, 1] scales to +/- these, N and Nm. Furniture: `task.cfg.hybrid.*` |
| `hybrid.default_force_gains` | list[float] | `[0.1, 0.1, 0.1, 0.01, 0.01, 0.01]` | Forge only: K_f per axis. Furniture: `task.cfg.hybrid.*` |

**On the furniture family** the env owns the controller: the hybrid task id runs the same law
on joint torques (Robonuke_Furniture_Bench `mdp/hybrid.py`, its README's "Hybrid force/position
control on joint torques": the fields, the gains tuned to track like ResiP's IK + joint PD, and
what the force branch does). `hybrid.enabled` then only says the agent's action is
`[abs target 10 | selection 3 | force target 3]`; the base still gives the 10-wide abs target and
the residual passes the blocks on as the agent gave them. The project section's force fields
stay at their defaults there (the config refuses them and names `task.cfg.hybrid.*`).
On the furniture family the force branch is `f_d + K_f (f_d - f)` (`task.cfg.hybrid.force_feedforward`,
on by default), so a still EE settles at the target. **Forge's law has no such term and is weak**:
pure P, a force axis settles at `K_f f_d / (1 + K_f)` (~9 % of the target at the default K_f 0.1).

### The measured wrench in the agent's observation

`force_obs` (`my_project/wrappers.py`, `ForceObservation`) appends the end-effector wrench to
the RL agent's observation and the critic state, **last** (after the history), so the base's
input never changes. The furniture family reads the env's joint-torque estimate
(`env.wrench.filtered`: the external wrench on the EE, robot base frame); Forge its smoothed
wrist sensor (`force_sensor_smooth`). Raw N / N m: the learner's normalizer sees it like any input.

| field | type | default | what it does |
| --- | --- | --- | --- |
| `force_obs.enabled` | bool | `false` | append the wrench; required by the `force_selection` loss |
| `force_obs.mode` | str | `no_torque` | `no_torque`: force, 3 dims; `with_torque`: force then torque, 6 dims. The `force_selection` term's `mode` must match |

### The force-selection loss

`force_selection` (`my_project/losses.py`, policy target, graph-safe): per selection axis, binary
cross entropy between the policy's probability of **force** control and
`sigmoid(scale * (|f_i| - offset))`, `f` the force the agent observed (the stored observation's
`force_obs` columns): the harder the arm presses on an axis, the more that axis should be
force-controlled. Needs a hybrid action space and `force_obs.enabled`. With
`losses.terms[i].graph_safe: true` it runs inside SAC's captured gradient step
(`tests/residual/GPU/test_force_selection_graph.py`).

| kwarg | type | default | what it does |
| --- | --- | --- | --- |
| `scale` | float | required | 1/N: how sharply the target turns from position to force; > 0 |
| `offset` | float | required | N: the force at which the target is 0.5 |
| `mode` | str | `no_torque` | `force_obs.mode`: where the force sits in the observation's last columns |
| `axes` | list[int] | `[0, 1, 2]` | the force components (x, y, z) in selection order, one per selection dim |

## Diffusion-policy behaviour cloning

`scripts/train_bc.py` behaviour-clones a diffusion policy from demonstrations: a port of
[robust-rearrangement's](https://github.com/ankile/robust-rearrangement) state-based
`src/train/bc.py` (Diffusion Policy, Chi et al. 2023) onto this project's config, logging and
eval. One policy, no RL: `trainer.learner: diffusion_bc` (registered in `setup()`),
`experiment.num_agents: 1`. Pure pose control, no hybrid controller.

```bash
python scripts/train_bc.py --config configs/experiments/resip_replica/bc_resip_replica.yaml --headless
python scripts/train_bc.py --config configs/experiments/resip_replica/bc_resip_replica.yaml --headless bc.num_demos=25
```

**Data format** (any preprocessing must keep it): a dataset directory as
`scripts/record_demos.py` writes it (LeRobot-style parquet), whose `data/**/*.parquet` rows
carry `episode_index`, `frame_index`, one flat float observation vector per frame
(`bc.obs_key`) and one flat float action vector (`bc.action_key`). Widths come from the data.
The default keys train on the env's own policy observation and the action the recorder sent,
so rollouts see exactly the observations the policy was trained on.

**Metrics** go to one wandb run per training, published once per epoch at the gradient-step
count: `epoch`, `epoch_loss`, `train_bc_loss`, `train_grad_norm` (epoch mean), `actor_lr`;
on eval epochs `test_epoch_loss`, `test_bc_loss`, `early_stopper/{counter,best_loss,ema_loss}`;
on sample epochs `action_sample/{train,val}_action_mse_error[_<group>]`; on rollout epochs
`success_rate`, `best_success_rate`, `epoch_mean_return`, `n_success`, `n_rollouts` and the
package eval summary under `rollout/`. The run summary holds the dataset sizes and the exact
train / test episode indices. **Checkpoints** (`<run dir>/checkpoints/`, mirrored to the
run's files): `actor_chkpt_last.pt` every epoch, `actor_chkpt_best_test_loss.pt`,
`actor_chkpt_best_success_rate.pt`, and `actor_chkpt_<epoch>.pt` every
`trainer.checkpoint_interval` gradient steps. Each holds the policy (network + normalizer
stats), optimizer, LR scheduler, epoch, step and best scores.

Package fields a BC run reads with its own meaning: `trainer.total_timesteps` is total
**gradient steps** and `trainer.checkpoint_interval` the gradient steps between numbered
checkpoints (both multiples of `bc.steps_per_epoch`); `eval.num_rollouts` and
`task.cfg.scene.num_envs` size the rollouts; `trainer.write_interval` is unused.

### Fields for `model.architecture: diffusion_transformer`

The policy (`my_project/models/diffusion_policy.py`) around the noise-prediction transformer
(`my_project/models/transformer.py`, a near copy of robust-rearrangement's). Everything both
the policy and the trainer read lives here. Defaults are the reference's `diffusion.yaml`,
`diffusion_model/transformer.yaml` and `base.yaml`. A plain single-agent module, not for the
package's SAC / PPO pipeline (no `VmapEnsemble`). There is no `model.critic` group.

| field | type | default | what it does |
| --- | --- | --- | --- |
| `model.actor.horizon` | int | `32` | action steps predicted per chunk (`pred_horizon`) |
| `model.actor.n_obs_steps` | int | `1` | observation steps the prediction is conditioned on (`obs_horizon`) |
| `model.actor.action_horizon` | int | `8` | predicted actions executed before predicting again; `<= horizon` |
| `model.actor.n_layer` | int | `8` | transformer decoder layers |
| `model.actor.n_head` | int | `4` | attention heads; must divide `n_emb` |
| `model.actor.n_emb` | int | `256` | token embedding width; even and >= 4 |
| `model.actor.p_drop_emb` | float | `0.0` | dropout on the token embeddings, in [0, 1) |
| `model.actor.p_drop_attn` | float | `0.3` | dropout inside the attention / feed-forward blocks, in [0, 1) |
| `model.actor.causal_attn` | bool | `true` | action tokens attend only to earlier ones, and to observations up to their own step |
| `model.actor.time_as_cond` | bool | `true` | the diffusion timestep is a conditioning token; must be true for a policy |
| `model.actor.n_cond_layers` | int | `0` | `> 0`: transformer encoder over the conditioning tokens; `0`: an MLP |
| `model.actor.num_diffusion_iters` | int | `100` | forward-process steps (DDPM in training, DDIM's reference grid) |
| `model.actor.beta_schedule` | str | `squaredcos_cap_v2` | `linear`, `scaled_linear` or `squaredcos_cap_v2` |
| `model.actor.prediction_type` | str | `epsilon` | what the network predicts: `epsilon`, `sample` or `v_prediction` |
| `model.actor.clip_sample` | bool | `true` | clip each denoised sample to [-1, 1] |
| `model.actor.inference_steps` | int | `16` | DDIM denoising steps when acting; `<= num_diffusion_iters` |
| `model.actor.warm_start` | bool | `true` | start each chunk from the previous chunk's unexecuted tail, noised to `warm_start_timestep` (the reference); `false`: pure noise |
| `model.actor.warm_start_timestep` | int | `50` | noise level the warm start re-noises to; in [0, `num_diffusion_iters`) |
| `model.actor.obs_clip` | float | `3.0` | normalized observations are clipped to ±`obs_clip` when acting (never in training) |
| `model.actor.action_frame` | str | `absolute` | what the network predicts. `absolute`: the dataset's / env's action as it is. `ee_at_obs` (FBAbs tasks on `action.abs`; `my_project/models/action_frame.py`): every step of the chunk as `[dpos 3 (m), drot 3 (axis-angle, rad), gripper]` in the frame of the EE pose the observation reports (its `ee_pos` / `ee_rot6d`), unscaled -- all the chunk's steps share that one pose, so zeros are that pose. The labels are built per training window from `action.abs`; when acting the chunk becomes the env's absolute targets with the observation's pose, and the warm start's unexecuted tail is re-expressed at the next observation. The network and its action normalizer work on the 7-d action (the BC metric groups default to `pos` / `rot` / `gripper` on it); not with `bc.normalization: per_unit`; a residual refuses such a base |

### Fields for `model.architecture: diffusion_unet`

The same policy around the 1-D temporal U-Net (`my_project/models/unet.py`, a port of
robust-rearrangement's / Diffusion Policy's `ConditionalUnet1D`), the network ResiP's state
results come from (paper Table IV). The horizons, the diffusion process, inference and warm
start are the shared fields above (`horizon` ... `action_horizon`, `num_diffusion_iters` ...
`obs_clip`, same defaults); the network's own fields:

| field | type | default | what it does |
| --- | --- | --- | --- |
| `model.actor.down_dims` | list[int] | `[256, 512, 1024]` | channels per level; each level but the last halves the sequence, so `horizon` must divide by `2 ** (len - 1)` |
| `model.actor.diffusion_step_embed_dim` | int | `256` | width of the timestep embedding (concatenated with the flattened observations for FiLM); even and >= 4 |
| `model.actor.kernel_size` | int | `5` | Conv1d kernel; odd |
| `model.actor.n_groups` | int | `8` | GroupNorm groups; must divide every `down_dims` entry |

The BC trainer, eval, the residual's base loader and `scripts/analyze_bc_chunks.py` take either
architecture (`my_project.models.cfg.DIFFUSION_ARCHITECTURES`, registered by `register_all()`
in every script's `setup()`).

### Fields of the `bc` section

Only what the training loop alone uses (`my_project/cfg.py:BCCfg`); its rules apply only
when `trainer.learner` is `diffusion_bc`.

| field | type | default | what it does |
| --- | --- | --- | --- |
| `bc.dataset` | str \| null | `null` | the demo dataset directory (`record.dataset_dir`); required for a BC run |
| `bc.obs_key` | str | `observation.state` | parquet column with the per-frame observation vector |
| `bc.action_key` | str | `action` | parquet column with the per-frame action vector |
| `bc.num_demos` | int \| null | `50` | training episodes, drawn at random (seeded by `experiment.seed`); `null`: every episode |
| `bc.test_split` | float | `0.05` | test episodes = ceil(`test_split` × `num_demos`), drawn from the episodes not trained on; with every episode in use, that many are held out of them |
| `bc.batch_size` | int | `256` | windows per gradient step |
| `bc.steps_per_epoch` | int | `1000` | gradient steps per epoch |
| `bc.lr` | float | `1.0e-4` | AdamW learning rate |
| `bc.weight_decay` | float | `1.0e-3` | AdamW weight decay, on every parameter (as the reference trains it) |
| `bc.lr_schedule` | str | `cosine` | a diffusers `get_scheduler` name (`cosine`, `constant_with_warmup`, ...) |
| `bc.warmup_steps` | int | `1000` | LR warmup, gradient steps |
| `bc.max_grad_norm` | float | `0.0` | gradient-norm clip; `0` disables (the norm is still logged) |
| `bc.eval_every` | int | `10` | epochs between test-loss passes (and the score checkpoints, rollouts, sampled MSE, early stopper) |
| `bc.sample_every` | int | `100` | epochs between sampled-action MSE metrics (on eval epochs) |
| `bc.rollout_every` | int | `10` | epochs between env rollouts (on eval epochs); `0`: no rollouts and no env is built |
| `bc.rollout_loss_threshold` | float | `0.1` | rollouts run only once the test loss is below this |
| `bc.rollout_progress` | bool | `false` | show each rollout's progress (a tqdm bar of episode steps out of `eval.num_rollouts` x the episode length; an episode that ends early adds its unused steps) and print its success rate when it ends |
| `bc.max_hours` | float | `null` | stop after this many hours of wall-clock training (checked after each epoch, after its checkpoints); `null`: no limit |
| `bc.save_only_best_ckpt` | bool | `false` | keep only `actor_chkpt_best_success_rate.pt`, on disk and in wandb: written at the first rollout, then overwritten only when a rollout beats the best success rate (no last, periodic or best-test-loss checkpoints); needs `bc.rollout_every` > 0 |
| `bc.normalization` | str | `per_dim` | how obs and actions are scaled to [-1, 1] from the demos' min / max: `per_dim` gives every dim its own range (robust-rearrangement's); `per_unit` gives every dim of one physical unit one shared range, taken over the observation and the action together -- positions (EE, parts, obstacle, the action's target), rot6d, linear velocity, angular velocity, finger width, gripper command (`families.obs_units` / `action_units`) -- so near-constant dims stay near-constant; furniture FBAbs tasks only. Either way a dim (unit) whose range is below `MinMaxNormalizer.NARROW_RANGE` (1e-4: constant up to float noise) gets `[mid - 1, mid + 1]` and maps to ~0, never noise stretched over [-1, 1] |
| `bc.normalize_action` | bool | `true` | `false`: the action is used as it is, never rescaled -- its range is fixed at [-1, 1] (`MinMaxNormalizer.fit_identity`), so normalizing is the identity and a zero output is the zero action (the FBRel interface: hold the EE). Every demo action must lie in [-1, 1] (training raises naming the dims that leave it) unless `model.actor.clip_sample` is off, when it may take any value. Not with `bc.normalization: per_unit` |
| `bc.symmetric_action` | bool | `false` | with `bc.normalize_action`: each action dim scaled by `s = max(\|min\|, \|max\|)` of the demos' actions (`MinMaxNormalizer.fit_symmetric`: stats `[-s, s]`, normalizing is `x / s`) instead of min-max, so a zero action stays a zero output -- for `model.actor.action_frame: ee_at_obs`, the observation's pose -- while every dim still spans [-1, 1]. Not with `bc.normalize_action: false` or `bc.normalization: per_unit` |
| `bc.early_stop_patience` | float | `.inf` | evals without a new best EMA test loss before stopping; `.inf` never stops |
| `bc.early_stop_smooth` | float | `0.9` | EMA factor of the early stopper's test loss |
| `bc.show_waypoints` | bool | `false` | eval / debug: draw each env's predicted chunk as spheres in the scene (viewer and recorded videos): the waypoint executing now yellow, the rest of the executed `action_horizon` green while still to come and grey once done, the remaining tail (never executed) red |
| `bc.waypoint_radius` | float | `0.0015` | m: those spheres' radius; at the recorder's 240 px width one pixel is ~3.6 mm at 1 m, so recordings want ~0.005 (`configs/eval/one_leg_phases.yaml`) |
| `bc.action_metric_groups` | dict[str, list[int]] | `{pos: [0, 1, 2], rot: [3, 4, 5]}` | named action dims for `action_sample/*_action_mse_error_<name>` |

Changed from the reference on purpose: the train / test split is by episode (the reference
split windows, so one demo landed on both sides); `train_grad_norm` is the epoch mean (the
reference logged the last batch); the sampled-action metric samples from pure noise (the
reference's warm-started from an unrelated batch's leftovers); the training target follows
`prediction_type`. No EMA weights and no resume-from-run yet (both off / absent by default in
the reference's state experiments).

Tests: `pytest` (CPU), `pytest -m gpu` (the CUDA variants).
