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
  independent of the training method), chained onto `hpc.yaml` (the cluster's resources). Base one on it
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
`observation.next_state_abs`, `reward` (the env's sparse assembly reward), `skill`.
`meta/episodes.jsonl` carries each demo's success and source file; re-running appends only new
files. `--include_failures` also imports `fails/*.pkl` (success false). Imported demos carry no
recording session (no env snapshot), so a residual run's `--check_in_sim` replay is Forge-only.

**Training** (`configs/experiments/resip_replica/`): `bc_resip_replica.yaml` is ResiP's released
one_leg recipe (batch 1024, warmup 2000, per-dim scaling, 512 rollouts every 10 epochs whatever the
test loss, warm start, 16 DDIM steps) on `data/physic_aligned`; `residual_resip_replica.yaml` is
residual SAC on the frozen `physic_aligned_evalHappy_a0`, named by `residual.base_project` /
`base_run_name` and downloaded from wandb, logging to its own project
(`hybrid_demo_furniture_residual`). Its demo transitions read the importer's
`observation.state_abs` / `action.abs` / `observation.next_state_abs`, with the recorded reward
times the env's `step_dt` (Isaac Lab's reward manager scales every term by it).
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
state gets `normalize(a_base)` appended), and its action `r` (4, in [-1, 1]) is sent as
`a_base + residual.action_scale * r`. `residual/position_mm` and `residual/yaw_deg` log the
correction's size per agent. The base's architecture comes from its BC run's resolved config,
its weights and normalizer from the checkpoint, and the env's widths are checked against it.

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
  demo's first frame), plus the `HistoryBuffer` when `history.enabled` — the same code the
  env wrappers run; critic states: the recorded critic state with the same appendices;
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
whose fingerprint differs raises, naming the differences. Only training is checked: a config
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
| `bc.normalization` | str | `per_dim` | how obs and actions are scaled to [-1, 1] from the demos' min / max: `per_dim` gives every dim its own range (robust-rearrangement's); `per_unit` gives every dim of one physical unit one shared range, taken over the observation and the action together -- positions (EE, parts, obstacle, the action's target), rot6d, linear velocity, angular velocity, finger width, gripper command (`families.obs_units` / `action_units`) -- so near-constant dims stay near-constant; furniture FBAbs tasks only |
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
