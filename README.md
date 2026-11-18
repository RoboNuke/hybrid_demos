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
  recording: `one_leg_phases.yaml`; the residual watcher's 100 envs: `residual_watch_furniture.yaml`; the
  ablations' 5 agents x 200 envs, 200 episodes each in one round: `residual_watch_ablation.yaml`), chained onto `hpc.yaml` for the cluster launcher's resources.
* `configs/methods/` -- a method with no task: its residual, learner, memory and networks (`resfit.yaml`: ResFiT's
  code defaults). A foundation chains it after a task file with a list base, `base: [../../base/<task>.yaml,
  ../../methods/<method>.yaml]`, and sets only the task's own values, so one method runs on every task unchanged.
* `configs/experiments/<group>/` -- one folder per experiment group: `_foundation.yaml` (based on a
  `base/` task, and a `methods/` file when the group runs a published method) holds what every training of the group shares -- the seed, the wandb entity, every
  learning parameter -- and one launch config per training (`bc_<group>.yaml`, `residual_<group>.yaml`)
  sets its learner, model, wandb project and group, and env count. The leading `_` keeps
  `launch_train.py <folder>` from submitting the foundation.

## Run

```bash
# train the BC policy, its raw demo episodes, then a residual on it (configs/experiments/resfit_can/)
python scripts/train_bc.py --config configs/experiments/resfit_can/bc_resfit_can.yaml --headless
python scripts/make_raw_demos.py --config configs/experiments/resfit_can/residual_resfit_can.yaml --out data/resfit_can/raw_demos.pt
python scripts/train.py --config configs/experiments/resfit_can/residual_resfit_can.yaml --headless
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

The cluster side (the `general` env as Apptainer images, the `~/hpc-share/robonuke/` layout,
the clones of this repo, the package and Robonuke_Furniture_Bench, a per-partition smoke test)
is set up once by the package's `hpc/install_hpc.sh`, run on a login node; its README `## hpc`
section is the reference. The launchers then submit SLURM + Apptainer jobs from a login node —
a light python is enough (omegaconf + pyyaml; wandb only for `launch_eval`'s run query; the
installer makes a `rnk-launch` conda env with them). Resources come from the `hpc` config
section (`configs/base/hpc.yaml`, overridable per experiment or on the CLI); the field
reference is the package README. This project's `hpc.yaml` also names the Furniture Bench
clone in `hpc.extra_repos`: the job binds it and puts it on `PYTHONPATH`, since `setup()`
imports `furniture_lab.tasks` and the image carries no research code. Names derive from
wandb: `--project` and `--group_prefix` are required, group = `{group_prefix}_{config_stem}`
(sweeps append `_{label}-{value}`), the SLURM job name is the group, runs are `{group}_a{i}`.

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

# one watcher job per training run of the folder (scripts/watch_residual_eval.py over the run's group
# directory, every agent side by side, every checkpoint on disk, then exit; --follow keeps polling):
# the launch_train command with the verb swapped, the eval config supplying the job's resources
python launchers/launch_watch.py configs/experiments/scaled_ablation/ --project one_leg_scaled_ablations \
    --group_prefix abs --eval_config configs/eval/residual_watch_ablation.yaml
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
converted into Forge's 7-d action by `my_project/wrappers/env_wrappers.py` (roll / pitch 0, success
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

`configs/experiments/parts_pd/` is PARTS (Su et al., arXiv:2609.21788) on 1 env: one residual per phase
listed in `sac.route_labels` (`insert`, `screw`), each its own SAC with its own replay memory and optimizers,
storing the env's own rewards, the step into another phase terminal (the package's `RoutedSAC`), a fixed std (`model.actor.fixed_std`), no entropy term, and the demos
written into each phase's replay memory as ordinary rows (`sac.demo_storage: memory`). The residual acts every
env step (`residual.steps_per_chunk: 1`). Its demos are built per phase when the learner loads them (see the raw demo episodes below):

```bash
python scripts/train.py --config configs/experiments/parts_pd/residual_parts_pd.yaml --headless
```

**Raw demo episodes** (`scripts/make_raw_demos.py`, `my_project/raw_demos.py`, CPU only, no sim) hold only what is
general about the demos: per recorded episode, frames `0..success_frame` (the step after which the training env's
`success` termination first holds), the raw env observations, the absolute targets the env executed (DART's perturbed
commands), the frozen base's action at each frame (a cold plan every `action_horizon` frames, as the env runs it; the
base is stochastic, so its sample is stored), and the env's own per-step data from the source pkl (recorded by
Robonuke_Furniture_Bench's `scripts/collect_scripted.py`): the infos published with each frame's observation
(`infos`, flattened to `"<key>/<name>"` paths, e.g. the phase tag `eval_state/phase/id`), every reward term unweighted
(`reward_terms`) and every termination term of the training env (`termination_terms`). Its `meta.fingerprint` is the
base run and checkpoint, the base's planning settings and the task; a run with another refuses the file. Demos
collected before the collector recorded the env's data are refused: re-collect them. Everything the run's settings
shape is applied when the learner loads the file (`my_project/demo_loader.py`, the package's `register_demo_loader`),
one loader call per learner: a routed residual loads only the frames its phase tag selects, an unrouted learner every
frame. The loader copies the env's data and computes none of its own: each frame's inputs and `r` through the run's
action space (`my_project/action_spaces.py`, the env wrapper's own instance: `observe` on the raw observations,
`invert` on the executed actions), the rewards as the recorded terms times the run's `task.cfg.rewards.<term>.weight`,
terminal where a run's termination term (not a time-out) holds and, for a routed residual, on the step into another
phase (as `RoutedSAC` marks it online). So changing the observation, the action map, the reward weights or the
normalization never needs a rebuild. One transition per demo step (`steps_per_chunk: 1`, no hybrid dims, no
`phase_scales`, no env critic state of its own):

```bash
python scripts/make_raw_demos.py --config configs/experiments/scaled_residual/residual_scaled_residual_v3.yaml \
    --dataset data/physic_aligned_tcp_300 --out data/scaled_residual_v3/raw_demos.pt
```

`configs/experiments/ResFiT/` is ResFiT (Ankile et al. 2025, arXiv:2509.19301) on one_leg: the method file
`configs/methods/resfit.yaml` (its residual, learner, memory and networks, unchanged) chained after
`configs/base/fb_one_leg.yaml`, plus one_leg's own values. The actor is a deterministic residual, `0.1 tanh(mu)`, with
fixed noise. There are 10 critics: the target takes the min over a random 2, and the actor trains on their mean. The
learner uses TD3's target smoothing, 4 critic updates per env step with the actor on the 4th, a 10k-update critic warm-up
and uniform warm-up actions. Half of every batch comes from the demos, and the actor trains on that same batch. The
residual is ResFiT's arithmetic on our action (`residual.action_space: per_dim`): the base's absolute target plus
`1.1 x max(demo half range, 0.05) x r` per dim, rot6d components and gripper included, clamped to the widened range
(`executed_clip: 1.1`). The critics see the observation and the executed action. On top of the method, one_leg runs one
residual per phase (`ddpg.route_labels`: `insert`, `screw`), each counting its learning start and warm-up in its own
phase's steps (`ddpg.route_clock`), with the Coffee / CanSorting paper runs' discount, 5-step returns and noise 0.025;
the insert phase is passed after 5 held steps (`furniture.phase_hold_steps: 5`); the unused legs and the fixed obstacle
are dropped from the observation. The foundation's header lists where it is not ResFiT's. The robosuite groups
(`resfit_<task>/`) chain the same method file onto their tasks.

```bash
python scripts/train.py --config configs/experiments/ResFiT/residual_ResFiT.yaml --headless
```

`residual_parts_pd_v2.yaml` moves it closer to the paper: no-progress resets (leg 4 moving at most 0.5 mm / 1 deg per
step) in Insert (30 steps) and Screw (120) with
the stalled transitions held and only the first kept as the failure (`furniture.stall_steps_*`, `sac.hold_*`),
critics on `Q(s, r)` (`residual.critic_action: residual`), reference dropout on the actor's base-action columns,
updates and retraining on each residual's own phase steps (`sac.route_clock`), noise held at most 5 steps, and
retraining every 10k phase steps. `configs/eval/residual_watch_furniture.yaml` turns the no-progress reset off, so
evals run every episode to its end.

`residual_parts_pd_v4.yaml` is v3 with an absolute residual of fixed size (`residual.target_axes: base`, r = 1 is 2 cm
and 6 deg about the robot-base axes), the actor observing the base chunk in absolute form, critics on `Q(s, a_base, r)`
(`critic_state: residual`), the base's warm start off, no demos, and PARTS' terms averaged over the action dims
(`sac.parts_term_reduction: mean`).

Every residual run also publishes an episode-bookkeeping check: `episode/isaac_steps` (the env's own episode length,
Robonuke_Furniture_Bench), `episode/agent_steps` (the agent steps the residual wrapper gave that episode) and
`episode/count_offset` = isaac - `steps_per_chunk` x agent steps, which must stay within +-(`steps_per_chunk` - 1): an
episode that ends mid-chunk hands the rest of that chunk to the next one; anything larger is a bookkeeping bug.

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

## ResFiT's robosuite tasks (resfit_can, resfit_square, resfit_boxcleanup, resfit_cansort, resfit_coffee)

ResFiT's residual TD3 recipe on the tasks it is known to work on, run through this project's pipeline, to tell whether
the one_leg failures are ours or the task's. `my_project/tasks/robosuite_tasks.py` builds each task as ResFiT's code
does (robosuite 1.5.1, their robots, controllers and horizons), low-dim, behind an Isaac Lab Direct env's surface (the
package and the project wrappers take it unchanged):

| task id | robosuite env | robots | horizon | observation | action |
|---|---|---|---|---|---|
| `Robosuite-Can-LowDim-v0` | `PickPlaceCan` | Panda | 200 | robot 9, `object` 14 | OSC_POSE delta 6, gripper 1 |
| `Robosuite-Square-LowDim-v0` | `NutAssemblySquare` | Panda | 300 | robot 9, `object` 14 | OSC_POSE delta 6, gripper 1 |
| `Robosuite-BoxCleanup-LowDim-v0` | `TwoArmBoxCleanup` | PandaDexRH, PandaDexLH | 300 | robots 38, `object_joints` 14 | per arm OSC_POSE 6, hand 6 |
| `Robosuite-CanSort-LowDim-v0` | `TwoArmCanSortRandom` | GR1FixedLowerBody | 400 | hands 36, `object_joints` 21 | Mink whole-body IK, absolute: right pose 6, left pose 6, fingers 12 |
| `Robosuite-Coffee-LowDim-v0` | `TwoArmCoffee` | GR1FixedLowerBody | 400 | hands 36, `object_joints` 15 | Mink whole-body IK, absolute: right pose 6, left pose 6, fingers 12 |

The robot state is ResFiT's (its eef pose and gripper joints); `object` is robosuite's `object-state` (robomimic's
low-dim key), `object_joints` the position of every joint that is not the robot's (the DexMimicGen tasks publish no
object observable). Their base configs set `task.simulator: standalone`, so `train.py`, `train_bc.py`,
`make_raw_demos.py` and `watch_residual_eval.py` never start Isaac Sim (their `setup()` skips
`install_obs_terms`, Factory's terms). Needs robosuite and dexmimicgen at ResFiT's commits and mujoco 3.3.2:

    git clone https://github.com/ARISE-Initiative/robosuite ~/deps/robosuite
    git -C ~/deps/robosuite checkout 77a4751233c29456a5381209e30dd0dbf39a6557
    pip install --no-deps -e ~/deps/robosuite && pip install mujoco==3.3.2 mink==0.0.7 PyOpenGL glfw etils termcolor
    git clone https://github.com/NVlabs/dexmimicgen.git ~/deps/dexmimicgen
    git -C ~/deps/dexmimicgen checkout e606f36a38b1d4ba8f56d06d6c0cd059b20ebbaf
    pip install --no-deps -e ~/deps/dexmimicgen

`--no-deps` keeps robosuite's `opencv-python` off the env (Isaac Lab's `opencv-python-headless` provides `cv2`) and
dexmimicgen's pins (numpy 1.23, h5py 3.12) from downgrading it.

Demos: robomimic's low-dim files (Can, Square) import directly; DexMimicGen's generated files
(huggingface.co/datasets/MimicGen/dexmimicgen_datasets, `generated/<task>.hdf5`) import with `--replay`, which sets every
recorded MuJoCo state in the task's env and reads the observation and success there:

    python scripts/import_robomimic.py --hdf5 data/robomimic_can_mh/low_dim_v15.hdf5 --out data/robomimic_can_mh_300
    python scripts/import_robomimic.py --hdf5 data/robomimic_square_mh/low_dim_v15.hdf5 --out data/robomimic_square_mh_300 \
        --task_name Robosuite-Square-LowDim-v0
    python scripts/import_robomimic.py --hdf5 data/dexmimicgen/two_arm_coffee.hdf5 --out data/dexmg_coffee_1000 \
        --task_name Robosuite-Coffee-LowDim-v0 --replay --max_demos 1000

Then per task (`<t>` one of can, square, boxcleanup, cansort, coffee): the base, its raw demo episodes, the residual
(`residual_resfit_<t>.yaml`: 3 agents in one process) and one watcher per agent:

    python scripts/train_bc.py --config configs/experiments/resfit_<t>/bc_resfit_<t>.yaml --headless
    python scripts/make_raw_demos.py --config configs/experiments/resfit_<t>/residual_resfit_<t>.yaml --out data/resfit_<t>/raw_demos.pt
    python scripts/train.py --config configs/experiments/resfit_<t>/residual_resfit_<t>.yaml --headless
    python scripts/watch_residual_eval.py --headless --run runs/resfit_validation_residual/resfit_<t>/resfit_<t>_a<i> \
        --eval_config configs/eval/residual_watch_can.yaml

Each group's `_foundation.yaml` lists ResFiT's settings for the task (`resfit_can`'s also every place ours differs).
`scripts/make_raw_demos.py` writes each demo's frames up to its success (`meta/episodes.jsonl`'s `success_frame`) with
the base's own action at each frame; the residual's transitions are built from them at load (see "Residual RL on the
diffusion policy"). `configs/experiments/can_compare/` and `square_compare/` run three residuals on the same task: additive,
scaled (multiplied, the gripper the base's) and additive with adaptive exploration.

### Fields of `task.cfg` for the `Robosuite-*-LowDim-v0` tasks

| field | default (Can) | meaning |
|---|---|---|
| `seed` | 42 | NumPy's global seed at construction (robosuite 1.5.1's placement samplers draw from it) |
| `scene.num_envs` | 1 | robosuite envs, stepped one after another on the CPU |
| `env_name` | `PickPlaceCan` | the robosuite env (`robosuite.make`'s `env_name`) |
| `robots` | `[Panda]` | its robots (ResFiT's `ENV_ROBOTS`); the first one's default composite controller, a composite IK's input frame set to `world` |
| `import_module` | `""` | a module imported before making it (`dexmimicgen` registers the DexMimicGen envs) |
| `controller_config` | `""` | a robosuite composite controller JSON (absolute, or relative to the repo root) in place of the first robot's default (a composite IK's input frame still set to `world`); `my_project/tasks/controllers/gr1_arms_only_mink_ik.json` is the one DexMimicGen's CanSort demos were recorded with (robosuite has no `GR1ArmsOnly` default) |
| `horizon` | 200 | steps per episode (ResFiT's horizon for the task): truncated after it |
| `control_freq` | 20 | control steps per second |
| `lite_physics` | true | robosuite's `lite_physics` (ResFiT leaves robosuite's default) |
| `obs_keys` | `[robot0_eef_pos, robot0_eef_quat, robot0_gripper_qpos, object]` | the observation's terms in order (`object`, `object_joints`: above; `can_colour`: TwoArmCanSortRandom's can, 1 red / 0 blue, which bin counts as success) |
| `terminate_on_success` | true | end the episode on the step the task succeeds (ResFiT's `terminated = reward == 1`) |
| `observation_space` | 23 | the observation's width; the env checks it against `obs_keys` |
| `action_space` | 7 | the controller's input width; the env checks it against robosuite's |
| `state_space` | 0 | no critic state of its own |

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

A residual trains on top of a frozen BC policy with the package's own learners (ddpg, sac, ppo); train, eval, debug
and the checkpoint watcher all build the same env, through the residual wrappers (`my_project/wrappers`, applied by
`project_env_wrappers` through the package's `register_env_wrapper`, every section registered by
`my_project/wrappers/register.py`'s `register_residual()` in each script's `setup()`). They stack in one fixed order,
innermost first, each on or off by its own section:

| wrapper | section | what it does |
| --- | --- | --- |
| `BasePolicy` (`wrappers/residual_policy/base_policy.py`) | `base_policy` | runs the frozen base on the env's own observation (each env its own chunk clock: a new chunk every `action_horizon` steps or when its episode starts); keeps `base_obs`, `base_action`, `fresh`, `new_chunk` as fields every wrapper above reads; lays the agent's observation out as `[observation columns, base action]` and the critics' as the observation columns (an env's own critic group is kept); logs the sent action against the base's (`residual/position_mm`, `rot_deg` / `yaw_deg`, by the task's action layout) |
| `DropColumnsWrapper` (`wrappers/action_spaces/drop_columns.py`) | `drop_columns` | removes observation columns (from both observations) and action dims; a dropped action dim is the base's (it leaves the base action too) |
| `TargetTrackingWrapper` (`wrappers/action_spaces/target_tracking.py`) | `target_tracking` | sends the env an absolute target per tracked body (an arm's pose, a finger's joints); the agent's action is each target's change. A target starts at the body at an episode's start (and, `reset_at_chunk`, whenever the base re-plans); the base action above is the base's own planned change; the current targets, relative to their bodies, can join the agent's and the critics' observation |
| `EERelativeWrapper` (`wrappers/observation_spaces/ee_relative.py`) | `ee_relative` | re-expresses listed observation poses (and vectors) relative to the EE, in its axes, and removes the EE's columns |
| `AdditiveResidual` (`wrappers/residual_policy/additive.py`) | `additive_residual` | `sent = base + scale * r` on every dim of the action space below |
| `MultiplyResidual` (`wrappers/residual_policy/multiply.py`) | `multiply_residual` | `sent = r * base` on every dim of the action space below (r the multiplier, the base alone at 1) |

Every conversion (drop, target tracking, EE-relative) is plain torch around a `Conversion` (`wrappers/chain.py`): the
step's observation columns and base action in, its own out; its action back down to the wrapper below's; an executed
action up to its own (for demos). A column index in a wrapper's config is the observation as it reaches that wrapper.
The two residuals (`wrappers/residual_policy/combine.py`) also publish the critics' side input (the base action, and
each env's scale with phase scales) as `infos["residual_base"]` / `["next_residual_base"]`, split every
`residual/<size>` metric per task phase when the env publishes its phase (`residual/<size>_<phase>`), count each
episode's agent steps (`episode/agent_steps`, `episode/count_offset`), and check that a phase-scale label is the one
the env publishes. The package hooks they answer: the critics' action is the sent action (`base + scale * r` or `r *
base`, rebuilt from the side input); DDPG's adaptive noise factor is the base action in r units, `|base| / scale`
(additive) or 1 (multiply), read right after the agent's observation columns; the neutral action (a phase without a
residual, the eval's base round, `base_only_episodes`) is 0 or 1. The inputs go out raw: the learner's
`state_preprocessor` normalizes them (`demo`: fitted once on the demo rows).

Three setups, the one_leg pinned run, smol2 and a robosuite task:

```yaml
# ResFiT_phase_v7: per-dim additive with each routed phase's own scale, legs 2-4 and the obstacle dropped
base_policy: {enabled: true, base_run: runs/hybrid_demo_furniture/obs-fix_true-jacobian_50demos/obs-fix_true-jacobian_50demos_a0}
drop_columns: {enabled: true, observation: [[25, 52], [61, 63]]}
additive_residual: {enabled: true, phase_stats: data/ResFiT_phase_v7/phase_stats.pt}
# scaled_ablation (smol2): the EE's target tracked, its change multiplied, the observation relative to the EE
drop_columns: {enabled: true, observation: [[25, 52], [61, 63]], action: [9]}
target_tracking: {enabled: true, targets: [{action_pos: [0, 3], obs_pos: [0, 3], action_rot: [3, 9], obs_rot: [3, 9]}]}
ee_relative: {enabled: true, poses: [{pos: [16, 19], rot: [19, 25]}, {pos: [25, 28], rot: [28, 34]}], vectors: [[9, 12], [12, 15]]}
multiply_residual: {enabled: true}
# can_compare: the multiplier on the OSC delta's 6 pose dims, the gripper the base's
drop_columns: {enabled: true, action: [6]}
multiply_residual: {enabled: true}
```

**Demos are converted at load.** A residual run's `<learner>.demo_transitions` is a raw demo file
(`scripts/make_raw_demos.py`, `my_project/raw_demos.py`: per episode the recorded observations, the executed actions,
the base's own action at each frame, and for FurnitureBench the env's infos, reward and termination terms). When the
learner loads it, `my_project/wrappers/residual_policy/demos.py` runs every frame through the run's own wrappers --
the base's observation layout, every conversion with its state carried frame to frame, the residual's side input, the
executed action through every inverse and the residual's (`(executed - base) / scale`, unclamped, or `executed /
base`, 1 under `invert_floor`, clamped to the learner's range) -- and builds the transitions: the env's reward terms
times this run's weights and its termination terms (robomimic / DexMimicGen demos: reward 1 on the success frame), a
routed residual loading its own phase's steps with the phase switch terminal. Changing a wrapper needs no new file;
the file's fingerprint (the base run, checkpoint and planning settings, the task) must match the run's.

```bash
python scripts/make_raw_demos.py --config configs/experiments/can_compare/can_add.yaml --out data/resfit_can/raw_demos.pt
```

| field | type | default | what it does |
| --- | --- | --- | --- |
| `base_policy.enabled` | bool | `false` | run a residual on a frozen diffusion policy (every other residual wrapper needs it; exactly one of `additive_residual` / `multiply_residual` with it) |
| `base_policy.base_run` | str \| null | `null` | the base's BC run as a local directory; set this or `base_project` + `base_run_name`, not both |
| `base_policy.base_project` / `base_run_name` / `base_entity` | str \| null | `null` | the base's BC run on wandb (its project must differ from this run's `wandb.project`; the entity null: `wandb.entity`), downloaded with its checkpoint into the package's run cache |
| `base_policy.base_checkpoint` | str | `actor_chkpt_best_success_rate.pt` | the base's checkpoint file |
| `base_policy.base_action_horizon` | int \| null | `null` | override the base's executed horizon (at most its `horizon`); null: the run's own |
| `base_policy.base_inference_steps` | int \| null | `null` | override the base's DDIM steps (at most its `num_diffusion_iters`); null: the run's own |
| `base_policy.base_warm_start` | bool \| null | `null` | override the base's warm start (`false`: every chunk from pure noise, as the raw demos plan it); null: the run's own |
| `drop_columns.enabled` | bool | `false` | drop observation columns and action dims |
| `drop_columns.observation` | list of `[start, stop)` | `[]` | observation columns removed from the agent's and the critics' observation (the env's own columns) |
| `drop_columns.action` | list[int] | `[]` | action dims the agent no longer controls: the base's own value is sent there |
| `target_tracking.enabled` | bool | `false` | track an absolute target per body; the agent's action is each target's change |
| `target_tracking.targets` | list | one EE pose | per target `action_pos` / `obs_pos` (`[start, stop)`: an arm's position in m, or a finger's joints, as wide in both) and optionally `action_rot` / `obs_rot` with `action_rot_param` / `obs_rot_param` (`rotvec`, `euler`, `rot6d`, `quat_xyzw`, `quat_wxyz`). The action: per target its position change then its rotation change (axis-angle, 3, base frame's axes), then the dims no target covers, as they are |
| `target_tracking.reset_at_chunk` | bool | `true` | every target starts over at its body whenever the base re-plans (smol2), not only at an episode's start |
| `target_tracking.target_in_policy` / `target_in_critic` | bool | `true` / `false` | append the current targets, relative to their bodies (`R^T (p_t - p)`, `log(R^T R_t)`; a joint target minus the joints), to the agent's / the critics' observation |
| `ee_relative.enabled` | bool | `false` | the observation relative to the EE |
| `ee_relative.ee_pos` / `ee_rot` / `ee_rot_param` | spans, parametrisation | `[0, 3]` / `[3, 9]` / `rot6d` | the EE's pose in the observation (removed after) |
| `ee_relative.poses` | list of `{pos, rot, param}` | `[]` | poses re-expressed `(R_ee^T (p - p_ee), R_ee^T R)`, in their own columns and parametrisation |
| `ee_relative.vectors` | list of `[start, stop)` | `[]` | 3-wide vectors rotated into the EE's axes (velocities) |
| `additive_residual.enabled` | bool | `false` | `sent = base + scale * r` |
| `additive_residual.scale` | list[float] \| `{phase: list[float]}` \| null | `null` | the scale per dim of the action below, the same in every phase or per phase (each env its current phase's from the env's tracker; every phase the residual acts in needs one; the critics get the row's scale in their side input, the demos invert each frame with its phase's, the adaptive noise factor divides by each routed learner's own phase's); null: one of the rules below (only where the action is still the base's own: the base policy, dropped dims and the EE-relative observation below) |
| `additive_residual.scale_fraction` | float | `0.1` | ResFiT's rule: `scale_fraction x` the base's half range per dim (`half_range_floor` set: of its demo dataset, floored) |
| `additive_residual.half_range_floor` | float \| null | `null` | the floor on that half range (ResFiT: 0.05); null: the base normalizer's half range |
| `additive_residual.phase_stats` | str \| null | `null` | per routed phase (`<learner>.route_labels`, phases from the env's tracker, `furniture.track_phases`) its own scale from this file; a phase without a residual gets 0; the critics' scale rides in the side input |
| `additive_residual.clip` | float \| null | `null` | clamp the sent action to `center +- clip x` the base's demo half range (ResFiT's widened range); null: no clamp |
| `multiply_residual.enabled` | bool | `false` | `sent = r * base`; r's range and start are `ddpg.action_range` and `model.actor.output_bias` (ddpg only) |
| `multiply_residual.invert_floor` | list[float] | `[1e-6]` | a demo's r is `executed / base`, 1 where `|base|` is under this (one number or one per dim), clamped to the range |
| `multiply_residual.r_range_clip` | float | `0.0` | not 0: the sent action clamped to `base +- r_range_clip x` the additive residual's scale per dim (`additive_residual.scale` or `scale_fraction x` the demo half range), the change an additive residual with r in `+-r_range_clip` can make; the critics' action too. 0: no clamp |
| `multiply_residual.r_range_clip_actor` | bool | `false` | true: `r_range_clip` clamps the policy's own action instead: the actor's r (unchanged) clamped so `r * base` stays within `base +- r_range_clip x scale`, i.e. r within `1 +- limit / \|base\|` (`register_action_bound`; straight-through: the actor's gradient passes the clamp); the exploration noise (times the base) is added after the clamp, so the sent action is `clamped r * base + noise * base`, and neither it nor the critics' action is clamped again. Needs `r_range_clip > 0` |

**The critic is asymmetric** on a Forge task: Forge gives the env a critic state (`task.cfg.state_order`: true peg and
hole poses, joint positions, gains, ...), which the base policy wrapper keeps for the critics while the actor sees the
observation. That is privileged sim information; a real-robot run needs a critic built from what the robot can
measure. The demo conversion refuses an env critic group of its own (it is not recorded in the raw demos).

### The agent's observation history

| field | type | default | what it does |
| --- | --- | --- | --- |
| `history.enabled` | bool | `false` | append the agent's observation history (`my_project/wrappers/env_wrappers.py`, `ObservationHistory`), outermost, after the residual: the base's input never changes |
| `history.length` | int | `8` | previous steps kept per term (15 Hz: ~0.5 s), oldest first; at an episode's first step the current value repeated |
| `history.terms` | list[str] | `[fingertip_pos, ft_force]` | `task.cfg.obs_order` terms to keep, read at their positions in the policy observation (Forge's `fingertip_pos` is the noisy estimate, `ft_force` the noisy 3-d force: Forge observes no torque). Appended to the critic state too, when the env has one |

### Ending episodes on success

`episode.reset_on_success` (`my_project/wrappers/env_wrappers.py`, `ResetOnSuccess`) ends an env's episode
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
`my_project/wrappers/env_wrappers.py`) attaches the package's controller (`ForgeControllerWrapper`,
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

`force_obs` (`my_project/wrappers/env_wrappers.py`, `ForceObservation`) appends the end-effector wrench to
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
| `bc.stop_success_rate` | float | `null` | stop after the first rollout whose success rate is above this (its checkpoint saved first, as the best one); needs rollouts (`bc.rollout_every` > 0); `null`: no success limit |
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
