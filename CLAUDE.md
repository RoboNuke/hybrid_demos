# <project> — Claude context

A research project built on the shared package
[robonuke_rl_core](https://github.com/RoboNuke/robonuke_rl_core). **The package's CLAUDE.md
(in its clone, usually `~/robonuke_rl_core`) holds the recipes** — add a task / model /
learner / loss / config class, the config system, the independence rule, eval and debug.
This file holds only what is project-specific. Read the package CLAUDE.md before changing
anything that touches package machinery.

Test env: the `general` conda env (`/home/hunter/miniconda3/envs/general/bin/python`), with
the package installed editable from its single clone (`pip install -e ~/robonuke_rl_core`).
Never create a new conda env; never vendor a copy of the package into this repo.

## Rules

* The rules of the package CLAUDE.md apply here unchanged: fail fast and loud, OmegaConf
  only, block-style YAML, one seed, do not commit or push without approval, report tests as
  passed / failed / total.
* **Every project config field is documented** in this repo's README table, same rule as the
  package's.
* Everything the project registers happens inside the entry scripts' `setup()` hook — task
  imports, `register_section`, loss/overlay/architecture imports — never at module top:
  task modules import Isaac Lab, which only works after `AppLauncher`.

## Layout

* `my_project/tasks/` — env classes + task cfgs + `gym.register` (in `tasks/__init__.py`).
  A task that should work with the package's Forge wrappers needs "forge" in its id.
* `my_project/cfg.py` — project config sections (`@dataclass`, `validate`), registered in
  `setup()`.
* `my_project/losses.py` — project `AuxLoss` subclasses, `@register_loss` at import time.
* `my_project/models/` — the diffusion policy: `transformer.py` (near copy of
  robust-rearrangement's), `diffusion_policy.py` (normalizer, DDPM/DDIM, chunked acting),
  `cfg.py` (`model.architecture: diffusion_transformer`). Plain single-agent modules, never
  `VmapEnsemble`.
* `my_project/bc/` — the `diffusion_bc` learner: `dataset.py` (LeRobot parquet → windows),
  `train.py` (the loop, run by `scripts/train_bc.py`). Its own entry point because the
  package's `train.main` runs only sac / ppo. A field both the policy and the trainer read
  goes in `model.actor`, never also in `bc`.
* `my_project/wrappers.py` — the policy's action meaning (`action.*`: world-pose actions),
  applied by the recorder, replay and BC rollouts through `apply_policy_wrappers`.
  `my_project/obs_terms.py` adds named Factory/Forge observation terms (`hole_estimate`), so
  the observation layout is `task.cfg.obs_order` config; installed in every script's `setup()`.
* `my_project/residual.py` — residual RL on a frozen diffusion policy (`residual.*`); its
  wrapper reaches the package's train / eval / debug through `register_env_wrapper`
  (`project_env_wrappers` in `my_project/wrappers.py`, registered in every script's `setup()`).
  `history.*` (`ObservationHistory`, same file) goes on outermost, after the residual, so the
  base's input never changes. `my_project/residual_demos.py` (`scripts/make_demo_transitions.py`)
  converts recorded demo frames offline (no sim) into an experiment's `sac.demo_transitions`, reusing `residual_observation` and `HistoryBuffer`, and
  `project_env_wrappers` refuses a file whose fingerprint is another setup's. The critic is
  asymmetric (Forge's `state_order`: privileged sim state).
* `my_project/demos/` — hand-teleoperated demos: `teleop.py` (velocity → Forge action),
  `gamepad.py`, `writer.py` (LeRobot-style dataset), `state.py` (episode start snapshot /
  restore), `record.py` and `replay.py` (run by `scripts/record_demos.py` /
  `scripts/replay_demos.py`). The env is bare `gym.make`: no package wrappers. Fixed
  dynamics are config only, never task code.
* `configs/` — `base:` chained, only non-defaults set:
  * `base/<task>.yaml` (`fb_one_leg.yaml`) — the env only: the task id and its physics, everything
    independent of the training method, so whatever bases on it gets a known task with known
    physics; chained onto `base/hpc.yaml`.
  * `eval/` — only what one test changes on top of the trained run's config (more friction, a
    recording); chained onto `base/hpc.yaml` for the eval launcher's resources.
  * `experiments/<group>/` — `_foundation.yaml` (based on a `base/` task) holds what all of the
    group's trainings share: seed, wandb entity, every learning parameter (`bc.*`, `sac.*`,
    `residual.*`, `history.*`, ...); one launch config per training, named type first
    (`bc_<group>.yaml`, `residual_<group>.yaml`), sets its learner (`trainer`), its `model`, its
    wandb project and group, its env count — and switches on the wrappers only it uses
    (`residual.enabled`, `history.enabled`: in the foundation they would wrap the BC rollouts too).
    The leading `_` keeps `launch_train.py <folder>` from submitting the foundation.
  * Demos are collected outside this repo (Robonuke_Furniture_Bench's `collect_scripted.py`); there
    is no demos folder.
* `scripts/` — thin callers of `robonuke_rl_core.{train,eval,debug}.main(setup=setup)`.
  Do not add logic here; flow fixes belong in the package.
* `launchers/` — thin callers of `robonuke_rl_core.hpc.launch_{train,sweep,eval}.main()`,
  run from the project root on a cluster login node. No `setup` hook: the submitters never
  load the full config. Cluster resources are the `hpc` config section —
  `configs/base/hpc.yaml` holds the cluster-wide values and sits at the bottom of every
  experiment's `base` chain; an experiment overrides only what differs (`hpc.time`,
  `hpc.mem`). The naming rules and field reference are in the package CLAUDE.md ("HPC
  launch") and README.

## Add things

Follow the package CLAUDE.md recipe for the component, then:

* a **task**: cfg + env subclass in `my_project/tasks/<name>.py`, `gym.register` in
  `tasks/__init__.py`, a base config `configs/base/<task>.yaml` (the task and its physics only).
  Tests that need the sim follow
  the package's `tests/<module>/GPU/` pattern.
* a **config section**: dataclass in `cfg.py`, `register_section` in all three scripts'
  `setup()`, README row.
* a **loss**: subclass in `losses.py`, a `losses.terms` entry in the experiment YAML.
