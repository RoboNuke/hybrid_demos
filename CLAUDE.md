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
* `configs/base/` — what every experiment shares; `configs/experiments/` — one file per
  experiment, `base:` chained, only non-defaults set; `configs/eval/` — eval conditions.
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
  `tasks/__init__.py`, a base config under `configs/base/`. Tests that need the sim follow
  the package's `tests/<module>/GPU/` pattern.
* a **config section**: dataclass in `cfg.py`, `register_section` in all three scripts'
  `setup()`, README row.
* a **loss**: subclass in `losses.py`, a `losses.terms` entry in the experiment YAML.
