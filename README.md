# robonuke_project_template

The starting point for a research project built on
[robonuke_rl_core](https://github.com/RoboNuke/robonuke_rl_core). One paper = one repo made
from this template.

## Start a project

0. Once, if the button in step 1 is missing: this repo's GitHub page -> **Settings ->
   General -> check "Template repository"**. That setting is what makes GitHub show
   **Use this template** on the repo's main page (green button, next to Code).
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
| experiment YAMLs (`base:` chains, override only what differs) | `configs/` | `--config` / `--eval_config` |
| cluster resources (the `hpc` section: account, partitions, walltime, image) | `configs/base/hpc.yaml` | chained under every experiment |
| SLURM submitters (thin callers of `robonuke_rl_core.hpc`) | `launchers/` | — |
| overlays / model architectures | same pattern: register in `setup()` | package CLAUDE.md recipes |

The demo task (`Isaac-Forge-DemoPegInsert-Direct-v0`) subclasses Isaac Lab's Forge peg
insert unchanged — "forge" in the task name is what lets the package's Forge wrappers
(controller, fragile peg, efficient reset, contact sensing) apply. Replace its `pass`
bodies with your observations, rewards and resets.

## Run

```bash
# train (3 agents, hybrid force/position control, fragile peg)
python scripts/train.py --config configs/experiments/hybrid_fragile.yaml --headless

# evaluate / record a trained run (downloads policy + config from the wandb run's files)
python scripts/eval.py --run hur/template_demo/<run_name> \
    --eval_config configs/eval/quick.yaml --headless [--record]

# watch it live (j pause / k reset / l quit), or inspect the reset distribution
python scripts/debug.py --run hur/template_demo/<run_name>
python scripts/debug.py --run hur/template_demo/<run_name> --resets --hold_seconds 2
```

`configs/experiments/match_fragile.yaml` is the MATCH variant: the selection-conditioned
distribution plus the supervised selection loss.

## Run on the cluster

The launchers submit SLURM + Apptainer jobs from a login node — a light python is enough
(omegaconf + pyyaml; wandb only for `launch_eval`'s run query). Resources come from the
`hpc` config section (`configs/base/hpc.yaml`, overridable per experiment or on the CLI);
the field reference is the package README. Names derive from wandb: `--project` and
`--group_prefix` are required, group = `{group_prefix}_{config_stem}` (sweeps append
`_{label}-{value}`), the SLURM job name is the group, runs are `{group}_a{i}`.

```bash
# one job per experiment yaml in the folder
python launchers/launch_train.py configs/experiments --project template_demo \
    --group_prefix oct06 --tag demo

# one job per swept value (group oct06_hybrid_fragile_lr-1.0e-4, ...)
python launchers/launch_sweep.py configs/experiments/hybrid_fragile.yaml \
    --project template_demo --group_prefix oct06 \
    --sweep_param sac.actor_lr --label lr --value 1.0e-4 --value 3.0e-4

# one eval job per (trained run x eval config), selected by group
python launchers/launch_eval.py --eval_config configs/eval/quick.yaml \
    --project template_demo --group oct06_hybrid_fragile
```

`--dry_run` on any of them prints the exact `sbatch` commands and submits nothing.
`WANDB_API_KEY` is read from your login shell (never a config field); without it the job
logs offline instead of crashing.

## Config sections this project adds

| field | type | default | what it does |
| --- | --- | --- | --- |
| `demo.reward_scale` | float | `1.0` | example field of the example section — replace with your project's |
