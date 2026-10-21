"""The config tree's rules, at the file level (the package's ``base``-chain reader, no Isaac Lab).

``configs/base/fb_one_leg.yaml`` is the env: the task and what physic_aligned_evalHappy_a0 changed from
its defaults. ``configs/experiments/resip_replica/`` holds the group's foundation (everything both
trainings share) and one launch config per training; each launch inherits the base's env unchanged but
for its env count. Only the residual launch switches the residual and the history on, and it logs to
another wandb project than its base BC run. The task cfg is not validated here (that needs the sim);
the sections are, through their dataclasses.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from robonuke_rl_core.configfile import load_file_chain
from robonuke_rl_core.learners.cfg import SACCfg

from my_project.cfg import BCCfg, HistoryCfg, ResidualCfg

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
GROUP = CONFIGS / "experiments" / "resip_replica"
LAUNCHES = ("bc_resip_replica", "residual_resip_replica")

#: what physic_aligned_evalHappy_a0 trained with on top of the ResiP task's defaults
EVAL_HAPPY_ENV = {
    "task.name": "Isaac-Furniture-OneLeg-FBAbs-ResiP-v0",
    "task.cfg.episode_length_s": 100.0,
    "task.cfg.furniture.part_assets": "furniturebench_top1024",
    "task.cfg.furniture.gripper_mode": "position",
    "task.cfg.furniture.robot_self_collisions": True,
    "task.cfg.furniture.gripper_velocity_limit": 0.2,
    "task.cfg.furniture.ik_jacobian": "isaaclab",
}
#: sections besides the task that shape the env the policy acts in
ENV_SECTIONS = ("task", "wrappers", "action", "hybrid", "episode", "force_obs", "demo")


def merged(path: Path) -> dict:
    return OmegaConf.to_container(OmegaConf.merge(*[layer for _, layer in load_file_chain(path)]), resolve=True)


def leaves(data, prefix=""):
    if isinstance(data, dict):
        out = {}
        for key, value in data.items():
            out.update(leaves(value, f"{prefix}.{key}" if prefix else key))
        return out
    return {prefix: data}


def env_of(data: dict) -> dict:
    return {k: v for k, v in leaves(data).items() if k.split(".")[0] in ENV_SECTIONS}


def test_the_base_is_the_task_with_eval_happys_physics():
    base = leaves(merged(CONFIGS / "base" / "fb_one_leg.yaml"))
    assert {k: base[k] for k in EVAL_HAPPY_ENV} == EVAL_HAPPY_ENV


@pytest.mark.parametrize("launch", LAUNCHES)
def test_each_launch_runs_in_the_bases_env_but_for_its_env_count(launch):
    base = env_of(merged(CONFIGS / "base" / "fb_one_leg.yaml"))
    exp = env_of(merged(GROUP / f"{launch}.yaml"))
    diff = {k for k in set(base) | set(exp) if base.get(k, "<unset>") != exp.get(k, "<unset>")}
    assert diff == {"task.cfg.scene.num_envs"}


def test_only_the_residual_launch_switches_the_residual_and_the_history_on():
    foundation, bc, residual = (merged(GROUP / f"{stem}.yaml") for stem in ("_foundation", *LAUNCHES))
    for cfg in (foundation, bc):
        assert not cfg["residual"].get("enabled", False) and not cfg["history"].get("enabled", False)
    assert residual["residual"]["enabled"] and residual["history"]["enabled"]
    assert bc["trainer"]["learner"] == "diffusion_bc" and residual["trainer"]["learner"] == "sac"


def test_the_residual_logs_to_another_project_than_its_base_bc_run():
    bc, residual = (merged(GROUP / f"{stem}.yaml") for stem in LAUNCHES)
    assert residual["residual"]["base_project"] == bc["wandb"]["project"] == "hybrid_demo_furniture"
    assert residual["residual"]["base_run_name"] == "physic_aligned_evalHappy_a0"
    assert residual["wandb"]["project"] != bc["wandb"]["project"]
    assert residual["residual"].get("base_run") is None  # the wandb run, not a local folder


@pytest.mark.parametrize("launch", LAUNCHES)
def test_the_sections_validate(launch):
    data = merged(GROUP / f"{launch}.yaml")
    for cls, name in ((SACCfg, "sac"), (HistoryCfg, "history"), (BCCfg, "bc"), (ResidualCfg, "residual")):
        section = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(cls), data.get(name, {})))
        section.validate(None)
