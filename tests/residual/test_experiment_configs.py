"""The config tree's rules, at the file level (the package's ``base``-chain reader, no Isaac Lab).

``configs/base/fb_one_leg.yaml`` is the env: the task and what physic_aligned_evalHappy_a0 changed from
its defaults. ``configs/experiments/resip_replica/`` holds the group's foundation (everything both
trainings share) and one launch config per training; each launch inherits the base's env unchanged but
for its env count. Only the residual launches switch the residual and the history on, sample demos for
their whole run, and log to another wandb project than their base BC run; residual_serl_g099_qmin is
residual_resip_replica but for its discount, its actor's Q and its own group and demo file, and
residual_serl_g099_qmin_std01_fixed_alpha is that but for a fixed small std and temperature, and
residual_serl_g0995_qmin_std01_fixed_alpha is that but for gamma 0.995, and its _scale002 child cuts the
residual's authority to 2% and logs every 1000 steps with a 10-episode success window. The task cfg
is not validated here (that needs the sim);
the sections are, through their dataclasses.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from robonuke_rl_core.configfile import load_file_chain
from robonuke_rl_core.learners.cfg import SACCfg

from my_project.cfg import BCCfg, HistoryCfg, ResidualCfg

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
GROUP = CONFIGS / "experiments" / "resip_replica"
RESIDUAL_LAUNCHES = ("residual_resip_replica", "residual_serl_g099_qmin", "residual_serl_g099_qmin_std01_fixed_alpha",
                     "residual_serl_g0995_qmin_std01_fixed_alpha", "residual_serl_g0995_qmin_std01_fixed_alpha_scale002")
LAUNCHES = ("bc_resip_replica", *RESIDUAL_LAUNCHES)

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


def test_only_the_residual_launches_switch_the_residual_and_the_history_on():
    foundation, bc = (merged(GROUP / f"{stem}.yaml") for stem in ("_foundation", "bc_resip_replica"))
    for cfg in (foundation, bc):
        assert not cfg["residual"].get("enabled", False) and not cfg["history"].get("enabled", False)
    assert bc["trainer"]["learner"] == "diffusion_bc"
    for launch in RESIDUAL_LAUNCHES:
        residual = merged(GROUP / f"{launch}.yaml")
        assert residual["residual"]["enabled"] and residual["history"]["enabled"]
        assert residual["trainer"]["learner"] == "sac"


def test_g099_qmin_is_the_serl_residual_but_for_its_discount_its_actors_q_and_its_own_group_and_demos():
    serl, new = (leaves(merged(GROUP / f"{stem}.yaml")) for stem in RESIDUAL_LAUNCHES[:2])
    diff = {k for k in set(serl) | set(new) if serl.get(k, "<unset>") != new.get(k, "<unset>")}
    assert diff == {"sac.discount_factor", "sac.actor_q_reduction", "wandb.group", "sac.demo_transitions"}
    assert (new["sac.discount_factor"], new["sac.actor_q_reduction"]) == (0.99, "min")


def test_std01_fixed_alpha_is_g099_qmin_but_for_a_fixed_small_std_and_temperature_and_its_own_group_and_demos():
    parent, new = (leaves(merged(GROUP / f"{stem}.yaml")) for stem in RESIDUAL_LAUNCHES[1:3])
    diff = {k for k in set(parent) | set(new) if parent.get(k, "<unset>") != new.get(k, "<unset>")}
    assert diff == {"model.actor.act_init_std", "model.actor.min_log_std", "model.actor.max_log_std",
                    "sac.learn_entropy", "wandb.group", "sac.demo_transitions"}
    assert new["sac.learn_entropy"] is False and new["sac.initial_entropy_value"] == 0.01
    assert 0.079 < math.exp(new["model.actor.min_log_std"]) < new["model.actor.act_init_std"] <= math.exp(new["model.actor.max_log_std"]) < 0.101


def test_g0995_is_std01_fixed_alpha_but_for_its_discount_and_its_own_group_and_demos():
    parent, new = (leaves(merged(GROUP / f"{stem}.yaml")) for stem in RESIDUAL_LAUNCHES[2:4])
    diff = {k for k in set(parent) | set(new) if parent.get(k, "<unset>") != new.get(k, "<unset>")}
    assert diff == {"sac.discount_factor", "wandb.group", "sac.demo_transitions"}
    assert (parent["sac.discount_factor"], new["sac.discount_factor"]) == (0.99, 0.995)


def test_scale002_is_g0995_but_for_its_authority_its_logging_and_its_own_group_and_demos():
    parent, new = (leaves(merged(GROUP / f"{stem}.yaml")) for stem in RESIDUAL_LAUNCHES[3:5])
    diff = {k for k in set(parent) | set(new) if parent.get(k, "<unset>") != new.get(k, "<unset>")}
    assert diff == {"residual.action_scale_fraction", "trainer.write_interval", "trainer.success_window",
                    "wandb.group", "sac.demo_transitions"}
    assert (parent["residual.action_scale_fraction"], new["residual.action_scale_fraction"]) == (0.1, 0.02)
    assert (new["trainer.write_interval"], new["trainer.success_window"], new["trainer.checkpoint_interval"]) == (1000, 10, 10000)


@pytest.mark.parametrize("launch", RESIDUAL_LAUNCHES)
def test_the_residual_samples_demos_and_keeps_every_transition_for_its_whole_run(launch):
    residual = merged(GROUP / f"{launch}.yaml")
    assert residual["sac"]["demo_sampling_start"] == 0
    assert residual["sac"]["demo_sampling_end"] >= residual["trainer"]["total_timesteps"]
    assert residual["memory"]["memory_size"] >= residual["trainer"]["total_timesteps"]


@pytest.mark.parametrize("launch", RESIDUAL_LAUNCHES)
def test_the_residual_logs_to_another_project_than_its_base_bc_run(launch):
    bc, residual = (merged(GROUP / f"{stem}.yaml") for stem in ("bc_resip_replica", launch))
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
