"""The config tree's rules, at the file level (the package's ``base``-chain reader, no Isaac Lab).

``configs/base/fb_one_leg.yaml`` is the env: the task and what physic_aligned_evalHappy_a0 changed from
its defaults. ``configs/experiments/resip_replica/`` holds the group's foundation (everything both
trainings share) and one launch config per training; each launch inherits the base's env unchanged but
for its env count. Only the residual launches switch the residual and the history on, sample demos for
their whole run, and log to another wandb project than their base BC run; residual_serl_g099_qmin is
residual_resip_replica but for its discount, its actor's Q and its own group and demo file, and
residual_serl_g099_qmin_std01_fixed_alpha is that but for a fixed small std and temperature, and
residual_serl_g0995_qmin_std01_fixed_alpha is that but for gamma 0.995, and its _scale002 child cuts the
residual's authority to 2% and logs every 1000 steps with a 10-episode success window; its
_std05_alpha4e-4_critic_full_action child rescales the std band (x5) and the temperature (/25) to that
authority and puts the critics on the executed action; its _nstep10 child goes back to ResiP's 10%
authority with the std band rescaled to it, alpha 25x too stiff, and 10-step targets, and its _alpha0p1
child is that with the alpha rescaled as meant, and its _alpha0p01 child that with alpha / 10. The task cfg is not validated here (that needs the sim);
the sections are, through their dataclasses.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from robonuke_rl_core.configfile import load_file_chain
from robonuke_rl_core.learners.cfg import SACCfg

from my_project.cfg import (AdditiveResidualCfg, BasePolicyCfg, BCCfg, DropColumnsCfg, EERelativeCfg, HistoryCfg,
                            MultiplyResidualCfg, TargetTrackingCfg)

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
GROUP = CONFIGS / "experiments" / "resip_replica"
RESIDUAL_LAUNCHES = ("residual_resip_replica", "residual_serl_g099_qmin", "residual_serl_g099_qmin_std01_fixed_alpha",
                     "residual_serl_g0995_qmin_std01_fixed_alpha", "residual_serl_g0995_qmin_std01_fixed_alpha_scale002",
                     "residual_serl_g0995_qmin_scale002_std05_alpha4e-4_critic_full_action",
                     "residual_serl_g0995_qmin_full_action_nstep10_scale01_std01_alpha2p5",
                     "residual_serl_g0995_qmin_full_action_nstep10_scale01_std01_alpha0p1",
                     "residual_serl_g0995_qmin_full_action_nstep10_scale01_std01_alpha0p01")
LAUNCHES = ("bc_resip_replica", *RESIDUAL_LAUNCHES)

#: what physic_aligned_evalHappy_a0 trained with on top of the ResiP task's defaults
EVAL_HAPPY_ENV = {
    "task.name": "Isaac-Furniture-OneLeg-FBAbs-ResiP-v0",
    "task.cfg.episode_length_s": 100.0,
    "task.cfg.furniture.part_assets": "furniturebench_top1024",
    "task.cfg.furniture.gripper_mode": "position",
    "task.cfg.furniture.robot_self_collisions": True,
    "task.cfg.furniture.gripper_velocity_limit": 0.2,
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
    assert "task.cfg.furniture.ik_jacobian" not in base  # the env's only IK is the TCP Jacobian; the option is gone


# ------------------------------------------------- the interface relative to the end-effector
REL_GROUP = CONFIGS / "experiments" / "resip_replica_rel"


def test_the_relative_base_is_fb_one_leg_with_only_the_task_name_changed():
    ab, rel = leaves(merged(CONFIGS / "base" / "fb_one_leg.yaml")), leaves(merged(CONFIGS / "base" / "fb_one_leg_rel.yaml"))
    assert {k for k in set(ab) | set(rel) if ab.get(k) != rel.get(k)} == {"task.name"}
    assert rel["task.name"] == "Isaac-Furniture-OneLeg-FBRel-ResiP-v0"


DELTA_GROUP = CONFIGS / "experiments" / "policy_dec_replica"


def test_the_delta_base_is_fb_one_leg_with_only_the_task_name_changed():
    ab, delta = leaves(merged(CONFIGS / "base" / "fb_one_leg.yaml")), leaves(merged(CONFIGS / "base" / "fb_one_leg_delta.yaml"))
    assert {k for k in set(ab) | set(delta) if ab.get(k) != delta.get(k)} == {"task.name"}
    assert delta["task.name"] == "Isaac-Furniture-OneLeg-FBDelta-ResiP-v0"


CHUNK_GROUP = CONFIGS / "experiments" / "policy_dec_replica_chunk"


def test_the_resip_ppo_replica_is_resips_residual_recipe_on_our_base():
    c = leaves(merged(CONFIGS / "experiments" / "resip_ppo_replica" / "residual_resip_ppo_replica.yaml"))
    assert c["trainer.learner"] == "ppo" and c["base_policy.enabled"] is True and c["model.architecture"] == "mlp"
    assert (c["ppo.rollouts"], c["ppo.learning_epochs"], c["ppo.mini_batches"], c["ppo.discount_factor"], c["ppo.gae_lambda"]) == (700, 50, 1, 0.999, 0.95)
    assert (c["ppo.kl_threshold"], c["ppo.grad_norm_clip"], c["ppo.policy_lr"], c["ppo.value_lr"], c["ppo.value_clip"]) == (0.1, 1.0, 3.0e-4, 5.0e-3, 0.0)
    assert c["ppo.normalize_rewards"] is True and c["ppo.reward_clip"] == 5.0 and c["ppo.normalize_values"] is False
    assert c["ppo.reset_every_iteration"] is True and c["ppo.time_limit_bootstrap"] is False and c["ppo.state_preprocessor"] == "none"
    assert (c["ppo.lr_schedule"], c["ppo.policy_lr_warmup"], c["ppo.value_lr_warmup"]) == ("cosine", 5, 0)
    assert (c["model.actor.hidden_size"], c["model.actor.num_layers"], c["model.actor.activation"]) == (512, 2, "SiLU")
    assert c["model.actor.init_log_std"] == -1.0 and c["model.actor.learn_std"] is False and c["model.actor.squash"] is False
    assert c["model.actor.output_init_scale"] == 0.01 and (c["model.critic.hidden_size"], c["model.critic.activation"]) == (512, "SiLU")
    assert c["task.cfg.scene.num_envs"] == 1024 and c["task.cfg.episode_length_s"] == 70.0
    assert c["trainer.total_timesteps"] % c["ppo.rollouts"] == 0 and c["trainer.checkpoint_interval"] == 5 * c["ppo.rollouts"]
    assert c["additive_residual.enabled"] is True and c["additive_residual.scale_fraction"] == 0.1
    assert c.get("history.enabled", False) is False


# ------------------------------------------------------------------ every experiment config loads
#: launch configs known not to validate, and why (left as they are, not guessed at)
KNOWN_BROKEN = {}


def _sections(path: Path):
    """``{name: section object}`` of the learner and residual wrapper sections the file's chain sets, merged onto their
    dataclasses (unknown keys raise)."""
    from robonuke_rl_core.learners.cfg import DDPGCfg, PPOCfg

    merged = OmegaConf.merge(*[layer for _, layer in load_file_chain(path)])
    out = {}
    for name, cls in (("base_policy", BasePolicyCfg), ("drop_columns", DropColumnsCfg), ("target_tracking", TargetTrackingCfg),
                      ("ee_relative", EERelativeCfg), ("additive_residual", AdditiveResidualCfg),
                      ("multiply_residual", MultiplyResidualCfg), ("ddpg", DDPGCfg), ("sac", SACCfg), ("ppo", PPOCfg)):
        if name in merged:
            out[name] = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(cls), merged[name]))
    return out


EXPERIMENT_FILES = sorted(p for p in (CONFIGS / "experiments").rglob("*.yaml"))


@pytest.mark.parametrize("path", EXPERIMENT_FILES, ids=lambda p: str(p.relative_to(CONFIGS / "experiments")))
def test_every_experiment_config_sets_only_known_fields_and_every_launch_config_validates(path):
    """A stale field (one a refactor removed) fails here, not on the cluster: every file's chain merges onto the section
    dataclasses; every launch config (no leading _) also passes the sections' validate. Foundations are partial by
    design (a demo file, a base run set by the launch), so they are checked for their keys only."""
    rel = str(path.relative_to(CONFIGS / "experiments"))
    if rel in KNOWN_BROKEN:
        pytest.xfail(KNOWN_BROKEN[rel])
    sections = _sections(path)
    if path.name.startswith("_"):
        return
    for name, section in sections.items():
        if name != "ppo":  # PPO's checks read the whole config (env counts): its keys are checked above
            section.validate(None)
    enabled = {name for name, section in sections.items() if getattr(section, "enabled", False)}
    if enabled & {"base_policy", "drop_columns", "target_tracking", "ee_relative", "additive_residual", "multiply_residual"}:
        assert "base_policy" in enabled, rel
        assert len(enabled & {"additive_residual", "multiply_residual"}) == 1, rel  # exactly one residual


def test_a_method_file_adds_onto_a_task_file_through_a_list_base():
    """configs/methods/resfit.yaml is ResFiT with no task: one_leg's foundation chains it after its task file; the robosuite
    residual launch configs chain it before their foundation (the task file and the task's own values), so the BC base
    trained from the same foundation never gets the method's MLP networks."""
    from robonuke_rl_core.configfile import chain

    files = [p.name for p in chain(CONFIGS / "experiments" / "ResFiT" / "_foundation.yaml")]
    assert files[-3:] == ["fb_one_leg.yaml", "resfit.yaml", "_foundation.yaml"]
    files = [p.name for p in chain(CONFIGS / "experiments" / "resfit_can" / "residual_resfit_can.yaml")]
    assert files[0] == "resfit.yaml" and files[-3:] == ["robosuite_can.yaml", "_foundation.yaml", "residual_resfit_can.yaml"]
    assert "resfit.yaml" not in [p.name for p in chain(CONFIGS / "experiments" / "resfit_can" / "bc_resfit_can.yaml")]
    can = leaves(merged(CONFIGS / "experiments" / "resfit_can" / "residual_resfit_can.yaml"))
    one_leg = leaves(merged(CONFIGS / "experiments" / "ResFiT" / "_foundation.yaml"))
    for key in ("additive_residual.scale_fraction", "additive_residual.half_range_floor", "additive_residual.clip",
                "ddpg.action_range", "ddpg.num_critics", "ddpg.critic_subset", "ddpg.policy_delay"):
        assert can[key] == one_leg[key], key  # the same method on two tasks
    assert can["additive_residual.enabled"] is True and can["ddpg.action_range"] == [-0.1, 0.1]
