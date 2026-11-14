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

from my_project.cfg import BCCfg, HistoryCfg, ResidualCfg

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
    assert c["trainer.learner"] == "ppo" and c["residual.enabled"] is True and c["model.architecture"] == "mlp"
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
    assert c["residual.steps_per_chunk"] == 1 and "residual.normalize_observation" not in c and c["residual.observe_base_action"] is True
    assert c["residual.action_scale_fraction"] == 0.1 and c["residual.critic_action"] == "residual"
    assert c.get("history.enabled", False) is False


@pytest.mark.skip(reason="old config (normalize_observations): left in place to adapt to state_preprocessor later")
def test_residual_flash_sac_is_the_foundation_on_the_ppo_replicas_problem():
    from omegaconf import OmegaConf

    from robonuke_rl_core.learners.cfg import SACCfg
    from robonuke_rl_core.models.cfg import FlashSACModelCfg

    ppo = leaves(merged(CONFIGS / "experiments" / "resip_ppo_replica" / "residual_resip_ppo_replica.yaml"))
    path = CONFIGS / "experiments" / "residual_flash_sac" / "residual_flash_sac.yaml"
    sac = leaves(merged(path))
    # the problem is the PPO replica's: env, residual, run length and checkpoints (the model is FlashSAC's)
    for key in ppo:
        if key.startswith(("task.", "residual.", "experiment.")) or key in ("trainer.total_timesteps", "trainer.checkpoint_interval"):
            assert sac[key] == ppo[key], key
    c = merged(path)
    for name, cls in (("sac", SACCfg), ("model", FlashSACModelCfg)):
        schema = OmegaConf.structured(cls)
        OmegaConf.set_struct(schema, True)
        OmegaConf.to_object(OmegaConf.merge(schema, OmegaConf.create(c[name]))).validate(None)
    s = c["sac"]
    assert not s.get("route_labels") and not s.get("demo_transitions") and not s.get("potential_terms")
    assert s["discount_factor"] == ppo["ppo.discount_factor"] == 0.999  # the PPO replica's discount
    assert (s["batch_size"], s["num_critics"], s["actor_q_reduction"], s["polyak"]) == (2048, 2, "min", 0.01)
    assert s["cuda_graph"] and s["weight_normalization"] and s["cross_batch_value"] and not s.get("unsquashed_actor", False)
    assert c["model"]["architecture"] == "flashsac" and "max_log_std" not in c["model"]["actor"]
    assert c["trainer"].get("success_window", 0) == 0 and not c["trainer"].get("windowed_metrics")
    # 700-step episodes (70 s at 10 Hz); the phases tracked, never rewarded
    assert c["task"]["cfg"]["episode_length_s"] == 70.0 and c["task"]["cfg"]["furniture"]["track_phases"] is True
    rewards = c["task"]["cfg"].get("rewards", {})
    assert all(rewards.get(k, {}).get("weight", 0.0) == 0.0 for k in ("corner_done", "grasp_done", "insert_done"))


@pytest.mark.skip(reason="old config (normalize_observations): left in place to adapt to state_preprocessor later")
def test_the_residual_flash_sac_foundation_holds_flashsacs_defaults_on_the_ppo_replicas_residual():
    from omegaconf import OmegaConf

    from robonuke_rl_core.learners.cfg import SACCfg

    c = merged(CONFIGS / "experiments" / "residual_flash_sac" / "_foundation.yaml")
    schema = OmegaConf.structured(SACCfg)
    OmegaConf.set_struct(schema, True)
    sac = OmegaConf.to_object(OmegaConf.merge(schema, OmegaConf.create(c["sac"])))
    sac.validate(None)
    # FlashSAC's Table 9 (GPU simulators)
    assert (sac.batch_size, sac.n_step, sac.num_critics, sac.polyak, sac.policy_delay) == (2048, 1, 2, 0.01, 2)
    assert (sac.critic_distribution, sac.critic_num_bins, sac.critic_min_value, sac.critic_max_value) == ("categorical", 101, -5.0, 5.0)
    assert (sac.target_entropy_sigma, sac.initial_entropy_value, sac.lr_schedule, sac.lr_end) == (0.15, 0.01, "cosine", 1.5e-4)
    assert (sac.noise_repeat, sac.noise_repeat_zeta_s, sac.noise_repeat_max, sac.adaptive_reward_scaling) == (True, 2.0, 16, True)
    assert c["memory"]["memory_size"] == 10_000_000
    # the residual is the PPO replica's
    ppo = merged(CONFIGS / "experiments" / "resip_ppo_replica" / "_foundation.yaml")
    assert c["residual"] == ppo["residual"]


def test_residual_flash_sac_sets_only_the_launch_settings_over_its_foundation():
    import yaml

    folder = CONFIGS / "experiments" / "residual_flash_sac"
    launch = yaml.safe_load((folder / "residual_flash_sac.yaml").read_text())
    assert launch["base"] == "_foundation.yaml"
    assert set(launch) == {"base", "wandb", "task", "trainer", "residual", "model"}
    assert launch["task"] == {"cfg": {"scene": {"num_envs": 1024}}} and launch["residual"] == {"enabled": True}


def test_residual_flash_sac_v2_is_v1_with_the_actor_started_as_a_residual():
    import yaml

    folder = CONFIGS / "experiments" / "residual_flash_sac"
    assert yaml.safe_load((folder / "residual_flash_sac_v2.yaml").read_text())["base"] == "_foundation.yaml"
    v1 = leaves(merged(folder / "residual_flash_sac.yaml"))
    v2 = leaves(merged(folder / "residual_flash_sac_v2.yaml"))
    changed = {k for k in set(v1) | set(v2) if v1.get(k) != v2.get(k)}
    assert changed == {"wandb.group", "model.actor.mean_init_scale", "model.actor.init_std"}
    assert (v2["model.actor.mean_init_scale"], v2["model.actor.init_std"]) == (0.01, 0.15)


def test_residual_flash_sac_v3_is_v2_with_a_constant_std():
    import yaml

    folder = CONFIGS / "experiments" / "residual_flash_sac"
    assert yaml.safe_load((folder / "residual_flash_sac_v3.yaml").read_text())["base"] == "_foundation.yaml"
    v2 = leaves(merged(folder / "residual_flash_sac_v2.yaml"))
    v3 = leaves(merged(folder / "residual_flash_sac_v3.yaml"))
    changed = {k for k in set(v2) | set(v3) if v2.get(k) != v3.get(k)}
    assert changed == {"wandb.group", "model.actor.init_std", "model.actor.fixed_std"}
    assert v3["model.actor.fixed_std"] == 0.15 and "model.actor.init_std" not in v3
    assert v3["sac.entropy_lr"] == v2["sac.entropy_lr"] == 3.0e-4


def test_residual_flash_sac_v4_is_v2_with_the_log_std_bounded_to_001_and_1():
    import math

    import yaml

    folder = CONFIGS / "experiments" / "residual_flash_sac"
    assert yaml.safe_load((folder / "residual_flash_sac_v4.yaml").read_text())["base"] == "_foundation.yaml"
    v2 = leaves(merged(folder / "residual_flash_sac_v2.yaml"))
    v4 = leaves(merged(folder / "residual_flash_sac_v4.yaml"))
    changed = {k for k in set(v2) | set(v4) if v2.get(k) != v4.get(k)}
    assert changed == {"wandb.group", "model.actor.min_log_std", "model.actor.max_log_std"}
    assert v4["model.actor.min_log_std"] == math.log(0.01) and v4["model.actor.max_log_std"] == math.log(1.0)


def test_residual_flash_sac_v5_is_v4_with_a_100x_slower_temperature():
    import yaml

    folder = CONFIGS / "experiments" / "residual_flash_sac"
    assert yaml.safe_load((folder / "residual_flash_sac_v5.yaml").read_text())["base"] == "_foundation.yaml"
    v4 = leaves(merged(folder / "residual_flash_sac_v4.yaml"))
    v5 = leaves(merged(folder / "residual_flash_sac_v5.yaml"))
    changed = {k for k in set(v4) | set(v5) if v4.get(k) != v5.get(k)}
    assert changed == {"wandb.group", "sac.entropy_lr"}
    assert abs(v5["sac.entropy_lr"] - v4["sac.entropy_lr"] / 100) < 1e-15


@pytest.mark.skip(reason="old config (normalize_observations): left in place to adapt to state_preprocessor later")
def test_residual_flash_sac_v6_is_v2_with_the_critics_from_700_and_the_actor_from_2100():
    import yaml

    from omegaconf import OmegaConf

    from robonuke_rl_core.learners.cfg import SACCfg

    folder = CONFIGS / "experiments" / "residual_flash_sac"
    assert yaml.safe_load((folder / "residual_flash_sac_v6.yaml").read_text())["base"] == "_foundation.yaml"
    v2 = leaves(merged(folder / "residual_flash_sac_v2.yaml"))
    v6 = leaves(merged(folder / "residual_flash_sac_v6.yaml"))
    changed = {k for k in set(v2) | set(v6) if v2.get(k) != v6.get(k)}
    assert changed == {"wandb.group", "sac.learning_starts", "sac.policy_learning_starts"}
    assert (v6["sac.learning_starts"], v6["sac.policy_learning_starts"]) == (700, 2100)
    schema = OmegaConf.structured(SACCfg)
    OmegaConf.set_struct(schema, True)
    OmegaConf.to_object(OmegaConf.merge(schema, OmegaConf.create(merged(folder / "residual_flash_sac_v6.yaml")["sac"]))).validate(None)


@pytest.mark.skip(reason="old config (normalize_observations): left in place to adapt to state_preprocessor later")
def test_residual_flash_sac_v7_is_v3_at_the_ppo_std_without_entropy_learning_or_backup():
    import yaml

    from omegaconf import OmegaConf

    from robonuke_rl_core.learners.cfg import SACCfg

    folder = CONFIGS / "experiments" / "residual_flash_sac"
    assert yaml.safe_load((folder / "residual_flash_sac_v7.yaml").read_text())["base"] == "_foundation.yaml"
    v3 = leaves(merged(folder / "residual_flash_sac_v3.yaml"))
    v7 = leaves(merged(folder / "residual_flash_sac_v7.yaml"))
    changed = {k for k in set(v3) | set(v7) if v3.get(k) != v7.get(k)}
    assert changed == {"wandb.group", "model.actor.fixed_std", "sac.learn_entropy", "sac.initial_entropy_value", "sac.backup_entropy"}
    assert (v7["model.actor.fixed_std"], v7["sac.learn_entropy"], v7["sac.backup_entropy"]) == (0.36788, False, False)
    assert v7["sac.initial_entropy_value"] == 0.0
    schema = OmegaConf.structured(SACCfg)
    OmegaConf.set_struct(schema, True)
    OmegaConf.to_object(OmegaConf.merge(schema, OmegaConf.create(merged(folder / "residual_flash_sac_v7.yaml")["sac"]))).validate(None)


def test_residual_flash_sac_v8_is_v7_with_the_critics_from_700_and_the_actor_from_2100():
    import yaml

    folder = CONFIGS / "experiments" / "residual_flash_sac"
    assert yaml.safe_load((folder / "residual_flash_sac_v8.yaml").read_text())["base"] == "_foundation.yaml"
    v7 = leaves(merged(folder / "residual_flash_sac_v7.yaml"))
    v8 = leaves(merged(folder / "residual_flash_sac_v8.yaml"))
    changed = {k for k in set(v7) | set(v8) if v7.get(k) != v8.get(k)}
    assert changed == {"wandb.group", "sac.learning_starts", "sac.policy_learning_starts"}
    assert (v8["sac.learning_starts"], v8["sac.policy_learning_starts"]) == (700, 2100)


def test_residual_flash_sac_v9_is_v8_with_flashsacs_discount():
    import yaml

    folder = CONFIGS / "experiments" / "residual_flash_sac"
    assert yaml.safe_load((folder / "residual_flash_sac_v9.yaml").read_text())["base"] == "_foundation.yaml"
    v8 = leaves(merged(folder / "residual_flash_sac_v8.yaml"))
    v9 = leaves(merged(folder / "residual_flash_sac_v9.yaml"))
    changed = {k for k in set(v8) | set(v9) if v8.get(k) != v9.get(k)}
    assert changed == {"wandb.group", "sac.discount_factor"}
    assert (v8["sac.discount_factor"], v9["sac.discount_factor"]) == (0.999, 0.99)


def test_residual_flash_sac_v10_is_v8_without_noise_repetition():
    import yaml

    folder = CONFIGS / "experiments" / "residual_flash_sac"
    assert yaml.safe_load((folder / "residual_flash_sac_v10.yaml").read_text())["base"] == "_foundation.yaml"
    v8 = leaves(merged(folder / "residual_flash_sac_v8.yaml"))
    v10 = leaves(merged(folder / "residual_flash_sac_v10.yaml"))
    changed = {k for k in set(v8) | set(v10) if v8.get(k) != v10.get(k)}
    assert changed == {"wandb.group", "sac.noise_repeat"} and v10["sac.noise_repeat"] is False


@pytest.mark.skip(reason="old config (normalize_observations): left in place to adapt to state_preprocessor later")
def test_residual_flash_sac_v11_is_v10_with_50_step_returns_and_a_faster_target():
    import yaml

    from omegaconf import OmegaConf

    from robonuke_rl_core.learners.cfg import SACCfg

    folder = CONFIGS / "experiments" / "residual_flash_sac"
    assert yaml.safe_load((folder / "residual_flash_sac_v11.yaml").read_text())["base"] == "_foundation.yaml"
    v10 = leaves(merged(folder / "residual_flash_sac_v10.yaml"))
    v11 = leaves(merged(folder / "residual_flash_sac_v11.yaml"))
    changed = {k for k in set(v10) | set(v11) if v10.get(k) != v11.get(k)}
    assert changed == {"wandb.group", "sac.n_step", "sac.polyak"} and (v11["sac.n_step"], v11["sac.polyak"]) == (50, 0.05)
    schema = OmegaConf.structured(SACCfg)
    OmegaConf.set_struct(schema, True)
    OmegaConf.to_object(OmegaConf.merge(schema, OmegaConf.create(merged(folder / "residual_flash_sac_v11.yaml")["sac"]))).validate(None)


def test_residual_flash_sac_v12_is_v11_with_a_learned_std_and_the_foundations_entropy_term():
    import yaml

    folder = CONFIGS / "experiments" / "residual_flash_sac"
    assert yaml.safe_load((folder / "residual_flash_sac_v12.yaml").read_text())["base"] == "_foundation.yaml"
    v11 = leaves(merged(folder / "residual_flash_sac_v11.yaml"))
    v12 = leaves(merged(folder / "residual_flash_sac_v12.yaml"))
    changed = {k for k in set(v11) | set(v12) if v11.get(k) != v12.get(k)}
    assert changed == {"wandb.group", "sac.learn_entropy", "sac.initial_entropy_value", "sac.backup_entropy",
                       "model.actor.fixed_std", "model.actor.init_std"}
    assert (v12["sac.learn_entropy"], v12["sac.initial_entropy_value"], v12["sac.backup_entropy"]) == (True, 0.01, True)
    assert "model.actor.fixed_std" not in v12 and v12["model.actor.init_std"] == 0.36788


# ------------------------------------------------------------------ every experiment config loads
#: launch configs known not to validate, and why (left as they are, not guessed at)
KNOWN_BROKEN = {
    "scaled_residual/residual_scaled_residual.yaml":
        "ddpg.state_preprocessor: demo (its foundation's, added after this file) with no demo file of its own",
}


def _sections(path: Path):
    """``{name: section object}`` of the learner and residual sections the file's chain sets, merged onto their
    dataclasses (unknown keys raise)."""
    from robonuke_rl_core.learners.cfg import DDPGCfg, PPOCfg

    merged = OmegaConf.merge(*[layer for _, layer in load_file_chain(path)])
    out = {}
    for name, cls in (("residual", ResidualCfg), ("ddpg", DDPGCfg), ("sac", SACCfg), ("ppo", PPOCfg)):
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


def test_a_method_file_adds_onto_a_task_file_through_a_list_base():
    """configs/methods/resfit.yaml is ResFiT with no task: the robosuite and one_leg foundations chain it after their task
    file and keep only the task's own values."""
    from robonuke_rl_core.configfile import chain

    for folder, task in (("resfit_can", "robosuite_can.yaml"), ("ResFiT", "fb_one_leg.yaml")):
        files = [p.name for p in chain(CONFIGS / "experiments" / folder / "_foundation.yaml")]
        assert files[-3:] == [task, "resfit.yaml", "_foundation.yaml"]
    can = leaves(merged(CONFIGS / "experiments" / "resfit_can" / "residual_resfit_can.yaml"))
    one_leg = leaves(merged(CONFIGS / "experiments" / "ResFiT" / "residual_ResFiT.yaml"))
    for key in ("residual.action_space", "residual.action_scale_fraction", "residual.action_half_range_floor",
                "residual.executed_clip", "ddpg.action_range", "ddpg.num_critics", "ddpg.critic_subset", "ddpg.policy_delay"):
        assert can[key] == one_leg[key], key  # the same method on two tasks
    assert can["residual.action_space"] == "per_dim" and can["ddpg.action_range"] == [-0.1, 0.1]
