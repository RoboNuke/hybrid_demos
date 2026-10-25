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


def test_critic_full_action_is_scale002_but_for_its_std_its_temperature_its_critic_action_and_its_own_group_and_demos():
    parent, new = (leaves(merged(GROUP / f"{stem}.yaml")) for stem in RESIDUAL_LAUNCHES[4:6])
    diff = {k for k in set(parent) | set(new) if parent.get(k, "<unset>") != new.get(k, "<unset>")}
    assert diff == {"model.actor.act_init_std", "model.actor.min_log_std", "model.actor.max_log_std",
                    "sac.initial_entropy_value", "residual.critic_action", "wandb.group", "sac.demo_transitions"}
    for bound in ("min_log_std", "max_log_std"):
        assert math.exp(new[f"model.actor.{bound}"]) == pytest.approx(5 * math.exp(parent[f"model.actor.{bound}"]), rel=0.01)
    assert new["model.actor.act_init_std"] == pytest.approx(5 * parent["model.actor.act_init_std"])
    assert new["sac.learn_entropy"] is False
    assert new["sac.initial_entropy_value"] == pytest.approx(parent["sac.initial_entropy_value"] / 25)
    assert "residual.critic_action" not in parent and ResidualCfg().critic_action == "residual"  # the default
    assert new["residual.critic_action"] == "full"


def test_nstep10_is_critic_full_action_at_10_percent_with_its_std_and_alpha_rescaled_and_10_step_targets():
    parent, new = (leaves(merged(GROUP / f"{stem}.yaml")) for stem in RESIDUAL_LAUNCHES[5:7])
    diff = {k for k in set(parent) | set(new) if parent.get(k, "<unset>") != new.get(k, "<unset>")}
    assert diff == {"residual.action_scale_fraction", "model.actor.act_init_std", "model.actor.min_log_std",
                    "model.actor.max_log_std", "sac.initial_entropy_value", "sac.n_step", "wandb.group",
                    "sac.demo_transitions"}
    ratio = new["residual.action_scale_fraction"] / parent["residual.action_scale_fraction"]
    assert ratio == pytest.approx(5.0)
    # the same physical noise: std / ratio. The alpha is what the stopped run had: the 10% runs' 0.01 x 10 x
    # ratio^2, 25x the parent's physical pull (the _alpha0p1 child has the intended one)
    for bound in ("min_log_std", "max_log_std"):
        assert math.exp(new[f"model.actor.{bound}"]) == pytest.approx(math.exp(parent[f"model.actor.{bound}"]) / ratio, rel=0.01)
    assert new["model.actor.act_init_std"] == pytest.approx(parent["model.actor.act_init_std"] / ratio)
    assert new["sac.initial_entropy_value"] == pytest.approx(0.01 * 10 * ratio ** 2)
    assert new["sac.learn_entropy"] is False
    assert new["sac.n_step"] == 10
    assert new["residual.critic_action"] == "full"


def test_alpha0p1_is_nstep10_with_the_parents_physical_pull_on_the_mean():
    """The 2% parent's alpha 4e-4 x 25 (the authority ratio squared: the pull on the mean is 2 alpha tanh(u) in
    r, the critic's push scales with the authority) x 10 (a success pays 1, not 0.1) = 0.1; nothing else."""
    one_step, parent, new = (leaves(merged(GROUP / f"{stem}.yaml")) for stem in RESIDUAL_LAUNCHES[5:8])
    diff = {k for k in set(parent) | set(new) if parent.get(k, "<unset>") != new.get(k, "<unset>")}
    assert diff == {"sac.initial_entropy_value", "wandb.group", "sac.demo_transitions"}
    ratio = new["residual.action_scale_fraction"] / one_step["residual.action_scale_fraction"]
    assert new["sac.initial_entropy_value"] == pytest.approx(one_step["sac.initial_entropy_value"] * ratio ** 2 * 10)
    assert new["sac.initial_entropy_value"] == pytest.approx(0.1)
    assert new["sac.learn_entropy"] is False


def test_alpha0p01_is_alpha0p1_with_alpha_over_10():
    """The one change the alpha0p1 run's failure led to: alpha / 10, nothing else (alpha enters only the actor)."""
    parent, new = (leaves(merged(GROUP / f"{stem}.yaml")) for stem in RESIDUAL_LAUNCHES[7:9])
    diff = {k for k in set(parent) | set(new) if parent.get(k, "<unset>") != new.get(k, "<unset>")}
    assert diff == {"sac.initial_entropy_value", "wandb.group", "sac.demo_transitions"}
    assert new["sac.initial_entropy_value"] == pytest.approx(parent["sac.initial_entropy_value"] / 10)
    assert new["sac.learn_entropy"] is False and new["sac.backup_entropy"] is False


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


# ------------------------------------------------- the interface relative to the end-effector
REL_GROUP = CONFIGS / "experiments" / "resip_replica_rel"


def test_the_relative_base_is_fb_one_leg_with_only_the_task_name_changed():
    ab, rel = leaves(merged(CONFIGS / "base" / "fb_one_leg.yaml")), leaves(merged(CONFIGS / "base" / "fb_one_leg_rel.yaml"))
    assert {k for k in set(ab) | set(rel) if ab.get(k) != rel.get(k)} == {"task.name"}
    assert rel["task.name"] == "Isaac-Furniture-OneLeg-FBRel-ResiP-v0"


def test_the_relative_bc_is_eval_happys_bc_on_the_relative_columns_for_150k_steps():
    """bc_resip_replica_rel = bc_resip_replica (what physic_aligned_evalHappy_a0 trained with) on the
    relative task and the relative columns of the same 50 demos, the action used as it is, 150k gradient steps; the residual-only
    sections of resip_replica's foundation are not carried over (BC reads none of them)."""
    rl_only = ("sac.", "memory.", "residual.", "history.")
    ab = {k: v for k, v in leaves(merged(GROUP / "bc_resip_replica.yaml")).items() if not k.startswith(rl_only)}
    rel = leaves(merged(REL_GROUP / "bc_resip_replica_rel.yaml"))
    diff = {k for k in set(ab) | set(rel) if ab.get(k, "<unset>") != rel.get(k, "<unset>")}
    assert diff == {
        "task.name", "bc.dataset", "bc.obs_key", "bc.action_key", "bc.normalize_action", "trainer.total_timesteps", "wandb.group",
    }
    assert rel["bc.normalize_action"] is False  # the relative action is already in [-1, 1]: zero holds the EE
    assert rel["bc.dataset"] == "data/physic_aligned_rel" and rel["bc.obs_key"] == "observation.state_rel"
    assert rel["bc.action_key"] == "action.rel" and rel["trainer.total_timesteps"] == 150000
    assert rel["trainer.total_timesteps"] % rel["bc.steps_per_epoch"] == 0
    assert rel["trainer.total_timesteps"] % rel["trainer.checkpoint_interval"] == 0


DELTA_GROUP = CONFIGS / "experiments" / "policy_dec_replica"


def test_the_delta_base_is_fb_one_leg_with_only_the_task_name_changed():
    ab, delta = leaves(merged(CONFIGS / "base" / "fb_one_leg.yaml")), leaves(merged(CONFIGS / "base" / "fb_one_leg_delta.yaml"))
    assert {k for k in set(ab) | set(delta) if ab.get(k) != delta.get(k)} == {"task.name"}
    assert delta["task.name"] == "Isaac-Furniture-OneLeg-FBDelta-ResiP-v0"


def test_policy_dec_replica_is_eval_happys_bc_on_the_delta_action_for_150k_steps():
    """bc_policy_dec_replica = bc_resip_replica (what physic_aligned_evalHappy_a0 trained with) on the
    delta task: the same observation column of the same 50 demos, the delta action used as it is, 150k
    gradient steps; the residual-only sections of resip_replica's foundation are not carried over."""
    rl_only = ("sac.", "memory.", "residual.", "history.")
    ab = {k: v for k, v in leaves(merged(GROUP / "bc_resip_replica.yaml")).items() if not k.startswith(rl_only)}
    delta = leaves(merged(DELTA_GROUP / "bc_policy_dec_replica.yaml"))
    diff = {k for k in set(ab) | set(delta) if ab.get(k, "<unset>") != delta.get(k, "<unset>")}
    assert diff == {"task.name", "bc.dataset", "bc.action_key", "bc.normalize_action", "trainer.total_timesteps", "wandb.group"}
    assert delta["task.name"] == "Isaac-Furniture-OneLeg-FBDelta-ResiP-v0" and delta["wandb.group"] == "bc_policy_dec_replica"
    assert delta["bc.dataset"] == "data/physic_aligned_delta" and delta["bc.obs_key"] == "observation.state_abs"
    assert delta["bc.action_key"] == "action.delta" and delta["bc.normalize_action"] is False
    assert delta["trainer.total_timesteps"] == 150000


CHUNK_GROUP = CONFIGS / "experiments" / "policy_dec_replica_chunk"


def test_policy_dec_replica_chunk_is_eval_happys_bc_with_the_chunk_in_the_ee_frame_of_its_observation():
    """bc_policy_dec_replica_chunk = bc_resip_replica (what physic_aligned_evalHappy_a0 trained with) on the
    same task, demos and columns, the network predicting its chunk in the EE frame of the observation, in m
    and rad, unscaled and unclipped, 150k gradient steps."""
    rl_only = ("sac.", "memory.", "residual.", "history.")
    ab = {k: v for k, v in leaves(merged(GROUP / "bc_resip_replica.yaml")).items() if not k.startswith(rl_only)}
    chunk = leaves(merged(CHUNK_GROUP / "bc_policy_dec_replica_chunk.yaml"))
    diff = {k for k in set(ab) | set(chunk) if ab.get(k, "<unset>") != chunk.get(k, "<unset>")}
    assert diff == {
        "model.actor.action_frame", "model.actor.clip_sample", "bc.normalize_action", "trainer.total_timesteps", "wandb.group",
    }
    assert chunk["model.actor.action_frame"] == "ee_at_obs" and chunk["model.actor.clip_sample"] is False
    assert chunk["bc.normalize_action"] is False and chunk["task.name"] == "Isaac-Furniture-OneLeg-FBAbs-ResiP-v0"
    assert (chunk["bc.dataset"], chunk["bc.action_key"]) == ("data/physic_aligned", "action.abs")
    assert chunk["trainer.total_timesteps"] == 150000 and chunk["wandb.group"] == "bc_policy_dec_replica_chunk"


def test_the_zero_centred_chunk_config_is_the_chunk_config_with_symmetric_action_scaling():
    """bc_policy_dec_replica_chunk_zero_centered_norm = bc_policy_dec_replica_chunk with the action scaled per
    dim by max(|min|, |max|) (zero stays zero) and clipped to [-1, 1] as the reference clips."""
    chunk = leaves(merged(CHUNK_GROUP / "bc_policy_dec_replica_chunk.yaml"))
    zero = leaves(merged(CONFIGS / "experiments" / "policy_dec_replica_chunk_zero_centered_norm" / "bc_policy_dec_replica_chunk_zero_centered_norm.yaml"))
    diff = {k for k in set(chunk) | set(zero) if chunk.get(k, "<unset>") != zero.get(k, "<unset>")}
    assert diff == {"bc.normalize_action", "bc.symmetric_action", "model.actor.clip_sample", "wandb.group"}
    assert zero["bc.normalize_action"] is True and zero["bc.symmetric_action"] is True
    assert zero.get("model.actor.clip_sample", "<unset>") == "<unset>"  # the default: clipped to [-1, 1]
    assert zero["wandb.group"] == "bc_policy_dec_replica_chunk_zero_centered_norm"


def test_the_ee_accumulating_residual_configs_change_only_the_residual_action():
    """A + B on the alpha 0.01 run, with demos (regenerated for the EE action) and without (the no-demo test's
    config); the std and every other learning parameter as there."""
    ee = {"residual.action_frame": "ee", "residual.ee_pos_scale": 0.01, "residual.ee_rot_scale_deg": 5.0, "residual.accumulate": True}
    for name, ref in (("", "residual_serl_g0995_qmin_full_action_nstep10_scale01_std01_alpha0p01.yaml"),
                      ("_no_demo_critic", "residual_serl_g0995_qmin_full_action_nstep10_scale01_std01_alpha0p01_no_demo_critic.yaml")):
        base = leaves(merged(GROUP / ref))
        new = leaves(merged(GROUP / f"residual_serl_g0995_qmin_full_action_nstep10_scale01_std01_alpha0p01_ee_accum{name}.yaml"))
        diff = {k for k in set(base) | set(new) if base.get(k, "<unset>") != new.get(k, "<unset>")}
        assert diff == set(ee) | {"wandb.group"} | ({"sac.demo_transitions"} if not name else set()), name
        assert all(new[k] == v for k, v in ee.items())


def test_the_low_std_ee_config_changes_only_the_std():
    ref = leaves(merged(GROUP / "residual_serl_g0995_qmin_full_action_nstep10_scale01_std01_alpha0p01_ee_accum.yaml"))
    new = leaves(merged(GROUP / "residual_serl_g0995_qmin_full_action_nstep10_scale01_std01_alpha0p01_ee_accum_std0004.yaml"))
    diff = {k for k in set(ref) | set(new) if ref.get(k, "<unset>") != new.get(k, "<unset>")}
    assert diff == {"model.actor.act_init_std", "model.actor.min_log_std", "model.actor.max_log_std", "wandb.group"}
    assert new["sac.demo_transitions"] == ref["sac.demo_transitions"]


def test_the_resip_ppo_replica_is_resips_residual_recipe_on_our_base():
    c = leaves(merged(CONFIGS / "experiments" / "resip_ppo_replica" / "residual_resip_ppo_replica.yaml"))
    assert c["trainer.learner"] == "ppo" and c["residual.enabled"] is True and c["model.architecture"] == "mlp"
    assert (c["ppo.rollouts"], c["ppo.learning_epochs"], c["ppo.mini_batches"], c["ppo.discount_factor"], c["ppo.gae_lambda"]) == (700, 50, 1, 0.999, 0.95)
    assert (c["ppo.kl_threshold"], c["ppo.grad_norm_clip"], c["ppo.policy_lr"], c["ppo.value_lr"], c["ppo.value_clip"]) == (0.1, 1.0, 3.0e-4, 5.0e-3, 0.0)
    assert c["ppo.normalize_rewards"] is True and c["ppo.reward_clip"] == 5.0 and c["ppo.normalize_values"] is False
    assert c["ppo.reset_every_iteration"] is True and c["ppo.time_limit_bootstrap"] is False and c["ppo.normalize_observations"] is False
    assert (c["ppo.lr_schedule"], c["ppo.policy_lr_warmup"], c["ppo.value_lr_warmup"]) == ("cosine", 5, 0)
    assert (c["model.actor.hidden_size"], c["model.actor.num_layers"], c["model.actor.activation"]) == (512, 2, "SiLU")
    assert c["model.actor.init_log_std"] == -1.0 and c["model.actor.learn_std"] is False and c["model.actor.squash"] is False
    assert c["model.actor.output_init_scale"] == 0.01 and (c["model.critic.hidden_size"], c["model.critic.activation"]) == (512, "SiLU")
    assert c["task.cfg.scene.num_envs"] == 1024 and c["task.cfg.episode_length_s"] == 70.0
    assert c["trainer.total_timesteps"] % c["ppo.rollouts"] == 0 and c["trainer.checkpoint_interval"] == 5 * c["ppo.rollouts"]
    assert c["residual.steps_per_chunk"] == 1 and c["residual.normalize_observation"] is True and c["residual.observe_base_action"] is True
    assert c["residual.action_scale_fraction"] == 0.1 and c["residual.critic_action"] == "residual"
    assert c.get("history.enabled", False) is False


def test_the_policy_decorator_replica_is_their_defaults_on_our_base():
    c = leaves(merged(CONFIGS / "experiments" / "policy_dec_sac_replica" / "residual_policy_dec_sac_replica.yaml"))
    assert c["trainer.learner"] == "sac" and c["residual.enabled"] is True and c["model.architecture"] == "mlp"
    assert (c["residual.steps_per_chunk"], c["residual.base_action_horizon"], c["residual.base_inference_steps"]) == (4, 4, 4)
    # their actor_input "obs": the raw observation alone; the critic takes the executed chunk (their "sum")
    assert c["residual.normalize_observation"] is False and c["residual.observe_base_action"] is False
    assert c["residual.critic_action"] == "full" and c["residual.action_scale_fraction"] == 0.1
    assert (c["sac.discount_factor"], c["sac.polyak"], c["sac.batch_size"], c["sac.actor_lr"], c["sac.critic_lr"]) == (0.97, 0.01, 1024, 1.0e-4, 1.0e-4)
    assert c["sac.learn_entropy"] is True and c["sac.target_entropy"] == -40.0 and c["sac.backup_entropy"] is True
    assert (c["sac.num_critics"], c["sac.critic_subset"], c["sac.policy_delay"], c["sac.n_step"]) == (2, 2, 1, 1)
    # UTD 0.25: 16 updates per 4 env steps x 16 envs
    assert c["sac.gradient_steps"] / (c["sac.update_every"] * c["task.cfg.scene.num_envs"]) == 0.25
    assert (c["sac.random_timesteps"], c["sac.learning_starts"]) == (8000 // 16, 8000 // 16)
    assert c["sac.progressive_exploration_steps"] == 100_000 // 16 and c["sac.grad_norm_clip"] == 50.0
    assert c["sac.demo_transitions"] is None and c["sac.demo_sampling_end"] == 0 and c["sac.normalize_observations"] is False
    assert c["task.cfg.rewards.step.weight"] * 4 == -1.0 and "task.cfg.rewards.assembled.weight" not in c  # assembled stays the task's 1.0
    assert (c["model.actor.hidden_size"], c["model.actor.num_layers"], c["model.actor.activation"]) == (256, 3, "ReLU")
    assert c["model.actor.use_state_dependent_std"] is True and c["model.actor.log_std_bound"] == "tanh"
    assert (c["model.actor.min_log_std"], c["model.actor.max_log_std"], c["model.actor.squash"]) == (-20.0, 2.0, True)
    assert (c["model.critic.num_layers"], c["model.critic.output_init_scale"]) == (3, 0.01)
    assert c["trainer.total_timesteps"] * c["task.cfg.scene.num_envs"] == 4_000_000
    assert c["memory.memory_size"] == c["trainer.total_timesteps"] * c["task.cfg.scene.num_envs"]
    assert c.get("history.enabled", False) is False


def test_the_policy_decorator_replica_validates():
    """critic_action: full with an actor that does not observe the base action: the critic state carries it."""
    c = merged(CONFIGS / "experiments" / "policy_dec_sac_replica" / "residual_policy_dec_sac_replica.yaml")
    rc = ResidualCfg(**{k: v for k, v in c["residual"].items() if k in ResidualCfg.__dataclass_fields__})
    rc.validate(None)  # the residual's own rules (the env-dependent ones need a loaded task)
    assert rc.critic_action == "full" and rc.observe_base_action is False and rc.steps_per_chunk == 4
