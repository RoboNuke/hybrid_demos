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
    # the raw observation (their actor_input adds the base chunk below); the critic takes the executed chunk (their "sum")
    assert c["residual.normalize_observation"] is False
    assert c["residual.critic_action"] == "full" and c["residual.action_scale_fraction"] == 0.1
    # their critic sees the observation alone; their pd_ee_delta_pose residual: 0.01 m / 0.01 rad per unit, 4 x 7
    assert c["residual.critic_state"] == "observation" and c["residual.action_frame"] == "target"
    # the base chunk observed, and the critic's executed chunk, in their EE-relative command form
    assert c["residual.observe_base_action"] is True and c["residual.ee_relative_actions"] is True
    assert c["model.actor.squash"] is False and c["sac.unsquashed_actor"] is True  # a plain Gaussian actor
    assert (c["residual.ee_relative_pos_unit"], c["residual.ee_relative_rot_unit"]) == (0.1, 0.1)
    assert c["residual.ee_pos_scale"] == 0.01 and abs(c["residual.ee_rot_scale_deg"] - math.degrees(0.01)) < 1e-12
    assert (c["sac.discount_factor"], c["sac.polyak"], c["sac.batch_size"], c["sac.actor_lr"], c["sac.critic_lr"]) == (0.97, 0.01, 1024, 1.0e-4, 1.0e-4)
    assert c["sac.learn_entropy"] is True and c["sac.target_entropy"] == -28.0 and c["sac.backup_entropy"] is True
    assert (c["sac.num_critics"], c["sac.critic_subset"], c["sac.policy_delay"], c["sac.n_step"]) == (2, 2, 1, 1)
    # UTD 0.25: 16 updates per 4 env steps x 16 envs
    assert c["sac.gradient_steps"] / (c["sac.update_every"] * c["task.cfg.scene.num_envs"]) == 0.25
    assert (c["sac.random_timesteps"], c["sac.learning_starts"]) == (8000 // 16, 8000 // 16)
    # theirs is 100k global steps (6250 per env); ours ramps over 50k per env, for 5x longer episodes
    assert c["sac.progressive_exploration_steps"] == 50_000 and c["sac.grad_norm_clip"] == 50.0
    assert c["sac.demo_transitions"] is None and c["sac.demo_sampling_end"] == 0 and c["sac.normalize_observations"] is False
    assert c["task.cfg.rewards.step.weight"] == 0.0 and "task.cfg.rewards.assembled.weight" not in c  # no step penalty; assembled stays the task's 1.0
    assert [c[f"task.cfg.rewards.{p}_done.weight"] for p in ("corner", "grasp", "insert")] == [1.0, 1.0, 1.0]
    assert c["task.cfg.furniture.track_phases"] is True  # the phase rewards' tracker (configs/base/fb_one_leg.yaml)
    assert (c["model.actor.hidden_size"], c["model.actor.num_layers"], c["model.actor.activation"]) == (256, 3, "ReLU")
    assert c["model.actor.use_state_dependent_std"] is True and c["model.actor.log_std_bound"] == "tanh"
    # their initial log std: the tanh bound's midpoint, -20 + 0.5 * 22 * (0 + 1)
    assert c["model.actor.init_log_std"] == -20.0 + 0.5 * (2.0 - -20.0) * 1.0 == -9.0
    assert (c["model.actor.min_log_std"], c["model.actor.max_log_std"], c["model.actor.squash"]) == (-20.0, 2.0, False)
    assert (c["model.critic.num_layers"], c["model.critic.output_init_scale"]) == (3, 0.01)
    assert c["trainer.total_timesteps"] * c["task.cfg.scene.num_envs"] == 4_000_000
    assert c["memory.memory_size"] == c["trainer.total_timesteps"] * c["task.cfg.scene.num_envs"]
    assert c.get("history.enabled", False) is False


def test_the_policy_decorator_replica_validates():
    """critic_action: full with the base chunk and the executed chunk in the EE-relative command form."""
    c = merged(CONFIGS / "experiments" / "policy_dec_sac_replica" / "residual_policy_dec_sac_replica.yaml")
    rc = ResidualCfg(**{k: v for k, v in c["residual"].items() if k in ResidualCfg.__dataclass_fields__})
    rc.validate(None)  # the residual's own rules (the env-dependent ones need a loaded task)
    assert rc.critic_action == "full" and rc.observe_base_action is True and rc.ee_relative_actions and rc.steps_per_chunk == 4


def test_the_flashsac_pd_config_is_b_with_every_flashsac_piece():
    """flashsac_pd: B's residual, FlashSAC's learner and network (Table 9, its IsaacLab script), 16 envs."""
    from types import SimpleNamespace

    from robonuke_rl_core.learners.cfg import SACCfg
    from robonuke_rl_core.models.cfg import FlashSACActorCfg, FlashSACCriticCfg, FlashSACModelCfg

    path = CONFIGS / "experiments" / "flashsac_pd" / "residual_flashsac_pd.yaml"
    c = merged(path)
    flat = leaves(c)
    sac = SACCfg(**c["sac"])
    sac.validate(None)
    model = FlashSACModelCfg(actor=FlashSACActorCfg(**c["model"]["actor"]), critic=FlashSACCriticCfg(**c["model"]["critic"]))
    model.validate(SimpleNamespace(trainer=SimpleNamespace(learner=c["trainer"]["learner"])))
    rc = ResidualCfg(**{k: v for k, v in c["residual"].items() if k in ResidualCfg.__dataclass_fields__})
    rc.validate(None)
    # FlashSAC
    assert flat["model.architecture"] == "flashsac"
    assert (flat["model.actor.hidden_size"], flat["model.actor.num_blocks"], flat["model.critic.hidden_size"], flat["model.critic.num_blocks"]) == (128, 2, 256, 2)
    assert (sac.critic_distribution, sac.critic_num_bins, sac.critic_min_value, sac.critic_max_value) == ("categorical", 101, -5.0, 5.0)
    assert sac.adaptive_reward_scaling and sac.adaptive_reward_g_max == 5.0
    assert sac.weight_normalization and sac.cross_batch_value and sac.noise_repeat
    assert (sac.noise_repeat_zeta_s, sac.noise_repeat_max) == (2.0, 16)
    assert sac.target_entropy is None and sac.target_entropy_sigma == 0.15 and sac.initial_entropy_value == 0.01
    assert sac.entropy_loss_form == "alpha" and sac.policy_delay == 2 and sac.num_critics == 2 and sac.polyak == 0.01
    assert (sac.batch_size, sac.gradient_steps, sac.update_every) == (2048, 2, 1)
    assert (sac.lr_schedule, sac.actor_lr, sac.critic_lr, sac.lr_end) == ("cosine", 3.0e-4, 3.0e-4, 1.5e-4)
    assert (sac.discount_factor, sac.n_step, sac.learning_starts, sac.random_timesteps) == (0.99, 3, 500, 0)
    assert sac.grad_norm_clip == 0.0 and sac.normalize_observations is False and sac.unsquashed_actor is False
    assert flat["model.actor.mean_init_scale"] == 0.01  # the residual starts near zero
    assert flat["model.actor.init_std"] == 0.15  # the state-dependent std starts at the entropy target
    assert sac.std_by_label == "eval_state/phase/id" and sac.std_label_names == ["corner", "grasp", "insert", "screw"]
    assert sac.metrics_by_label == ["residual/position_mm", "residual/rot_deg"]
    assert sac.bootstrap_time_limits is False  # a reset's observation is never bootstrapped from
    assert flat["trainer.best_by"] == "success"  # ckpt_best is the most successful interval's
    # B, unchanged
    assert flat["task.cfg.scene.num_envs"] == 16 and flat["trainer.total_timesteps"] == 250000
    assert rc.action_frame == "target" and rc.critic_state == "observation" and rc.ee_relative_actions and rc.steps_per_chunk == 4
    assert [flat[f"task.cfg.rewards.{p}_done.weight"] for p in ("corner", "grasp", "insert")] == [1.0, 1.0, 1.0]
    assert sac.progressive_exploration_steps == 50000


def test_flashsac_pd_v4_is_flashsac_pd_with_small_constant_learning_rates():
    base = leaves(merged(CONFIGS / "experiments" / "flashsac_pd" / "residual_flashsac_pd.yaml"))
    low = leaves(merged(CONFIGS / "experiments" / "flashsac_pd" / "residual_flashsac_pd_lowlr.yaml"))
    assert (low["sac.actor_lr"], low["sac.critic_lr"], low["sac.entropy_lr"]) == (1.0e-5, 1.0e-4, 1.0e-5)
    assert low["sac.lr_schedule"] == "constant" and low["wandb.group"] == "flashsac_pd_v4"
    changed = {k for k in set(base) | set(low) if base.get(k) != low.get(k)}
    assert changed == {"sac.actor_lr", "sac.critic_lr", "sac.entropy_lr", "sac.lr_schedule", "wandb.group"}


def test_the_parts_configs_change_only_what_they_say():
    from robonuke_rl_core.learners.cfg import SACCfg

    folder = CONFIGS / "experiments" / "flashsac_pd"
    v3 = leaves(merged(folder / "residual_flashsac_pd.yaml"))
    parts = leaves(merged(folder / "residual_flashsac_pd_parts.yaml"))
    grasp = leaves(merged(folder / "residual_flashsac_pd_parts_grasp.yaml"))
    changed = {k for k in set(v3) | set(parts) if v3.get(k) != parts.get(k)}
    assert changed == {"sac.success_clone_weight", "sac.failure_anchor_weight", "sac.outcome_metric", "sac.actor_q_scale", "wandb.group"}
    assert (parts["sac.success_clone_weight"], parts["sac.failure_anchor_weight"], parts["sac.actor_q_scale"]) == (1.0, 1.0, 2.5)
    changed = {k for k in set(parts) | set(grasp) if parts.get(k) != grasp.get(k)}
    assert changed == {"residual.active_from_phase", "residual.ee_rot_scale_deg", "sac.store_labels", "wandb.group"}
    assert grasp["residual.ee_rot_scale_deg"] == 6.0  # 6 deg per unit r per rotation axis
    assert grasp["residual.active_from_phase"] == "insert"
    assert grasp["sac.store_labels"] == ["insert", "screw"]  # the pre-grasp steps never enter the memory
    for path in (folder / "residual_flashsac_pd_parts.yaml", folder / "residual_flashsac_pd_parts_grasp.yaml"):
        c = merged(path)
        SACCfg(**c["sac"]).validate(None)
        ResidualCfg(**{k: v for k, v in c["residual"].items() if k in ResidualCfg.__dataclass_fields__}).validate(None)


def test_the_parts_pd_config_is_the_last_flashsac_run_on_one_env_with_per_phase_residuals():
    """parts_pd: residual_flashsac_pd_parts_grasp on 1 env, one residual per phase, no chunk, fixed std, no entropy,
    no exploration ramp, the demos as replay memory rows, no retraining."""
    from robonuke_rl_core.learners.cfg import SACCfg
    from robonuke_rl_core.models.cfg import FlashSACActorCfg, FlashSACCriticCfg, FlashSACModelCfg

    grasp = leaves(merged(CONFIGS / "experiments" / "flashsac_pd" / "residual_flashsac_pd_parts_grasp.yaml"))
    path = CONFIGS / "experiments" / "parts_pd" / "residual_parts_pd.yaml"
    parts = leaves(merged(path))
    changed = {k for k in set(grasp) | set(parts) if grasp.get(k) != parts.get(k)}
    assert changed == {
        "wandb.group", "task.cfg.scene.num_envs", "trainer.total_timesteps", "trainer.checkpoint_interval",
        "trainer.write_interval", "trainer.success_window", "trainer.windowed_metrics", "sac.noise_repeat",
        "residual.active_from_phase", "residual.steps_per_chunk", "sac.store_labels", "sac.route_labels",
        "sac.std_label_names", "sac.discount_factor", "sac.batch_size", "sac.update_every", "sac.learn_entropy",
        "sac.initial_entropy_value", "sac.target_entropy_sigma", "sac.backup_entropy",
        "sac.progressive_exploration_steps", "sac.demo_transitions", "sac.demo_storage",
        "memory.memory_size", "model.actor.init_std", "model.actor.fixed_std",
        # each residual's outcome is its local one (route_label): outcome_metric is not read
        "sac.outcome_metric",
    }
    assert parts["sac.route_labels"] == ["insert", "screw"] and parts["task.cfg.scene.num_envs"] == 1
    assert parts["residual.steps_per_chunk"] == 1 and parts["residual.base_action_horizon"] == 4
    assert parts["model.actor.fixed_std"] == 0.15
    assert (parts["sac.learn_entropy"], parts["sac.initial_entropy_value"]) == (False, 0.0)
    assert "{route}" in parts["sac.demo_transitions"] and parts["sac.demo_storage"] == "memory"
    assert parts["sac.noise_repeat"] is False and parts["sac.discount_factor"] == 0.999
    assert "sac.retrain_every" not in parts and parts["trainer.checkpoint_interval"] == 10000
    assert (parts["trainer.write_interval"], parts["trainer.success_window"]) == (1000, 10)
    assert parts["trainer.windowed_metrics"] == ["phase/passed_*"]
    c = merged(path)
    SACCfg(**c["sac"]).validate(None)
    ResidualCfg(**{k: v for k, v in c["residual"].items() if k in ResidualCfg.__dataclass_fields__}).validate(None)
    FlashSACModelCfg(actor=FlashSACActorCfg(**c["model"]["actor"]), critic=FlashSACCriticCfg(**c["model"]["critic"])).validate(None)


def test_the_parts_pd_v2_config_changes_only_what_it_says():
    """parts_pd v2: no-progress resets with held transitions, Q(s, r), reference dropout, per-phase clocks, noise
    held at most 5 steps, retraining every 10k phase steps."""
    from robonuke_rl_core.learners.cfg import SACCfg

    folder = CONFIGS / "experiments" / "parts_pd"
    v1 = leaves(merged(folder / "residual_parts_pd.yaml"))
    path = folder / "residual_parts_pd_v2.yaml"
    v2 = leaves(merged(path))
    changed = {k for k in set(v1) | set(v2) if v1.get(k) != v2.get(k)}
    assert changed == {
        "wandb.group", "task.cfg.furniture.stall_steps_insert", "task.cfg.furniture.stall_steps_screw",
        "task.cfg.furniture.stall_pos_mm", "task.cfg.furniture.stall_rot_deg", "residual.critic_action",
        "sac.route_clock", "sac.hold_still_key", "sac.hold_stall_key", "sac.reference_dropout", "sac.reference_columns",
        "sac.noise_repeat", "sac.noise_repeat_max", "sac.retrain_every", "sac.retrain_failure_fraction",
        "sac.retrain_gradient_steps",
    }
    assert (v2["task.cfg.furniture.stall_steps_insert"], v2["task.cfg.furniture.stall_steps_screw"]) == (30, 120)
    assert v2["residual.critic_action"] == "residual" and v2["residual.critic_state"] == "observation"
    assert (v2["sac.reference_dropout"], v2["sac.reference_columns"]) == (0.3, 7)
    assert (v2["sac.noise_repeat"], v2["sac.noise_repeat_max"]) == (True, 5)
    assert (v2["sac.retrain_every"], v2["sac.retrain_failure_fraction"], v2["sac.retrain_gradient_steps"]) == (10000, 0.1, 10000)
    c = merged(path)
    SACCfg(**c["sac"]).validate(None)
    ResidualCfg(**{k: v for k, v in c["residual"].items() if k in ResidualCfg.__dataclass_fields__}).validate(None)
    # the evals run every episode to its end
    ev = leaves(merged(CONFIGS / "eval" / "residual_watch_furniture.yaml"))
    assert (ev["task.cfg.furniture.stall_steps_insert"], ev["task.cfg.furniture.stall_steps_screw"]) == (0, 0)


def test_the_parts_pd_v3_config_is_v2_with_chunks_and_no_correlated_noise():
    from robonuke_rl_core.learners.cfg import SACCfg

    folder = CONFIGS / "experiments" / "parts_pd"
    v2 = leaves(merged(folder / "residual_parts_pd_v2.yaml"))
    path = folder / "residual_parts_pd_v3.yaml"
    v3 = leaves(merged(path))
    changed = {k for k in set(v2) | set(v3) if v2.get(k) != v3.get(k)}
    assert changed == {
        "wandb.group", "trainer.total_timesteps", "trainer.checkpoint_interval", "trainer.write_interval",
        "residual.steps_per_chunk", "sac.discount_factor", "sac.learning_starts", "sac.noise_repeat",
        "sac.reference_columns", "sac.retrain_every", "sac.demo_transitions",
    }
    assert v3["residual.steps_per_chunk"] == 4 and v3["residual.base_action_horizon"] == 4
    assert v3["sac.n_step"] == 3 and v3["sac.noise_repeat"] is False and v3["sac.reference_columns"] == 4 * 7
    assert v3["trainer.total_timesteps"] * 4 == 1_000_000 and v3["sac.retrain_every"] == 0
    assert "{route}" in v3["sac.demo_transitions"]
    c = merged(path)
    SACCfg(**c["sac"]).validate(None)
    ResidualCfg(**{k: v for k, v in c["residual"].items() if k in ResidualCfg.__dataclass_fields__}).validate(None)


def test_the_parts_pd_v4_config_is_v3_with_an_absolute_residual_no_demos_and_a_base_aware_critic():
    from robonuke_rl_core.learners.cfg import SACCfg

    folder = CONFIGS / "experiments" / "parts_pd"
    v3 = leaves(merged(folder / "residual_parts_pd_v3.yaml"))
    path = folder / "residual_parts_pd_v4.yaml"
    v4 = leaves(merged(path))
    changed = {k for k in set(v3) | set(v4) if v3.get(k) != v4.get(k)}
    assert changed == {
        "wandb.group", "residual.target_axes", "residual.ee_pos_scale", "residual.ee_relative_actions",
        "residual.critic_state", "residual.base_warm_start", "sac.demo_transitions", "sac.demo_storage",
        "sac.parts_term_reduction", "sac.reference_dropout", "sac.reference_columns",
    }
    assert (v4["residual.target_axes"], v4["residual.ee_pos_scale"], v4["residual.ee_rot_scale_deg"]) == ("base", 0.02, 6.0)
    assert (v4["residual.critic_state"], v4["residual.critic_action"]) == ("residual", "residual")
    assert v4["residual.base_warm_start"] is False and v4["sac.demo_transitions"] is None
    assert v4["residual.steps_per_chunk"] == 4 and v4["sac.n_step"] == 3 and v4["sac.retrain_every"] == 0
    c = merged(path)
    SACCfg(**c["sac"]).validate(None)
    ResidualCfg(**{k: v for k, v in c["residual"].items() if k in ResidualCfg.__dataclass_fields__}).validate(None)


def test_the_parts_pd_v4_warmup_config_is_v4_with_five_base_episodes_first():
    from robonuke_rl_core.learners.cfg import SACCfg

    folder = CONFIGS / "experiments" / "parts_pd"
    v4 = leaves(merged(folder / "residual_parts_pd_v4.yaml"))
    path = folder / "residual_parts_pd_v4_warmup.yaml"
    w = leaves(merged(path))
    assert {k for k in set(v4) | set(w) if v4.get(k) != w.get(k)} == {"wandb.group", "sac.warmup_episodes"}
    assert w["sac.warmup_episodes"] == 5
    SACCfg(**merged(path)["sac"]).validate(None)


def test_the_parts_pd_v5_config_is_v4_with_critic_only_demos_and_a_critic_first_start():
    from robonuke_rl_core.learners.cfg import SACCfg

    folder = CONFIGS / "experiments" / "parts_pd"
    v4 = leaves(merged(folder / "residual_parts_pd_v4.yaml"))
    path = folder / "residual_parts_pd_v5.yaml"
    v5 = leaves(merged(path))
    assert {k for k in set(v4) | set(v5) if v4.get(k) != v5.get(k)} == {
        "wandb.group", "sac.demo_transitions", "sac.demo_storage", "sac.policy_learning_starts",
    }
    assert v5["sac.demo_storage"] == "memory_critic" and "{route}" in v5["sac.demo_transitions"]
    assert (v5["sac.learning_starts"], v5["sac.policy_learning_starts"]) == (125, 250)
    assert v5["sac.parts_term_reduction"] == "mean" and v5.get("sac.warmup_episodes", 0) == 0
    SACCfg(**merged(path)["sac"]).validate(None)


def test_the_parts_pd_v6_config_is_v5_with_a_data_based_start_and_std_0p1():
    from robonuke_rl_core.learners.cfg import SACCfg

    folder = CONFIGS / "experiments" / "parts_pd"
    v5 = leaves(merged(folder / "residual_parts_pd_v5.yaml"))
    path = folder / "residual_parts_pd_v6.yaml"
    v6 = leaves(merged(path))
    assert {k for k in set(v5) | set(v6) if v5.get(k) != v6.get(k)} == {
        "wandb.group", "sac.learning_starts", "sac.policy_learning_starts", "sac.learning_starts_rows", "model.actor.fixed_std",
    }
    assert (v6["sac.learning_starts_rows"], v6["sac.batch_size"], v6["model.actor.fixed_std"]) == (512, 512, 0.1)
    assert v6["sac.demo_storage"] == "memory_critic"
    SACCfg(**merged(path)["sac"]).validate(None)


def test_the_parts_pd_v7_config_is_v6_with_clone_and_anchor_weights_of_10():
    from robonuke_rl_core.learners.cfg import SACCfg

    folder = CONFIGS / "experiments" / "parts_pd"
    v6 = leaves(merged(folder / "residual_parts_pd_v6.yaml"))
    path = folder / "residual_parts_pd_v7.yaml"
    v7 = leaves(merged(path))
    assert {k for k in set(v6) | set(v7) if v6.get(k) != v7.get(k)} == {
        "wandb.group", "sac.success_clone_weight", "sac.failure_anchor_weight",
    }
    assert (v7["sac.success_clone_weight"], v7["sac.failure_anchor_weight"]) == (10.0, 10.0)
    SACCfg(**merged(path)["sac"]).validate(None)


def test_the_serl_bc_v1_config_is_parts_pd_v7_without_flashsac_on_simba_with_serls_learner():
    """SERL_BC v1 keeps every run setting of parts_pd v7 except the FlashSAC pieces, the network, SERL's learner and
    the actor's normalized observation (and its own demo files, group)."""
    from robonuke_rl_core.learners.cfg import SACCfg
    from robonuke_rl_core.models.cfg import SimbaActorCfg, SimbaCriticCfg, SimbaModelCfg

    v7 = leaves(merged(CONFIGS / "experiments" / "parts_pd" / "residual_parts_pd_v7.yaml"))
    path = CONFIGS / "experiments" / "serl_bc" / "residual_serl_bc_v1.yaml"
    s = leaves(merged(path))
    changed = {k for k in set(v7) | set(s) if v7.get(k) != s.get(k)}
    expected_sac = {
        "sac.critic_distribution", "sac.adaptive_reward_scaling", "sac.weight_normalization",
        "sac.cross_batch_value", "sac.lr_schedule", "sac.num_critics", "sac.actor_q_reduction", "sac.batch_size",
        "sac.normalize_observations", "sac.cuda_graph",
        "sac.gradient_steps", "sac.update_every", "sac.policy_delay", "sac.polyak", "sac.learn_entropy",
        "sac.initial_entropy_value", "sac.target_entropy", "sac.entropy_loss_form", "sac.entropy_observations",
    }
    model = {k for k in changed if k.startswith("model.")}
    assert changed - model == expected_sac | {"wandb.group"}
    assert s["sac.normalize_observations"] is True and s.get("residual.normalize_observation") in (None, False)
    assert s["model.architecture"] == "simba" and s["model.actor.act_init_std"] == 0.1
    assert (s["sac.num_critics"], s["sac.critic_subset"], s["sac.gradient_steps"], s["sac.policy_delay"]) == (10, 2, 10, 10)
    assert s["sac.actor_q_reduction"] == "mean" and s["sac.n_step"] == 3 and s["sac.learn_entropy"] is True
    assert s["sac.cuda_graph"] is True
    # the PARTS pieces are v7's
    for key in ("sac.route_clock", "sac.demo_storage", "sac.learning_starts_rows", "sac.success_clone_weight",
                "sac.failure_anchor_weight", "sac.parts_term_reduction", "residual.steps_per_chunk", "residual.target_axes",
                "task.cfg.furniture.stall_steps_screw", "sac.hold_still_key"):
        assert s[key] == v7[key], key
    c = merged(path)
    SACCfg(**c["sac"]).validate(None)
    SimbaModelCfg(actor=SimbaActorCfg(**c["model"]["actor"]), critic=SimbaCriticCfg(**c["model"]["critic"])).validate(None)
    ResidualCfg(**{k: v for k, v in c["residual"].items() if k in ResidualCfg.__dataclass_fields__}).validate(None)


def test_the_serl_bc_v2_config_is_v1_with_a_std_ceiling_and_a_faster_temperature():
    import math

    from robonuke_rl_core.learners.cfg import SACCfg
    from robonuke_rl_core.models.cfg import SimbaActorCfg, SimbaCriticCfg, SimbaModelCfg

    folder = CONFIGS / "experiments" / "serl_bc"
    v1 = leaves(merged(folder / "residual_serl_bc_v1.yaml"))
    path = folder / "residual_serl_bc_v2.yaml"
    v2 = leaves(merged(path))
    assert {k for k in set(v1) | set(v2) if v1.get(k) != v2.get(k)} == {"wandb.group", "sac.entropy_lr", "model.actor.max_log_std"}
    assert v2["sac.entropy_lr"] == 3.0e-3 and math.exp(v2["model.actor.max_log_std"]) == pytest.approx(0.2)
    c = merged(path)
    SACCfg(**c["sac"]).validate(None)
    SimbaModelCfg(actor=SimbaActorCfg(**c["model"]["actor"]), critic=SimbaCriticCfg(**c["model"]["critic"])).validate(None)


def test_the_serl_bc_v3_config_is_v2_with_symmetric_sampling_and_no_bc_terms():
    from robonuke_rl_core.learners.cfg import SACCfg

    folder = CONFIGS / "experiments" / "serl_bc"
    v2 = leaves(merged(folder / "residual_serl_bc_v2.yaml"))
    path = folder / "residual_serl_bc_v3.yaml"
    v3 = leaves(merged(path))
    assert {k for k in set(v2) | set(v3) if v2.get(k) != v3.get(k)} == {
        "wandb.group", "sac.demo_storage", "sac.demo_sampling_end", "sac.demo_batch_shared",
        "sac.success_clone_weight", "sac.failure_anchor_weight", "sac.actor_q_scale",
    }
    assert v3["sac.demo_batch_shared"] is True and v3["sac.demo_sampling_end"] == v3["trainer.total_timesteps"]
    assert (v3["sac.success_clone_weight"], v3["sac.failure_anchor_weight"], v3["sac.actor_q_scale"]) == (0.0, 0.0, None)
    SACCfg(**merged(path)["sac"]).validate(None)


def test_the_serl_bc_v4_config_is_v3_with_the_bc_terms_back():
    from robonuke_rl_core.learners.cfg import SACCfg

    folder = CONFIGS / "experiments" / "serl_bc"
    v3 = leaves(merged(folder / "residual_serl_bc_v3.yaml"))
    path = folder / "residual_serl_bc_v4.yaml"
    v4 = leaves(merged(path))
    assert {k for k in set(v3) | set(v4) if v3.get(k) != v4.get(k)} == {
        "wandb.group", "sac.success_clone_weight", "sac.failure_anchor_weight", "sac.actor_q_scale",
    }
    assert (v4["sac.success_clone_weight"], v4["sac.failure_anchor_weight"], v4["sac.actor_q_scale"]) == (10.0, 10.0, 2.5)
    assert v4["sac.demo_batch_shared"] is True
    SACCfg(**merged(path)["sac"]).validate(None)


def test_the_serl_bc_v5_config_is_v4_cloning_the_demo_rows():
    from robonuke_rl_core.learners.cfg import SACCfg

    folder = CONFIGS / "experiments" / "serl_bc"
    v4 = leaves(merged(folder / "residual_serl_bc_v4.yaml"))
    path = folder / "residual_serl_bc_v5.yaml"
    v5 = leaves(merged(path))
    assert {k for k in set(v4) | set(v5) if v4.get(k) != v5.get(k)} == {"wandb.group", "sac.clone_demo_rows"}
    assert v5["sac.clone_demo_rows"] is True
    SACCfg(**merged(path)["sac"]).validate(None)


def test_the_serl_bc_v6_config_is_plain_serl_with_critic_only_symmetric_sampling_and_mlps():
    import math

    from robonuke_rl_core.learners.cfg import SACCfg
    from robonuke_rl_core.models.cfg import MlpActorCfg, MlpCriticCfg, MlpModelCfg

    folder = CONFIGS / "experiments" / "serl_bc"
    v5 = leaves(merged(folder / "residual_serl_bc_v5.yaml"))
    path = folder / "residual_serl_bc_v6.yaml"
    v6 = leaves(merged(path))
    changed = {k for k in set(v5) | set(v6) if v5.get(k) != v6.get(k)}
    assert {k for k in changed if not k.startswith("model.")} == {
        "wandb.group", "sac.demo_batch_shared", "sac.clone_demo_rows", "sac.success_clone_weight",
        "sac.failure_anchor_weight", "sac.actor_q_scale", "sac.gradient_steps", "sac.policy_delay", "sac.entropy_lr",
    }
    assert v6.get("sac.demo_batch_shared", False) is False and v6.get("sac.clone_demo_rows", False) is False
    assert (v6["sac.gradient_steps"], v6["sac.policy_delay"], v6["sac.entropy_lr"]) == (2, 2, 3.0e-4)
    assert v6["sac.demo_storage"] == "sample" and v6["sac.demo_sampling_end"] == 250000
    assert v6["model.architecture"] == "mlp" and (v6["model.actor.hidden_size"], v6["model.actor.num_layers"]) == (256, 2)
    assert math.exp(v6["model.actor.init_log_std"]) == pytest.approx(0.1) and math.exp(v6["model.actor.max_log_std"]) == pytest.approx(0.2)
    c = merged(path)
    SACCfg(**c["sac"]).validate(None)
    MlpModelCfg(actor=MlpActorCfg(**c["model"]["actor"]), critic=MlpCriticCfg(**c["model"]["critic"])).validate(None)


def test_the_serl_bc_v7_config_is_v6_with_an_action_l2_penalty():
    from robonuke_rl_core.learners.cfg import SACCfg

    folder = CONFIGS / "experiments" / "serl_bc"
    v6 = leaves(merged(folder / "residual_serl_bc_v6.yaml"))
    path = folder / "residual_serl_bc_v7.yaml"
    v7 = leaves(merged(path))
    assert {k for k in set(v6) | set(v7) if v6.get(k) != v7.get(k)} == {"wandb.group", "sac.action_l2_weight"}
    assert v7["sac.action_l2_weight"] == 1.0
    SACCfg(**merged(path)["sac"]).validate(None)


def test_serl_bc_v8_is_v7_per_env_step_with_shaping_and_no_l2():
    from omegaconf import OmegaConf

    from robonuke_rl_core.learners.cfg import SACCfg

    folder = CONFIGS / "experiments" / "serl_bc"
    v7 = leaves(merged(folder / "residual_serl_bc_v7.yaml"))
    v8 = leaves(merged(folder / "residual_serl_bc_v8.yaml"))
    changed = {k for k in set(v7) | set(v8) if v7.get(k) != v8.get(k)}
    assert changed == {
        "wandb.group", "trainer.total_timesteps", "trainer.checkpoint_interval", "trainer.write_interval",
        "residual.steps_per_chunk", "sac.action_l2_weight", "sac.potential_terms", "sac.demo_transitions",
        "sac.demo_sampling_end", "sac.discount_factor", "sac.learning_starts_rows", "sac.update_every", "sac.target_entropy",
    }
    assert v8["residual.steps_per_chunk"] == 1 and v8["sac.action_l2_weight"] == 0.0
    # the section's schema takes the entries as the strict config loader does (struct mode), and they validate
    schema = OmegaConf.structured(SACCfg)
    OmegaConf.set_struct(schema, True)
    sac = OmegaConf.to_object(OmegaConf.merge(schema, OmegaConf.create(merged(folder / "residual_serl_bc_v8.yaml")["sac"])))
    sac.validate(None)
    terms = {(t.route, t.channel): t.weight for t in sac.potential_terms}
    assert terms == {("insert", "eval_state/phase/tip_mm"): 0.01, ("insert", "eval_state/phase/tilt_deg"): 0.01,
                     ("screw", "eval_state/phase/height_mm"): 0.1, ("screw", "eval_state/phase/tilt_deg"): 0.05}


def test_serl_bc_fixed_demo_v1_is_v8_on_its_own_demo_files():
    folder = CONFIGS / "experiments" / "serl_bc"
    v8 = leaves(merged(folder / "residual_serl_bc_v8.yaml"))
    fixed = leaves(merged(folder / "residual_serl_bc_fixed_demo_v1.yaml"))
    changed = {k for k in set(v8) | set(fixed) if v8.get(k) != fixed.get(k)}
    assert changed == {"wandb.group", "sac.demo_transitions"}
    assert fixed["sac.demo_transitions"] == "data/serl_bc_fixed_demo_v1/demos_{route}.pt"


def test_serl_bc_fixed_demo_v2_is_v1_on_its_own_demo_files():
    folder = CONFIGS / "experiments" / "serl_bc"
    v1 = leaves(merged(folder / "residual_serl_bc_fixed_demo_v1.yaml"))
    v2 = leaves(merged(folder / "residual_serl_bc_fixed_demo_v2.yaml"))
    changed = {k for k in set(v1) | set(v2) if v1.get(k) != v2.get(k)}
    assert changed == {"wandb.group", "sac.demo_transitions"}


def test_serl_bc_fixed_demo_v3_changes_only_the_posing():
    folder = CONFIGS / "experiments" / "serl_bc"
    v2 = leaves(merged(folder / "residual_serl_bc_fixed_demo_v2.yaml"))
    v3 = leaves(merged(folder / "residual_serl_bc_fixed_demo_v3.yaml"))
    changed = {k for k in set(v2) | set(v3) if v2.get(k) != v3.get(k)}
    assert changed == {"wandb.group", "sac.demo_transitions", "residual.observation_drop", "residual.critic_action",
                       "residual.residual_gripper", "sac.discount_factor", "sac.target_entropy"}
    assert v3["residual.observation_drop"] == [[25, 52], [61, 63]]
    c = merged(folder / "residual_serl_bc_fixed_demo_v3.yaml")
    ResidualCfg(**{k: v for k, v in c["residual"].items() if k in ResidualCfg.__dataclass_fields__}).validate(None)


def test_serl_bc_fixed_demo_v4_is_v3_with_the_shaping_ten_times_stronger():
    folder = CONFIGS / "experiments" / "serl_bc"
    v3 = merged(folder / "residual_serl_bc_fixed_demo_v3.yaml")
    v4 = merged(folder / "residual_serl_bc_fixed_demo_v4.yaml")
    a, b = leaves(v3), leaves(v4)
    changed = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert changed == {"wandb.group", "sac.demo_transitions", "sac.potential_terms"}
    w3 = {(t["route"], t["channel"]): t["weight"] for t in v3["sac"]["potential_terms"]}
    w4 = {(t["route"], t["channel"]): t["weight"] for t in v4["sac"]["potential_terms"]}
    assert w4.keys() == w3.keys() and all(abs(w4[k] - 10 * w3[k]) < 1e-9 for k in w3)
