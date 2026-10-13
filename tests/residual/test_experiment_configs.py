"""The residual ablation configs: each differs from ``residual_sac.yaml`` by its own change only.

Read at the file level (the package's ``base``-chain reader, no Isaac Lab): every layer of an
experiment's chain merged in order. The task cfg is not validated here (that needs the sim);
the sections these experiments change are, through their dataclasses.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from robonuke_rl_core.configfile import load_file_chain
from robonuke_rl_core.learners.cfg import SACCfg

from my_project.cfg import HistoryCfg

EXPERIMENTS = Path(__file__).resolve().parents[2] / "configs" / "experiments"

HISTORY = {"history.enabled"}
ENSEMBLE = {
    "sac.num_critics",
    "sac.critic_subset",
    "sac.actor_q_reduction",
    "sac.gradient_steps",
    "sac.policy_delay",
}
DEMO = {
    "sac.learning_starts",
    "sac.policy_learning_starts",
    "sac.demo_transitions",
    "sac.demo_sampling_start",
    "sac.demo_sampling_end",
}
CASES = {
    "residual_sac_history": HISTORY,
    "residual_sac_ensemble": ENSEMBLE,
    "residual_sac_demo_critic": DEMO,
    "residual_sac_history_ensemble": HISTORY | ENSEMBLE,
    "residual_sac_all": HISTORY | ENSEMBLE | DEMO,
}


def merged(stem: str) -> dict:
    layers = [layer for _, layer in load_file_chain(EXPERIMENTS / f"{stem}.yaml")]
    return OmegaConf.to_container(OmegaConf.merge(*layers), resolve=True)


def leaves(data, prefix=""):
    if isinstance(data, dict):
        out = {}
        for key, value in data.items():
            out.update(leaves(value, f"{prefix}.{key}" if prefix else key))
        return out
    return {prefix: data}


@pytest.mark.parametrize("stem, changed", sorted(CASES.items()))
def test_only_the_named_change_differs_from_the_baseline(stem, changed):
    base, exp = leaves(merged("residual_sac")), leaves(merged(stem))
    diff = {k for k in set(base) | set(exp) if base.get(k, "<unset>") != exp.get(k, "<unset>")}
    assert diff == changed | {"wandb.group"}
    assert exp["wandb.group"] == stem


@pytest.mark.parametrize("stem", sorted(CASES))
def test_the_changed_sections_validate(stem):
    data = merged(stem)
    sac = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(SACCfg), data["sac"]))
    sac.validate(None)
    history = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(HistoryCfg), data.get("history", {})))
    history.validate(None)


def test_each_demo_experiment_has_its_own_transition_file():
    """The transitions carry the agent's observation, so a history run cannot share a file
    with a run without one."""
    files = {stem: merged(stem)["sac"].get("demo_transitions") for stem in CASES}
    with_demos = {stem: path for stem, path in files.items() if path}
    assert set(with_demos) == {"residual_sac_demo_critic", "residual_sac_all"}
    assert len(set(with_demos.values())) == 2


def test_the_scale_ablation_differs_from_residual_sac_all_by_the_bound_only():
    base, exp = leaves(merged("residual_sac_all")), leaves(merged("residual_sac_all_scale03"))
    diff = {k for k in set(base) | set(exp) if base.get(k, "<unset>") != exp.get(k, "<unset>")}
    # the demo file differs because its actions are divided by the scale
    assert diff == {"wandb.group", "residual.action_scale_fraction", "sac.demo_transitions"}
    assert exp["residual.action_scale_fraction"] == 0.3


def test_the_0p2_scale_ablation_differs_from_residual_sac_all_by_the_bound_only():
    base, exp = leaves(merged("residual_sac_all")), leaves(merged("residual_sac_all_scale02"))
    diff = {k for k in set(base) | set(exp) if base.get(k, "<unset>") != exp.get(k, "<unset>")}
    assert diff == {"wandb.group", "residual.action_scale_fraction", "sac.demo_transitions"}
    assert exp["residual.action_scale_fraction"] == 0.2


def test_the_hybrid_experiment_differs_from_residual_sac_all_by_the_hybrid_action_only():
    base, exp = leaves(merged("residual_sac_all")), leaves(merged("residual_hybrid_all"))
    diff = {k for k in set(base) | set(exp) if base.get(k, "<unset>") != exp.get(k, "<unset>")}
    # the demo file differs because its actions carry the selection and force blocks; the
    # entropy target follows the action width (-|A| / 2)
    assert diff == {
        "wandb.group",
        "sac.demo_transitions",
        "sac.target_entropy",
        "hybrid.enabled",
        "hybrid.force_axes",
        "hybrid.force_target_bounds",
        "model.actor.bernoulli_action_dims",
    }
    assert exp["sac.target_entropy"] == -5.0
    assert exp["model.actor.bernoulli_action_dims"] == [4, 5, 6]
