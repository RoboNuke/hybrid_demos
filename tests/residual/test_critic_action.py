"""``residual.critic_action: full``: the SAC critics take the executed action, normalized as the base's."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from my_project.cfg import ResidualCfg
from my_project.models.diffusion_policy import MinMaxNormalizer
from my_project.residual import full_action_transform, residual_critic_action, residual_observation, residual_scale

OBS, ACT, HISTORY = 7, 10, 5


def a_normalizer(seed: int = 0) -> MinMaxNormalizer:
    torch.manual_seed(seed)
    normalizer = MinMaxNormalizer(OBS, ACT)
    normalizer.fit("obs", torch.randn(64, OBS) * 3)
    normalizer.fit("action", torch.randn(64, ACT) * torch.linspace(0.05, 2.0, ACT))
    return normalizer


def agent_observation(normalizer, obs, base_action):
    """What the agent sees: the residual's observation, then the history (outermost)."""
    policy_obs, _ = residual_observation(normalizer, obs, base_action)
    return torch.cat((policy_obs, torch.randn(obs.shape[0], HISTORY)), dim=-1)


def the_transform(normalizer, fraction=0.02):
    policy = SimpleNamespace(normalizer=normalizer, device=torch.device("cpu"))
    scale = residual_scale(policy, None, fraction)
    gain = scale / ((normalizer.action_max - normalizer.action_min) / 2)
    return scale, full_action_transform(slice(OBS, OBS + ACT), gain)


def test_the_critic_action_is_the_executed_action_normalized_as_the_bases():
    normalizer = a_normalizer()
    scale, transform = the_transform(normalizer)
    obs, base = torch.randn(32, OBS), torch.randn(32, ACT)
    r = torch.rand(32, ACT) * 2 - 1
    got = transform(agent_observation(normalizer, obs, base), r)
    assert torch.allclose(got, normalizer.normalize(base + scale * r, "action"), atol=1e-5)


def test_a_demo_beyond_the_residuals_reach_maps_to_its_own_action():
    """The demo file stores r = (a_demo - a_base) / scale unclipped (|r| up to ~55 at 0.02): the
    critic then sees exactly normalize(a_demo), an ordinary action, not an r far outside [-1, 1]."""
    normalizer = a_normalizer(1)
    scale, transform = the_transform(normalizer)
    obs, base = torch.randn(16, OBS), torch.randn(16, ACT)
    demo = base + torch.randn(16, ACT) * 0.5
    r = (demo - base) / scale
    assert r.abs().max() > 5.0  # far beyond the residual's reach
    got = transform(agent_observation(normalizer, obs, base), r)
    assert torch.allclose(got, normalizer.normalize(demo, "action"), atol=1e-4)


def test_action_dims_past_the_pose_pass_through():
    normalizer = a_normalizer()
    _, transform = the_transform(normalizer)
    obs, base = torch.randn(4, OBS), torch.randn(4, ACT)
    extra = torch.randn(4, 3)
    got = transform(agent_observation(normalizer, obs, base), torch.cat((torch.zeros(4, ACT), extra), dim=-1))
    assert torch.equal(got[:, ACT:], extra)
    assert torch.allclose(got[:, :ACT], normalizer.normalize(base, "action"), atol=1e-6)


def test_the_builder_answers_only_for_an_enabled_residual_with_full():
    off = SimpleNamespace(residual=SimpleNamespace(enabled=True, critic_action="residual"))
    disabled = SimpleNamespace(residual=SimpleNamespace(enabled=False, critic_action="full"))
    assert residual_critic_action(off, env=None) is None
    assert residual_critic_action(disabled, env=None) is None
    assert residual_critic_action(SimpleNamespace(), env=None) is None
    with pytest.raises(RuntimeError, match="ResidualPolicyWrapper"):
        residual_critic_action(SimpleNamespace(residual=SimpleNamespace(enabled=True, critic_action="full")), env=object())


def test_critic_action_is_validated():
    with pytest.raises(ValueError, match="critic_action"):
        ResidualCfg(critic_action="executed").validate(None)
    ResidualCfg(critic_action="full").validate(None)
    ppo = SimpleNamespace(trainer=SimpleNamespace(learner="ppo"))
    with pytest.raises(ValueError, match="no Q critic"):
        ResidualCfg(enabled=True, base_run="runs/bc", critic_action="full").validate(ppo)
