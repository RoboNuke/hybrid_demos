"""residual.action_half_range_floor (ResFiT's ActionScaler floor) and the absolute frame's executed clip:
my_project/residual.py residual_scale_and_clip. (The residual's inputs are never scaled here: the learner's
state_preprocessor normalizes them.)"""

from types import SimpleNamespace

import pytest
import torch

import my_project.residual as residual
from my_project.models.diffusion_policy import MinMaxNormalizer


def _policy(action_min, action_max, obs_min, obs_max):
    norm = MinMaxNormalizer(len(obs_min), len(action_min))
    norm.fit("action", torch.tensor([action_min, action_max], dtype=torch.float32))
    norm.fit("obs", torch.tensor([obs_min, obs_max], dtype=torch.float32))
    return SimpleNamespace(normalizer=norm, device=torch.device("cpu"))


def _rc(**over):
    rc = dict(action_scale=None, action_scale_fraction=1.2, action_half_range_floor=None, action_frame="absolute", executed_clip=None)
    rc.update(over)
    return SimpleNamespace(**rc)


@pytest.fixture
def stats(monkeypatch):
    s = {"action_min": torch.tensor([-0.5, 1.57]), "action_max": torch.tensor([0.5, 1.57])}
    monkeypatch.setattr(residual, "demo_action_range", lambda base_dir: s)
    return s


def test_the_defaults_leave_the_scale_as_it_was(stats):
    policy = _policy([-0.5, 1.57], [0.5, 1.57], [0.0, 5.0], [2.0, 5.0])
    scale, _ = residual.residual_scale_and_clip(_rc(), policy, None)
    # the constant action dim's widened half range is 1: 1.2 per unit r (what the floor fixes)
    assert torch.allclose(scale, torch.tensor([0.6, 1.2]))


def test_the_half_range_floor_shrinks_only_the_narrow_dims(stats):
    policy = _policy([-0.5, 1.57], [0.5, 1.57], [0.0, 5.0], [2.0, 5.0])
    scale, _ = residual.residual_scale_and_clip(_rc(action_half_range_floor=0.05), policy, None)
    # ResFiT's ActionScaler: (1 + action_scale) x max(half range, min_action_range / 2)
    assert torch.allclose(scale, torch.tensor([1.2 * 0.5, 1.2 * 0.05]))


def test_the_config_refuses_the_floor_with_a_set_scale():
    from my_project.cfg import ResidualCfg

    with pytest.raises(ValueError, match="action_scale: null"):
        ResidualCfg(action_half_range_floor=0.05, action_scale=[0.1] * 7).validate(None)


def test_the_absolute_clip_is_resfits_widened_range_of_the_floored_half_range(stats):
    policy = _policy([-0.5, 1.57], [0.5, 1.57], [0.0, 5.0], [2.0, 5.0])
    scale, clip = residual.residual_scale_and_clip(_rc(action_half_range_floor=0.05, executed_clip=1.2), policy, None)
    # ResFiT clamps the combined normalized action to [-1, 1] of mid +- (1 + action_scale) x max(half range, floor)
    assert torch.allclose(clip[0], torch.tensor([-0.6, 1.57 - 0.06]))
    assert torch.allclose(clip[1], torch.tensor([0.6, 1.57 + 0.06]))
    assert residual.residual_scale_and_clip(_rc(), policy, None)[1] is None
