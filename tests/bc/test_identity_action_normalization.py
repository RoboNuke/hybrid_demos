"""bc.normalize_action: false -- the action is used as it is: its range is fixed at [-1, 1]."""

import pytest
import torch

from my_project.models.diffusion_policy import MinMaxNormalizer


def test_an_unnormalized_action_maps_to_itself_and_zero_stays_zero():
    # a narrow column and a one-sided one: a fit to the data would move both, the fixed range does not
    action = torch.tensor([[0.02, 0.5, -1.0], [0.0201, 0.9, 1.0], [0.0202, 0.7, 0.0]])
    n = MinMaxNormalizer(2, 3)
    n.fit_identity("action", action)
    assert torch.equal(n.action_min, -torch.ones(3)) and torch.equal(n.action_max, torch.ones(3))
    assert torch.allclose(n.normalize(action, "action"), action, atol=1e-7)
    assert torch.allclose(n.denormalize(action, "action"), action, atol=1e-7)
    assert torch.equal(n.denormalize(torch.zeros(1, 3), "action"), torch.zeros(1, 3))  # zero output: zero action


def test_an_action_outside_minus_one_one_is_refused_naming_its_dims():
    n = MinMaxNormalizer(2, 3)
    with pytest.raises(ValueError, match=r"action dims \[0, 2\] leave \[-1, 1\]"):
        n.fit_identity("action", torch.tensor([[1.4, 0.0, 0.0], [0.0, 0.0, -1.2]]))


def test_symmetric_scaling_keeps_zero_at_zero_and_fills_minus_one_one():
    data = torch.tensor([[-0.156, 0.02, -3.14, 1.0], [0.23, 0.175, 1.3, -1.0], [0.0, 0.1, 0.0, 1.0]])
    n = MinMaxNormalizer(2, 4)
    n.fit_symmetric("action", data)
    assert torch.allclose(n.action_max, torch.tensor([0.23, 0.175, 3.14, 1.0])) and torch.equal(n.action_min, -n.action_max)
    assert torch.allclose(n.normalize(data, "action"), data / n.action_max, atol=1e-6)
    assert torch.allclose(n.denormalize(torch.zeros(1, 4), "action"), torch.zeros(1, 4), atol=1e-7)  # zero output: zero action
    assert n.normalize(data, "action").abs().max() <= 1.0 + 1e-6


def test_without_clip_sample_an_unnormalized_action_may_take_any_value():
    data = torch.tensor([[0.2, 3.1, -0.4], [0.0, -2.5, 0.1]])
    n = MinMaxNormalizer(2, 3)
    n.fit_identity("action", data, bounded=False)  # clip_sample off: m and rad as they are
    assert torch.allclose(n.normalize(data, "action"), data, atol=1e-6)
