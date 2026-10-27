"""``residual.action_frame: target`` (Policy Decorator's delta-pose meaning of r on an absolute base) and
``residual.critic_state: observation`` (its critic: the observation alone, the base chunk a side input)."""

from __future__ import annotations

import math

import pytest
import torch

from my_project.cfg import ResidualCfg
from my_project.models.action_frame import mat_to_rot6d, rot6d_to_mat
from my_project.models.diffusion_policy import MinMaxNormalizer
from my_project.residual import SIDE_INPUT, SideInputTransform, target_full_transform
from my_project.residual_ee import exp_so3
from my_project.residual_target import WIDTH, EERelative, TargetResidual


def random_targets(n: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    R = exp_so3(torch.randn(n, 3))
    return torch.cat((torch.randn(n, 3) * 0.2, mat_to_rot6d(R), torch.rand(n, 1)), dim=-1)


def test_r_one_is_a_centimetre_and_a_hundredth_of_a_radian_along_the_targets_axes():
    target = TargetResidual(0.01, math.degrees(0.01), 0.004)
    base = random_targets(8)
    R_base = rot6d_to_mat(base[:, 3:9])
    for axis in range(3):
        r = torch.zeros(8, WIDTH)
        r[:, axis] = 1.0
        sent = target.step(base, r)
        torch.testing.assert_close(sent[:, 0:3] - base[:, 0:3], 0.01 * R_base[:, :, axis])
        torch.testing.assert_close(sent[:, 3:9], base[:, 3:9])
        r = torch.zeros(8, WIDTH)
        r[:, 3 + axis] = 1.0
        sent = target.step(base, r)
        relative = R_base.transpose(-1, -2) @ rot6d_to_mat(sent[:, 3:9])
        angle = torch.arccos(((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1))
        torch.testing.assert_close(angle, torch.full((8,), 0.01), atol=1e-4, rtol=0)  # float32 arccos near 1
        torch.testing.assert_close(sent[:, 0:3], base[:, 0:3])
    r = torch.zeros(8, WIDTH)
    r[:, 6] = -1.0
    torch.testing.assert_close(target.step(base, r)[:, 9], base[:, 9] - 0.004)


def test_a_zero_residual_sends_the_base():
    target = TargetResidual(0.01, 0.573, 0.004)
    base = random_targets(5)
    torch.testing.assert_close(target.step(base, torch.zeros(5, WIDTH)), base, atol=1e-6, rtol=0)


def a_normalizer() -> MinMaxNormalizer:
    torch.manual_seed(3)
    normalizer = MinMaxNormalizer(4, 10)
    normalizer.fit("obs", torch.randn(64, 4))
    normalizer.fit("action", random_targets(64, 1) * 1.5)
    return normalizer


def test_the_side_input_critic_action_is_the_chunks_executed_targets():
    normalizer, target, k = a_normalizer(), TargetResidual(0.01, 0.573, 0.004), 4
    base = random_targets(6 * k, 2).reshape(6, k, 10)
    r = torch.rand(6, k * WIDTH) * 2 - 1
    transform = SideInputTransform(normalizer, k, 10, target=target)
    assert transform.side_inputs == {SIDE_INPUT: k * 10} and transform.critic_action_width == k * 10
    side = {SIDE_INPUT: normalizer.normalize(base, "action").reshape(6, -1)}
    got = transform(torch.randn(6, 4), r, side)
    expected = normalizer.normalize(target.step(base, r.reshape(6, k, WIDTH)), "action").reshape(6, -1)
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)
    r.requires_grad_(True)
    transform(torch.randn(6, 4), r, side).sum().backward()
    assert torch.isfinite(r.grad).all()


def test_the_critic_state_variant_reads_the_same_base_from_its_slice():
    normalizer, target, k = a_normalizer(), TargetResidual(0.01, 0.573, 0.004), 4
    base_n = normalizer.normalize(random_targets(3 * k, 4).reshape(3, k, 10), "action").reshape(3, -1)
    r = torch.rand(3, k * WIDTH) * 2 - 1
    critic = torch.cat((torch.randn(3, 4), base_n), dim=-1)
    via_state = target_full_transform(slice(4, 4 + k * 10), normalizer, k, 10, target)
    via_side = SideInputTransform(normalizer, k, 10, target=target)
    torch.testing.assert_close(via_state(critic, r), via_side(None, r, {SIDE_INPUT: base_n}))
    assert via_state.critic_action_width == k * 10


def test_the_absolute_side_input_transform_is_normalize_base_plus_gain_r():
    normalizer = a_normalizer()
    gain = torch.linspace(0.01, 0.1, 10)
    transform = SideInputTransform(normalizer, 2, 10, gain=gain)
    assert not hasattr(transform, "critic_action_width")
    base_n, r = torch.randn(5, 20), torch.rand(5, 20) * 2 - 1
    torch.testing.assert_close(transform(None, r, {SIDE_INPUT: base_n}), base_n + gain.repeat(2) * r)


def test_the_new_fields_default_to_the_old_behaviour_and_fail_loud():
    cfg = ResidualCfg()
    assert cfg.critic_state == "residual" and cfg.action_frame == "absolute"
    with pytest.raises(ValueError, match="critic_state"):
        ResidualCfg(critic_state="nothing").validate(None)
    with pytest.raises(NotImplementedError, match="observation"):
        ResidualCfg(critic_state="observation", action_frame="ee").validate(None)


def test_ee_relative_is_the_target_against_the_ee_pose_in_its_frame():
    normalizer = a_normalizer()
    rel = EERelative(0.1, 0.1, normalizer)
    ee = random_targets(5, 7)[:, :9]
    R_ee = rot6d_to_mat(ee[:, 3:9])
    # a target 2 cm along the EE's own y axis, turned 0.05 rad about its own z, gripper as in the data
    shift = torch.tensor([0.0, 0.02, 0.0])
    p = ee[:, 0:3] + (R_ee @ shift)
    R = R_ee @ exp_so3(torch.tensor([0.0, 0.0, 0.05]).expand(5, 3))
    targets = torch.cat((p, mat_to_rot6d(R), torch.full((5, 1), 0.3)), -1).unsqueeze(1)
    out = rel(targets, ee)
    torch.testing.assert_close(out[:, 0:3], torch.tensor([0.0, 0.2, 0.0]).expand(5, 3), atol=1e-5, rtol=0)
    torch.testing.assert_close(out[:, 3:6], torch.tensor([0.0, 0.0, 0.5]).expand(5, 3), atol=1e-4, rtol=0)
    torch.testing.assert_close(out[:, 6], normalizer.normalize(targets, "action")[:, 0, 9])
    # the target at the EE itself is zero motion
    at_ee = torch.cat((ee, torch.zeros(5, 1)), -1).unsqueeze(1)
    torch.testing.assert_close(rel(at_ee, ee)[:, 0:6], torch.zeros(5, 6), atol=1e-4, rtol=0)


def test_the_ee_relative_critic_action_is_the_executed_chunk_in_that_form():
    normalizer, target, k = a_normalizer(), TargetResidual(0.01, 0.573, 0.004), 4
    rel = EERelative(0.1, 0.1, normalizer)
    base = random_targets(6 * k, 2).reshape(6, k, 10)
    ee = random_targets(6, 9)[:, :9]
    r = torch.rand(6, k * WIDTH) * 2 - 1
    transform = SideInputTransform(normalizer, k, 10, target=target, ee_rel=rel)
    assert transform.side_inputs == {SIDE_INPUT: k * 10 + 9} and transform.critic_action_width == k * 7
    side = {SIDE_INPUT: torch.cat((normalizer.normalize(base, "action").reshape(6, -1), ee), -1)}
    got = transform(None, r, side)
    torch.testing.assert_close(got, rel(target.step(base, r.reshape(6, k, WIDTH)), ee), atol=1e-4, rtol=1e-4)
    r.requires_grad_(True)
    transform(None, r, side).sum().backward()
    assert torch.isfinite(r.grad).all()


def test_ee_relative_needs_the_target_frame_and_the_observation_critic():
    with pytest.raises(NotImplementedError, match="ee_relative"):
        ResidualCfg(ee_relative_actions=True).validate(None)
    ResidualCfg(ee_relative_actions=True, action_frame="target", critic_state="observation").validate(None)
