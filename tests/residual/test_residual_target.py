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
    transform = SideInputTransform(k, 10, target=target)
    assert transform.side_inputs == {SIDE_INPUT: k * 10} and transform.critic_action_width == k * 10
    side = {SIDE_INPUT: base.reshape(6, -1)}  # raw: the learner's state_preprocessor normalizes the critics' action
    got = transform(torch.randn(6, 4), r, side)
    expected = target.step(base, r.reshape(6, k, WIDTH)).reshape(6, -1)
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)
    r.requires_grad_(True)
    transform(torch.randn(6, 4), r, side).sum().backward()
    assert torch.isfinite(r.grad).all()


def test_the_critic_state_variant_reads_the_same_base_from_its_slice():
    normalizer, target, k = a_normalizer(), TargetResidual(0.01, 0.573, 0.004), 4
    base_n = random_targets(3 * k, 4).reshape(3, -1)
    r = torch.rand(3, k * WIDTH) * 2 - 1
    critic = torch.cat((torch.randn(3, 4), base_n), dim=-1)
    via_state = target_full_transform(slice(4, 4 + k * 10), k, 10, target)
    via_side = SideInputTransform(k, 10, target=target)
    torch.testing.assert_close(via_state(critic, r), via_side(None, r, {SIDE_INPUT: base_n}))
    assert via_state.critic_action_width == k * 10


def test_the_absolute_side_input_transform_is_base_plus_scale_r():
    scale = torch.linspace(0.01, 0.1, 10)
    transform = SideInputTransform(2, 10, scale=scale)
    assert not hasattr(transform, "critic_action_width")
    base, r = torch.randn(5, 20), torch.rand(5, 20) * 2 - 1
    torch.testing.assert_close(transform(None, r, {SIDE_INPUT: base}), base + scale.repeat(2) * r)


def test_the_new_fields_default_to_the_old_behaviour_and_fail_loud():
    cfg = ResidualCfg()
    assert cfg.critic_state == "residual" and cfg.action_frame == "absolute"
    with pytest.raises(ValueError, match="critic_state"):
        ResidualCfg(critic_state="nothing").validate(None)
    with pytest.raises(NotImplementedError, match="observation"):
        ResidualCfg(critic_state="observation", action_frame="ee").validate(None)


def test_ee_relative_is_the_target_against_the_ee_pose_in_its_frame():
    rel = EERelative()
    ee = random_targets(5, 7)[:, :9]
    R_ee = rot6d_to_mat(ee[:, 3:9])
    # a target 2 cm along the EE's own y axis, turned 0.05 rad about its own z, gripper as in the data
    shift = torch.tensor([0.0, 0.02, 0.0])
    p = ee[:, 0:3] + (R_ee @ shift)
    R = R_ee @ exp_so3(torch.tensor([0.0, 0.0, 0.05]).expand(5, 3))
    targets = torch.cat((p, mat_to_rot6d(R), torch.full((5, 1), 0.3)), -1).unsqueeze(1)
    out = rel(targets, ee)
    torch.testing.assert_close(out[:, 0:3], torch.tensor([0.0, 0.02, 0.0]).expand(5, 3), atol=1e-5, rtol=0)  # m
    torch.testing.assert_close(out[:, 3:6], torch.tensor([0.0, 0.0, 0.05]).expand(5, 3), atol=1e-4, rtol=0)  # rad
    torch.testing.assert_close(out[:, 6], targets[:, 0, 9])  # the gripper command as it is
    # the target at the EE itself is zero motion
    at_ee = torch.cat((ee, torch.zeros(5, 1)), -1).unsqueeze(1)
    torch.testing.assert_close(rel(at_ee, ee)[:, 0:6], torch.zeros(5, 6), atol=1e-4, rtol=0)


def test_the_ee_relative_critic_action_is_the_executed_chunk_in_that_form():
    target, k = TargetResidual(0.01, 0.573, 0.004), 4
    rel = EERelative()
    base = random_targets(6 * k, 2).reshape(6, k, 10)
    ee = random_targets(6, 9)[:, :9]
    r = torch.rand(6, k * WIDTH) * 2 - 1
    transform = SideInputTransform(k, 10, target=target, ee_rel=rel)
    assert transform.side_inputs == {SIDE_INPUT: k * 10 + 9} and transform.critic_action_width == k * 7
    side = {SIDE_INPUT: torch.cat((base.reshape(6, -1), ee), -1)}
    got = transform(None, r, side)
    torch.testing.assert_close(got, rel(target.step(base, r.reshape(6, k, WIDTH)), ee), atol=1e-4, rtol=1e-4)
    r.requires_grad_(True)
    transform(None, r, side).sum().backward()
    assert torch.isfinite(r.grad).all()


def test_ee_relative_needs_the_target_frame_and_the_observation_critic():
    with pytest.raises(NotImplementedError, match="ee_relative"):
        ResidualCfg(ee_relative_actions=True).validate(None)
    ResidualCfg(ee_relative_actions=True, action_frame="target", critic_state="observation").validate(None)


def test_base_axes_r_one_is_two_centimetres_and_six_degrees_about_the_robot_base_axes():
    """residual.target_axes: base: r = 1 is a fixed absolute correction in the robot-base frame, whatever the target's
    orientation: p_base + 0.02 m e_axis, Exp(6 deg e_axis) R_base."""
    target = TargetResidual(0.02, 6.0, 0.004, axes="base")
    base = random_targets(8)
    R_base = rot6d_to_mat(base[:, 3:9])
    for axis in range(3):
        r = torch.zeros(8, WIDTH)
        r[:, axis] = 1.0
        sent = target.step(base, r)
        step = torch.zeros(8, 3)
        step[:, axis] = 0.02
        torch.testing.assert_close(sent[:, 0:3] - base[:, 0:3], step)
        torch.testing.assert_close(sent[:, 3:9], base[:, 3:9])
        r = torch.zeros(8, WIDTH)
        r[:, 3 + axis] = 1.0
        R_sent = rot6d_to_mat(target.step(base, r)[:, 3:9])
        world = R_sent @ R_base.transpose(-1, -2)  # the rotation applied in the robot-base frame
        angle = torch.arccos(((world.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1))
        torch.testing.assert_close(angle, torch.full((8,), math.radians(6.0)), atol=1e-4, rtol=0)
        axis_vec = torch.stack((world[:, 2, 1] - world[:, 1, 2], world[:, 0, 2] - world[:, 2, 0], world[:, 1, 0] - world[:, 0, 1]), -1)
        assert torch.all(axis_vec.argmax(dim=-1) == axis)  # about the base frame's own axis
    with pytest.raises(ValueError, match="axes"):
        TargetResidual(0.02, 6.0, 0.004, axes="world")


def test_target_axes_and_warm_start_are_validated_and_applied():
    from types import SimpleNamespace

    from my_project.residual import override_base

    with pytest.raises(ValueError, match="target_axes"):
        ResidualCfg(enabled=True, base_run="r", action_frame="target", target_axes="world").validate(SimpleNamespace(task_name="Isaac-Furniture-OneLeg-FBAbs-ResiP-v0", hybrid=SimpleNamespace(enabled=False)))
    with pytest.raises(ValueError, match="target_axes"):
        ResidualCfg(enabled=True, base_run="r", action_frame="absolute", target_axes="base").validate(SimpleNamespace(task_name="Isaac-Furniture-OneLeg-FBAbs-ResiP-v0", hybrid=SimpleNamespace(enabled=False)))
    policy = SimpleNamespace(cfg=SimpleNamespace(warm_start=True, inference_steps=4, num_diffusion_iters=100), horizon=32, action_horizon=8)
    override_base(policy, SimpleNamespace(base_action_horizon=4, base_inference_steps=None, steps_per_chunk=4, base_warm_start=False))
    assert policy.cfg.warm_start is False and policy.action_horizon == 4


def test_without_the_gripper_the_residual_is_the_pose_and_the_gripper_is_the_bases():
    base = torch.cat((torch.randn(5, 3), mat_to_rot6d(exp_so3(torch.randn(5, 3))), torch.randn(5, 1)), dim=-1)
    full = TargetResidual(0.02, 6.0, 0.004, axes="base")
    pose = TargetResidual(0.02, 6.0, 0.004, axes="base", gripper=False)
    assert (full.width, pose.width) == (7, 6)
    r = torch.randn(5, 7)
    sent = pose.step(base, r[:, :6])
    torch.testing.assert_close(sent[:, :9], full.step(base, r)[:, :9])  # the same pose correction
    torch.testing.assert_close(sent[:, 9:], base[:, 9:])  # the base's gripper, untouched
    with pytest.raises(ValueError, match="6 per step"):
        pose.step(base, r)


def test_dropped_observation_columns_leave_the_agent_and_the_critic_state():
    from my_project.residual import critic_state, observation_keep, residual_observation

    width = 4
    obs = torch.randn(4, width)
    keep = observation_keep(width, [[1, 3], [width - 1, width]])
    assert keep.tolist() == [0] + list(range(3, width - 1))
    base = torch.randn(4, 10)
    agent, base_n = residual_observation(obs, base, True, keep)
    torch.testing.assert_close(agent, torch.cat((obs[:, keep], base), dim=-1))
    critic = critic_state(obs, base, keep=keep)
    torch.testing.assert_close(critic, torch.cat((obs[:, keep], base), dim=-1))
    assert observation_keep(width, []) is None
    with pytest.raises(ValueError, match="runs past"):
        observation_keep(width, [[0, width + 1]])


def test_euler_rpy_is_rz_ry_rx_and_rpy_of_inverts_it():
    from my_project.residual_target import euler_rpy, rpy_of

    def axis(a, i):
        v = torch.zeros(a.shape[0], 3)
        v[:, i] = a
        return exp_so3(v)

    torch.manual_seed(0)
    rpy = torch.rand(16, 3) * 2.0 - 1.0  # pitch inside (-90, 90) deg
    R = euler_rpy(rpy)
    torch.testing.assert_close(R, axis(rpy[:, 2], 2) @ axis(rpy[:, 1], 1) @ axis(rpy[:, 0], 0), atol=1e-6, rtol=0)
    torch.testing.assert_close(rpy_of(R), rpy, atol=1e-5, rtol=0)


def test_euler_rotation_with_base_axes_is_the_rpy_correction_applied_in_the_robot_base_frame():
    from my_project.residual_target import euler_rpy

    target = TargetResidual(0.02, 6.0, 0.004, axes="base", rotation="euler")
    base = random_targets(8)
    r = torch.zeros(8, WIDTH)
    r[:, 3:6] = torch.tensor([0.5, -1.0, 0.25])
    R_sent = rot6d_to_mat(target.step(base, r)[:, 3:9])
    expected = euler_rpy(torch.deg2rad(torch.tensor([3.0, -6.0, 1.5])).expand(8, 3)) @ rot6d_to_mat(base[:, 3:9])
    torch.testing.assert_close(R_sent, expected, atol=1e-5, rtol=0)
    with pytest.raises(ValueError, match="rotation"):
        TargetResidual(0.02, 6.0, 0.004, rotation="quat")


def test_a_per_axis_position_scale_and_the_executed_clip():
    scale = torch.tensor([0.1, 0.2, 0.3])
    low, high = torch.full((10,), -1.0), torch.full((10,), 1.0)
    high[0], high[9] = 0.05, 0.5
    target = TargetResidual(scale, 6.0, 0.25, axes="base", rotation="euler", clip=(low, high))
    base = torch.cat((torch.zeros(4, 3), mat_to_rot6d(torch.eye(3).expand(4, 3, 3)), torch.full((4, 1), 0.4)), dim=-1)
    r = torch.zeros(4, WIDTH)
    r[:, 0:3], r[:, 6] = 1.0, 1.0
    sent = target.step(base, r)
    torch.testing.assert_close(sent[:, 0:3], torch.tensor([0.05, 0.2, 0.3]).expand(4, 3))  # x clamped at its bound
    torch.testing.assert_close(sent[:, 9], torch.full((4,), 0.5))  # 0.4 + 0.25 clamped to 0.5
    with pytest.raises(ValueError, match="pos_scale"):
        TargetResidual(torch.ones(2), 6.0, 0.25)


def test_the_resfit_residual_fields_are_the_target_frames_and_build_its_map():
    from types import SimpleNamespace

    from my_project.residual import target_residual

    for field in ({"target_rotation": "euler"}, {"target_scale_from_base": True}):
        with pytest.raises(ValueError, match="action_frame: target"):
            ResidualCfg(**field).validate(None)
    # executed_clip clamps the target or absolute frame's executed action (ResFiT's clamp), not the ee frame's
    ResidualCfg(action_frame="absolute", executed_clip=1.1).validate(None)
    with pytest.raises(ValueError, match="executed_clip"):
        ResidualCfg(action_frame="ee", executed_clip=1.1).validate(None)
    with pytest.raises(ValueError, match="target_rotation"):
        ResidualCfg(action_frame="target", target_rotation="quat").validate(None)
    with pytest.raises(ValueError, match="executed_clip"):
        ResidualCfg(action_frame="target", executed_clip=0.0).validate(None)
    rc = ResidualCfg(action_frame="target", target_axes="base", target_rotation="euler", target_scale_from_base=True,
                     executed_clip=1.1, ee_rot_scale_deg=30.0)
    rc.validate(None)
    normalizer = SimpleNamespace(action_min=torch.zeros(10), action_max=torch.full((10,), 2.0))
    scale = torch.arange(1.0, 11.0) / 10
    target = target_residual(rc, scale, normalizer, 9)
    torch.testing.assert_close(target.pos_scale, scale[0:3])
    assert target.grip_scale == pytest.approx(1.0) and target.rotation == "euler" and target.axes == "base"
    low, high = target.clip
    torch.testing.assert_close(low, torch.full((10,), -0.1))  # center 1, half range 1, widened to 1.1
    torch.testing.assert_close(high, torch.full((10,), 2.1))
    plain = target_residual(ResidualCfg(action_frame="target"), scale, normalizer, 9)
    assert plain.pos_scale == pytest.approx(0.01) and plain.clip is None and plain.rotation == "rotvec"


def test_the_gripper_gets_its_own_range_per_unit_r():
    from types import SimpleNamespace

    from my_project.residual import target_residual

    normalizer = SimpleNamespace(action_min=torch.full((10,), -1.0), action_max=torch.full((10,), 1.0))
    scale = torch.full((10,), 0.1)  # the absolute frame's: 0.1 of the gripper's half range
    rc = ResidualCfg(action_frame="target", target_axes="base", ee_grip_scale=2.0)
    rc.validate(None)
    target = target_residual(rc, scale, normalizer, 9)
    assert target.grip_scale == 2.0
    base = torch.cat((torch.zeros(1, 3), mat_to_rot6d(torch.eye(3)[None]), torch.tensor([[0.998]])), dim=-1)
    r = torch.zeros(1, WIDTH)
    r[0, 6] = -1.0
    assert float(target.step(base, r)[0, 9]) == pytest.approx(-1.002)  # r = -1 reverses a closed base gripper
    assert target_residual(ResidualCfg(action_frame="target"), scale, normalizer, 9).grip_scale == pytest.approx(0.1)
    with pytest.raises(ValueError, match="ee_grip_scale"):
        ResidualCfg(action_frame="target", ee_grip_scale=0.0).validate(None)
    with pytest.raises(ValueError, match="action_frame: target"):
        ResidualCfg(ee_grip_scale=1.0).validate(None)


def test_the_accumulating_target_residual_carries_its_offset_and_the_critic_rebuilds_the_sent_target():
    torch.manual_seed(0)
    pos, rot = torch.tensor([0.019, 0.015, 0.021]), torch.tensor([8.9, 10.9, 21.7])
    t = TargetResidual(pos, rot, 0.1, axes="base", rotation="euler", gripper=False)
    base = random_targets(5)
    p0, R0 = t.identity(5, "cpu")
    r1, r2 = torch.rand(5, 6) * 2 - 1, torch.rand(5, 6) * 2 - 1
    s1, p1, R1 = t.accumulate(base, r1, p0, R0)
    torch.testing.assert_close(s1, t.step(base, r1), atol=1e-5, rtol=0)  # from a zero offset: the one-step map
    s2, p2, R2 = t.accumulate(base, r2, p1, R1)
    torch.testing.assert_close(s2[:, 0:3] - base[:, 0:3], pos * (r1[:, 0:3] + r2[:, 0:3]))  # the position offset persists
    torch.testing.assert_close(p2, pos * (r1[:, 0:3] + r2[:, 0:3]))
    # the offset observation round-trips, and the critic's side input rebuilds the sent target from it
    obs = t.offset_obs(p1, R1, 8)
    q, Q = t.from_offset_obs(obs, 8)
    torch.testing.assert_close(q, p1) and torch.testing.assert_close(Q, R1, atol=1e-5, rtol=0)
    tr = SideInputTransform(1, 10, target=t, accumulate_reach=8)
    assert tr.side_inputs == {SIDE_INPUT: 16}
    side = torch.cat((base, obs), -1)
    torch.testing.assert_close(tr(None, r2, {SIDE_INPUT: side}), s2, atol=1e-5, rtol=0)


def test_accumulate_and_per_axis_scales_are_validated():
    ResidualCfg(action_frame="target", target_axes="base", target_accumulate=True, critic_state="observation",
                target_pos_scale=[0.019, 0.015, 0.021], target_rot_scale_deg=[8.9, 10.9, 21.7]).validate(None)
    with pytest.raises(ValueError, match="target_axes: base"):
        ResidualCfg(action_frame="target", target_accumulate=True, critic_state="observation").validate(None)
    with pytest.raises(NotImplementedError, match="critic_state: observation"):
        ResidualCfg(action_frame="target", target_axes="base", target_accumulate=True).validate(None)
    with pytest.raises(ValueError, match="target_pos_scale"):
        ResidualCfg(action_frame="target", target_pos_scale=[0.01, 0.0, 0.01]).validate(None)
    with pytest.raises(ValueError, match="action_frame: target"):
        ResidualCfg(target_accumulate=True).validate(None)
