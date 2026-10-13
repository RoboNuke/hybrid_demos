"""SetpointTeleop: Forge's action conversions, the integration, the leash and the bounds."""

from __future__ import annotations

import math

import pytest
import torch

from my_project.demos.teleop import (
    YAW_HIGH,
    YAW_LOW,
    SetpointTeleop,
    TeleopLimits,
    action_to_yaw,
    fingertip_yaw,
    world_pose_to_forge_action,
    yaw_to_action,
)

DT = 1 / 15
BOUNDS = [0.05, 0.05, 0.05]


def euler_quat(roll: float, pitch: float, yaw: float) -> torch.Tensor:
    """(w, x, y, z) for intrinsic x-y-z, like isaacsim's quat_from_euler_xyz."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return torch.tensor([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def quat_mul(a, b):
    from my_project.demos.teleop import _quat_mul

    return _quat_mul(a, b)


def gripper_down(yaw: float) -> torch.Tensor:
    """Forge's target orientation for a hole-frame yaw: roll 180 deg, then the yaw."""
    return quat_mul(euler_quat(math.pi, 0.0, 0.0), euler_quat(0.0, 0.0, yaw))


def make(**overrides) -> SetpointTeleop:
    limits = dict(max_speed_pos=0.03, max_speed_yaw=0.5, leash_pos=0.01, leash_yaw=0.15, pos_action_bounds=BOUNDS)
    limits.update(overrides)
    return SetpointTeleop(TeleopLimits(**limits), dt=DT)


# ------------------------------------------------------------------ Forge's own conversions
def test_yaw_action_mapping_is_forges_and_inverts():
    a = torch.linspace(-1, 1, 11)
    assert torch.allclose(action_to_yaw(a), math.radians(-180) + math.radians(270) * (a + 1) / 2)
    assert torch.allclose(yaw_to_action(action_to_yaw(a)), a, atol=1e-6)
    assert action_to_yaw(torch.tensor(-1.0)) == YAW_LOW and torch.isclose(action_to_yaw(torch.tensor(1.0)), torch.tensor(YAW_HIGH))


@pytest.mark.parametrize("yaw", [-3.0, -1.2, 0.0, 0.7, 1.5])
def test_fingertip_yaw_recovers_the_yaw_forge_commanded(yaw):
    assert math.isclose(float(fingertip_yaw(gripper_down(yaw))), yaw, abs_tol=1e-5)


def test_reset_then_still_stick_targets_the_current_pose_which_is_forges_reset_action():
    teleop = make()
    hole = torch.tensor([0.6, 0.0, 0.05])
    here = hole + torch.tensor([0.01, -0.02, 0.04])
    quat = gripper_down(0.3)
    teleop.reset(here, quat)
    pose = teleop.step(torch.zeros(4), 1.0, here, quat, hole)
    assert torch.allclose(pose, torch.cat([here, torch.tensor([0.3])]), atol=1e-6)
    # ForgeEnv._reset_idx seeds its smoothing with exactly this action
    action = world_pose_to_forge_action(pose, hole, torch.tensor(BOUNDS))
    assert torch.allclose(action[0:3], (here - hole) / torch.tensor(BOUNDS), atol=1e-6)
    assert math.isclose(float(action[5]), (0.3 + math.pi) / math.radians(270) * 2 - 1, abs_tol=1e-5)
    assert action[3] == 0 and action[4] == 0 and action[6] == -1


def test_world_pose_conversion_is_batched_and_inverts_forges_apply_action():
    hole = torch.tensor([[0.6, 0.0, 0.05], [0.55, 0.02, 0.06]])
    pose = torch.tensor([[0.61, -0.01, 0.09, -1.0], [0.5, 0.03, 0.07, 1.2]])
    bounds = torch.tensor(BOUNDS)
    action = world_pose_to_forge_action(pose, hole, bounds)
    assert action.shape == (2, 7)
    # Forge: target = hole estimate + action * bounds; yaw = -180 + 270 * (a + 1) / 2 deg
    assert torch.allclose(hole + action[:, 0:3] * bounds, pose[:, 0:3], atol=1e-6)
    assert torch.allclose(action_to_yaw(action[:, 5]), pose[:, 3], atol=1e-6)


# ------------------------------------------------------------------ integration, leash, bounds
def test_full_stick_moves_the_setpoint_at_max_speed_and_fine_mode_scales_it():
    teleop = make()
    hole, quat = torch.zeros(3), gripper_down(0.0)
    here = torch.tensor([0.0, 0.0, 0.03])
    teleop.reset(here, quat)
    teleop.step(torch.tensor([1.0, 0.0, -1.0, 0.0]), 1.0, here, quat, hole)
    assert torch.allclose(teleop.pos - here, torch.tensor([0.03 * DT, 0.0, -0.03 * DT]))
    teleop.reset(here, quat)
    teleop.step(torch.tensor([1.0, 0.0, 0.0, 1.0]), 0.25, here, quat, hole)
    assert math.isclose(float(teleop.pos[0] - here[0]), 0.25 * 0.03 * DT, rel_tol=1e-5)
    assert math.isclose(float(teleop.yaw), 0.25 * 0.5 * DT, rel_tol=1e-5)


def test_the_leash_stops_a_blocked_push_from_winding_up():
    teleop = make()
    hole, quat = torch.zeros(3), gripper_down(0.0)
    here = torch.tensor([0.0, 0.0, 0.0])  # blocked: the fingertip never moves
    teleop.reset(here, quat)
    for _ in range(200):
        teleop.step(torch.tensor([0.0, 0.0, -1.0, 1.0]), 1.0, here, quat, hole)
    assert math.isclose(float(teleop.pos[2]), -0.01, abs_tol=1e-6)
    assert math.isclose(float(teleop.yaw), 0.15, abs_tol=1e-6)


def test_the_setpoint_stays_inside_what_forge_can_express():
    teleop = make(leash_pos=1.0, leash_yaw=10.0)
    hole, quat = torch.zeros(3), gripper_down(1.5)
    here = torch.tensor([0.0, 0.0, 0.049])
    teleop.reset(here, quat)
    for _ in range(100):
        pose = teleop.step(torch.tensor([0.0, 0.0, 1.0, 1.0]), 1.0, here, quat, hole)
    action = world_pose_to_forge_action(pose, hole, torch.tensor(BOUNDS))
    assert math.isclose(float(pose[2]), 0.05, abs_tol=1e-6) and math.isclose(float(action[2]), 1.0, abs_tol=1e-6)
    assert math.isclose(float(pose[3]), YAW_HIGH, abs_tol=1e-6) and math.isclose(float(action[5]), 1.0, abs_tol=1e-6)
