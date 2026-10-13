"""Teleoperation in Forge's action space: a velocity command integrated into a pose setpoint.

A Forge action is **not** a velocity. ``action[0:3] * ctrl.pos_action_bounds`` is the fingertip
target relative to the env's estimate of the hole, ``action[5]`` maps [-1, 1] onto an absolute
world-frame yaw in [-180 deg, +90 deg] (the wrist's joint limit; Forge calls it the "bolt
frame" but never uses the hole's orientation), roll / pitch are overridden, and the env smooths
the action (``ctrl.ema_factor_range``) before using it. So the operator commands velocity and
:class:`SetpointTeleop` keeps the **world pose** they are steering, ``[x, y, z, yaw]`` in the
env (robot base) frame; :func:`world_pose_to_forge_action` turns such a pose into Forge's
action (``my_project/wrappers.py`` uses it for the policy too):

* the setpoint starts at the fingertip's pose at reset (exactly the action Forge itself seeds
  its smoothing with), so the first step does not jump;
* each step it moves by ``command * max_speed * dt``;
* it is **leashed** to within ``leash`` of the current fingertip pose, so a held command
  against contact cannot wind the target into the table: releasing the stick stops the arm,
  and the steady contact force is bounded by roughly ``gain * leash``;
* it is clamped to what Forge can express (the ``pos_action_bounds`` box around the hole and
  the yaw range), so nothing the operator asks for is silently clipped by the env.

Pure torch, one env, no Isaac imports. Quaternions are (w, x, y, z), as Isaac Lab stores them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

__all__ = [
    "YAW_LOW",
    "YAW_HIGH",
    "yaw_to_action",
    "action_to_yaw",
    "fingertip_yaw",
    "world_pose_to_forge_action",
    "TeleopLimits",
    "SetpointTeleop",
]

#: the yaw range Forge's ``action[5]`` spans (forge_env.py: -180 deg + 270 deg * (a + 1) / 2)
YAW_LOW, YAW_HIGH = -math.pi, math.pi / 2


def yaw_to_action(yaw: torch.Tensor) -> torch.Tensor:
    """Hole-frame yaw (rad) -> Forge ``action[5]`` (the inverse of ``_apply_action``)."""
    return (yaw - YAW_LOW) / (YAW_HIGH - YAW_LOW) * 2.0 - 1.0


def action_to_yaw(action: torch.Tensor) -> torch.Tensor:
    return YAW_LOW + (YAW_HIGH - YAW_LOW) * (action + 1.0) / 2.0


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def fingertip_yaw(fingertip_quat: torch.Tensor) -> torch.Tensor:
    """The world-frame yaw Forge's reset derives from the fingertip quat: undo the 180 deg roll,
    take yaw. Matches ``ForgeEnv._reset_idx`` (which wraps into [-pi, pi/2] the same way
    ``atan2`` does over the reachable range).
    """
    unroll = torch.tensor([0.0, -1.0, 0.0, 0.0], dtype=fingertip_quat.dtype, device=fingertip_quat.device)
    w, x, y, z = _quat_mul(unroll.expand_as(fingertip_quat), fingertip_quat).unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def world_pose_to_forge_action(
    pose: torch.Tensor, hole_estimate: torch.Tensor, pos_action_bounds: torch.Tensor
) -> torch.Tensor:
    """``(..., 4)`` world poses ``[x, y, z, yaw]`` -> ``(..., 7)`` Forge actions.

    The inverse of ``ForgeEnv._apply_action``: position relative to the env's hole estimate
    (which ``_apply_action`` adds back, so the estimate's noise cancels and the pose is reached
    in true coordinates), divided by ``ctrl.pos_action_bounds``; yaw rescaled; roll and pitch 0
    (Forge overrides them); the success prediction -1 (unused). Not clamped: Forge accepts any
    value, the bounds only scale.
    """
    action = torch.zeros((*pose.shape[:-1], 7), dtype=pose.dtype, device=pose.device)
    action[..., 0:3] = (pose[..., 0:3] - hole_estimate) / pos_action_bounds
    action[..., 5] = yaw_to_action(pose[..., 3])
    action[..., 6] = -1.0
    return action


@dataclass
class TeleopLimits:
    """Speeds at full stick, the leash, and the env's own action bounds."""

    #: m/s at full stick, per axis
    max_speed_pos: float
    #: rad/s at full stick
    max_speed_yaw: float
    #: the setpoint stays within this of the fingertip, m (per axis) and rad
    leash_pos: float
    leash_yaw: float
    #: Forge's ``ctrl.pos_action_bounds``: action 1.0 is this far from the hole, per axis, m
    pos_action_bounds: Sequence[float]


class SetpointTeleop:
    """One env's setpoint. ``step`` returns the world pose ``[x, y, z, yaw]`` it now targets."""

    def __init__(self, limits: TeleopLimits, dt: float, device="cpu") -> None:
        self.limits = limits
        self.dt = float(dt)
        self.device = torch.device(device)
        self.bounds = torch.tensor(limits.pos_action_bounds, dtype=torch.float32, device=self.device)
        self.pos = torch.zeros(3, device=self.device)
        self.yaw = torch.zeros((), device=self.device)

    def reset(self, fingertip_pos: torch.Tensor, fingertip_quat: torch.Tensor) -> None:
        """Start at the current pose (Forge seeds its smoothing with the same action)."""
        self.pos = fingertip_pos.detach().to(self.device, torch.float32).clone()
        self.yaw = fingertip_yaw(fingertip_quat.detach().to(self.device, torch.float32))

    def step(
        self,
        command: torch.Tensor,
        scale: float,
        fingertip_pos: torch.Tensor,
        fingertip_quat: torch.Tensor,
        hole_pos: torch.Tensor,
    ) -> torch.Tensor:
        """``command`` is (vx, vy, vz, yaw rate) in [-1, 1]; ``scale`` multiplies the speed.

        ``hole_pos`` is the env's hole estimate (``fixed_pos_obs_frame +
        init_fixed_pos_obs_noise``), around which Forge's action box is centred.
        """
        lim = self.limits
        command = command.to(self.device, torch.float32).clamp(-1.0, 1.0)
        here = fingertip_pos.to(self.device, torch.float32)
        hole = hole_pos.to(self.device, torch.float32)
        yaw_here = fingertip_yaw(fingertip_quat.to(self.device, torch.float32))

        pos = self.pos + command[:3] * lim.max_speed_pos * scale * self.dt
        pos = torch.minimum(torch.maximum(pos, here - lim.leash_pos), here + lim.leash_pos)
        pos = torch.minimum(torch.maximum(pos, hole - self.bounds), hole + self.bounds)

        yaw = self.yaw + command[3] * lim.max_speed_yaw * scale * self.dt
        yaw = yaw.clamp(yaw_here - lim.leash_yaw, yaw_here + lim.leash_yaw)
        yaw = yaw.clamp(YAW_LOW, YAW_HIGH)
        self.pos, self.yaw = pos, yaw

        return torch.cat([pos, yaw.reshape(1)])
