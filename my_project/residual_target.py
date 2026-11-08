"""``residual.action_frame: target``: Policy Decorator's meaning of the residual, on an absolute base.

Policy Decorator acts in ManiSkill2's ``pd_ee_delta_pose`` (``[-1, 1]`` scaled to +-0.1 m and a 0.1 rad
rotation vector, composed on the EE pose in the EE's frame: ``target = ee_pose * delta``) and the gripper's
absolute ``pd_joint_pos``; its residual is ``res_scale * r`` (0.1) added to the base's command, so ``r = 1`` is
1 cm, 0.01 rad and a tenth of the gripper's half range. Our base gives absolute targets
``[goal_pos 3, goal_rot6d 6, gripper]`` in the robot-base frame; the residual ``r = [r_pos 3, r_rot 3, r_grip 1]``
per step (7) is the same relative pose composed on the base's target in the target's own frame:

* ``p = p_base + R_base (pos_scale * r_pos)``, ``R = R_base Exp(rot_scale * r_rot)``;
* ``gripper = gripper_base + grip_scale * r_grip`` (the absolute frame's gripper scale, ``residual_scale``).

The base target's axes stand in for the measured EE's so that the executed chunk is a function of the base's
chunk and ``r`` alone: the critics' executed action (``residual.critic_action: full``) is rebuilt from them.
One map for the env wrapper and the critics. Plain torch, gradients finite everywhere (``exp_so3``).
"""

from __future__ import annotations

import math

import torch

from my_project.models.action_frame import mat_to_rot6d, rot6d_to_mat
from my_project.residual_ee import exp_so3

__all__ = ["TargetResidual", "EERelative", "WIDTH", "euler_rpy", "rpy_of"]

#: the agent's per-step action: position 3, rotation 3, gripper 1
WIDTH = 7


def euler_rpy(v: torch.Tensor) -> torch.Tensor:
    """``(..., 3)`` roll, pitch, yaw (rad) -> ``R = Rz(yaw) Ry(pitch) Rx(roll)`` ``(..., 3, 3)`` (fixed-axis x-y-z)."""
    cr, sr = torch.cos(v[..., 0]), torch.sin(v[..., 0])
    cp, sp = torch.cos(v[..., 1]), torch.sin(v[..., 1])
    cy, sy = torch.cos(v[..., 2]), torch.sin(v[..., 2])
    rows = (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )
    return torch.stack([torch.stack(row, dim=-1) for row in rows], dim=-2)


def rpy_of(R: torch.Tensor) -> torch.Tensor:
    """:func:`euler_rpy`'s inverse for pitch inside (-90, 90) deg: ``(..., 3, 3)`` -> ``(..., 3)`` roll, pitch, yaw."""
    roll = torch.atan2(R[..., 2, 1], R[..., 2, 2])
    pitch = torch.asin((-R[..., 2, 0]).clamp(-1.0, 1.0))
    yaw = torch.atan2(R[..., 1, 0], R[..., 0, 0])
    return torch.stack((roll, pitch, yaw), dim=-1)


class TargetResidual:
    """``axes`` (``residual.target_axes``): "target" composes the correction in the base target's own frame
    (``p_base + R_base (pos_scale r_pos)``, ``R_base Exp(rot_scale r_rot)``); "base" adds it in the robot-base frame
    the absolute targets are in (``p_base + pos_scale r_pos``, ``Exp(rot_scale r_rot) R_base``), so r = 1 is the same
    fixed absolute correction whatever the target's orientation.

    ``rotation`` (``residual.target_rotation``): "rotvec", ``r_rot`` an axis-angle (``Exp``); "euler", roll, pitch,
    yaw (:func:`euler_rpy`) in place of ``Exp``. ``pos_scale`` is one value or one per axis (3; ``residual.
    target_scale_from_base``: the base's own half range). ``clip`` ``(low, high)`` raw ``(10,)`` bounds clamp the sent
    position and gripper (``residual.executed_clip``; the rotation is never clamped)."""

    AXES = ("target", "base")
    ROTATIONS = ("rotvec", "euler")

    def __init__(self, pos_scale, rot_scale_deg: float, grip_scale: float, axes: str = "target",
                 gripper: bool = True, rotation: str = "rotvec", clip=None) -> None:
        if axes not in self.AXES:
            raise ValueError(f"TargetResidual axes must be one of {self.AXES}, got {axes!r}")
        if rotation not in self.ROTATIONS:
            raise ValueError(f"TargetResidual rotation must be one of {self.ROTATIONS}, got {rotation!r}")
        self.pos_scale = pos_scale if isinstance(pos_scale, torch.Tensor) else float(pos_scale)
        if isinstance(self.pos_scale, torch.Tensor) and self.pos_scale.shape != (3,):
            raise ValueError(f"TargetResidual pos_scale is one value or 3, got shape {tuple(self.pos_scale.shape)}")
        #: rad per unit r_rot: one value, or one per axis (3)
        self.rot_scale = (torch.deg2rad(rot_scale_deg.to(torch.float32)) if isinstance(rot_scale_deg, torch.Tensor)
                          else math.radians(float(rot_scale_deg)))
        if isinstance(self.rot_scale, torch.Tensor) and self.rot_scale.shape != (3,):
            raise ValueError(f"TargetResidual rot_scale_deg is one value or 3, got shape {tuple(self.rot_scale.shape)}")
        self.grip_scale = float(grip_scale)
        self.axes = axes
        self.rotation = rotation
        #: ``(low, high)`` raw ``(10,)`` bounds on the sent target's position and gripper, or None
        self.clip = clip
        #: ``residual.residual_gripper``: r carries r_grip (7 per step), or the pose alone (6) and the gripper is the base's
        self.gripper = bool(gripper)
        self.width = WIDTH if self.gripper else WIDTH - 1

    def accumulate(self, base: torch.Tensor, r: torch.Tensor, off_pos: torch.Tensor, off_rot: torch.Tensor):
        """``residual.target_accumulate`` (base axes): the residual is a per-step increment of an offset the base's
        targets carry, ``off_pos' = off_pos + pos_scale r_pos`` and ``off_rot' = rotate(r_rot) off_rot``, sent as
        ``[p_base + off_pos', rot6d(off_rot' R_base), gripper]`` -- so a step's correction persists into the next steps
        (until the env resets the offset when the base replans), as a delta command's would. ``base`` ``(..., 10)``,
        ``r`` ``(..., width)``, ``off_pos`` ``(..., 3)`` m, ``off_rot`` ``(..., 3, 3)``. With ``clip`` the sent position is
        clamped and the offset made the one actually sent. Returns ``(sent, off_pos', off_rot')``."""
        if self.axes != "base":
            raise ValueError("the accumulating target residual is built for target_axes: base")
        if r.shape[-1] != self.width:
            raise ValueError(f"the target-frame residual is {self.width} per step, got {r.shape[-1]}")
        off_pos = off_pos + self.pos_scale * r[..., 0:3]
        off_rot = self.rotate(r[..., 3:6]) @ off_rot
        gripper = base[..., 9:10] + self.grip_scale * r[..., 6:7] if self.gripper else base[..., 9:10]
        position = base[..., 0:3] + off_pos
        if self.clip is not None:
            low, high = self.clip
            position = torch.maximum(torch.minimum(position, high[0:3]), low[0:3])
            gripper = torch.maximum(torch.minimum(gripper, high[9:10]), low[9:10])
            off_pos = position - base[..., 0:3]
        R = off_rot @ rot6d_to_mat(base[..., 3:9])
        return torch.cat((position, mat_to_rot6d(R), gripper), dim=-1), off_pos, off_rot

    def offset_obs(self, off_pos: torch.Tensor, off_rot: torch.Tensor, reach: int) -> torch.Tensor:
        """An accumulated offset as the agent and the critics see it, ``(..., 6)``: its position over ``reach`` steps of
        the position scale, and its roll / pitch / yaw (or axis-angle) over ``reach`` steps of the rotation scale --
        about [-1, 1] for a chunk of full-reach steps."""
        from my_project.residual_ee import log_so3

        rot = rpy_of(off_rot) if self.rotation == "euler" else log_so3(off_rot)
        return torch.cat((off_pos / (reach * self.pos_scale), rot / (reach * self.rot_scale)), dim=-1)

    def from_offset_obs(self, x: torch.Tensor, reach: int):
        """:meth:`offset_obs`'s inverse: ``(off_pos, off_rot)``."""
        pos = x[..., 0:3] * (reach * self.pos_scale)
        v = x[..., 3:6] * (reach * self.rot_scale)
        return pos, (euler_rpy(v) if self.rotation == "euler" else exp_so3(v))

    def identity(self, n: int, device):
        """A zero offset for ``n`` envs: ``(off_pos (n, 3), off_rot (n, 3, 3))``."""
        return torch.zeros((n, 3), device=device), torch.eye(3, device=device).expand(n, 3, 3).clone()

    def rotate(self, r_rot: torch.Tensor) -> torch.Tensor:
        """The correction's rotation matrix for ``r_rot`` ``(..., 3)``."""
        v = self.rot_scale * r_rot
        return euler_rpy(v) if self.rotation == "euler" else exp_so3(v)

    def step(self, base: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """``base`` ``(..., 10)`` absolute targets, ``r`` ``(..., width)`` -> the sent targets ``(..., 10)``."""
        if r.shape[-1] != self.width:
            raise ValueError(f"the target-frame residual is {self.width} per step, got {r.shape[-1]}")
        R_base = rot6d_to_mat(base[..., 3:9])
        gripper = base[..., 9:10] + self.grip_scale * r[..., 6:7] if self.gripper else base[..., 9:10]
        if self.axes == "base":
            dp = self.pos_scale * r[..., 0:3]
            R = self.rotate(r[..., 3:6]) @ R_base
        else:
            dp = (R_base @ (self.pos_scale * r[..., 0:3]).unsqueeze(-1)).squeeze(-1)
            R = R_base @ self.rotate(r[..., 3:6])
        position = base[..., 0:3] + dp
        if self.clip is not None:
            low, high = self.clip
            position = torch.maximum(torch.minimum(position, high[0:3]), low[0:3])
            gripper = torch.maximum(torch.minimum(gripper, high[9:10]), low[9:10])
        return torch.cat((position, mat_to_rot6d(R), gripper), dim=-1)


class EERelative:
    """``residual.ee_relative_actions``: absolute targets ``(N, k, 10)`` -> Policy Decorator's command form, relative to
    the current EE pose ``ee`` ``(N, 9)`` (the observation's ``ee_pos 3, ee_rot6d 6``, raw) in the EE's frame --
    ManiSkill2's ``pd_ee_delta_pose`` with ``frame: ee`` (``target = ee_pose * delta``) -- per target ``[R_ee^T (p -
    p_ee), log(R_ee^T R), gripper]`` (7: m, rad, the gripper command; raw -- the learner's ``state_preprocessor``
    normalizes it), flattened step-major ``(N, k x 7)``. Every target of a chunk is taken against the EE pose of the
    chunk's observation. Plain torch, gradients finite (``log_so3``): the critics' executed action goes through it."""

    def __init__(self, gripper: int = 9) -> None:
        self.gripper = int(gripper)

    def __call__(self, targets: torch.Tensor, ee: torch.Tensor) -> torch.Tensor:
        from my_project.residual_ee import log_so3

        n, k = targets.shape[0], targets.shape[1]
        p_ee = ee[:, None, 0:3]
        R_ee = rot6d_to_mat(ee[:, 3:9])[:, None]  # (N, 1, 3, 3)
        R_eeT = R_ee.transpose(-1, -2)
        dp = (R_eeT @ (targets[..., 0:3] - p_ee).unsqueeze(-1)).squeeze(-1)
        dr = log_so3(R_eeT @ rot6d_to_mat(targets[..., 3:9]))
        grip = targets[..., self.gripper : self.gripper + 1]
        return torch.cat((dp, dr, grip), dim=-1).reshape(n, k * 7)
