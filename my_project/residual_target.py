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

__all__ = ["TargetResidual", "EERelative", "WIDTH"]

#: the agent's per-step action: position 3, rotation 3, gripper 1
WIDTH = 7


class TargetResidual:
    """``axes`` (``residual.target_axes``): "target" composes the correction in the base target's own frame
    (``p_base + R_base (pos_scale r_pos)``, ``R_base Exp(rot_scale r_rot)``); "base" adds it in the robot-base frame
    the absolute targets are in (``p_base + pos_scale r_pos``, ``Exp(rot_scale r_rot) R_base``), so r = 1 is the same
    fixed absolute correction whatever the target's orientation."""

    AXES = ("target", "base")

    def __init__(self, pos_scale: float, rot_scale_deg: float, grip_scale: float, axes: str = "target",
                 gripper: bool = True) -> None:
        if axes not in self.AXES:
            raise ValueError(f"TargetResidual axes must be one of {self.AXES}, got {axes!r}")
        self.pos_scale = float(pos_scale)
        self.rot_scale = math.radians(float(rot_scale_deg))
        self.grip_scale = float(grip_scale)
        self.axes = axes
        #: ``residual.residual_gripper``: r carries r_grip (7 per step), or the pose alone (6) and the gripper is the base's
        self.gripper = bool(gripper)
        self.width = WIDTH if self.gripper else WIDTH - 1

    def step(self, base: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """``base`` ``(..., 10)`` absolute targets, ``r`` ``(..., width)`` -> the sent targets ``(..., 10)``."""
        if r.shape[-1] != self.width:
            raise ValueError(f"the target-frame residual is {self.width} per step, got {r.shape[-1]}")
        R_base = rot6d_to_mat(base[..., 3:9])
        gripper = base[..., 9:10] + self.grip_scale * r[..., 6:7] if self.gripper else base[..., 9:10]
        if self.axes == "base":
            dp = self.pos_scale * r[..., 0:3]
            R = exp_so3(self.rot_scale * r[..., 3:6]) @ R_base
            return torch.cat((base[..., 0:3] + dp, mat_to_rot6d(R), gripper), dim=-1)
        dp = self.pos_scale * (R_base @ r[..., 0:3].unsqueeze(-1)).squeeze(-1)
        R = R_base @ exp_so3(self.rot_scale * r[..., 3:6])
        return torch.cat((base[..., 0:3] + dp, mat_to_rot6d(R), gripper), dim=-1)


class EERelative:
    """``residual.ee_relative_actions``: absolute targets ``(N, k, 10)`` -> Policy Decorator's command form,
    relative to the current EE pose ``ee`` ``(N, 9)`` (the observation's ``ee_pos 3, ee_rot6d 6``, raw) in the
    EE's frame -- ManiSkill2's ``pd_ee_delta_pose`` with ``frame: ee`` (``target = ee_pose * delta``) -- per target
    ``[R_ee^T (p - p_ee) / pos_unit, log(R_ee^T R) / rot_unit, normalize(gripper)]`` (7), flattened step-major
    ``(N, k x 7)``. Every target of a chunk is taken against the EE pose of the chunk's observation. Plain torch,
    gradients finite (``log_so3``): the critics' executed action goes through it."""

    def __init__(self, pos_unit: float, rot_unit: float, normalizer, gripper: int = 9) -> None:
        self.pos_unit = float(pos_unit)
        self.rot_unit = float(rot_unit)
        self.normalizer = normalizer
        self.gripper = int(gripper)

    def __call__(self, targets: torch.Tensor, ee: torch.Tensor) -> torch.Tensor:
        from my_project.residual_ee import log_so3

        n, k = targets.shape[0], targets.shape[1]
        p_ee = ee[:, None, 0:3]
        R_ee = rot6d_to_mat(ee[:, 3:9])[:, None]  # (N, 1, 3, 3)
        R_eeT = R_ee.transpose(-1, -2)
        dp = (R_eeT @ (targets[..., 0:3] - p_ee).unsqueeze(-1)).squeeze(-1) / self.pos_unit
        dr = log_so3(R_eeT @ rot6d_to_mat(targets[..., 3:9])) / self.rot_unit
        grip = self.normalizer.normalize(targets, "action")[..., self.gripper : self.gripper + 1]
        return torch.cat((dp, dr, grip), dim=-1).reshape(n, k * 7)
