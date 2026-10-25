"""``residual.action_frame: ee``: the residual as a correction in the end-effector's frame, on an absolute base.

The base (an FBAbs policy, ``[goal_pos 3, goal_rot6d 6, gripper]`` in the robot-base frame) is left as it
is. The agent's action is ``r = [r_pos 3, r_rot 3]`` in [-1, 1] (6; the gripper is the base's):

* each step's increment is ``dp = ee_pos_scale * R_ee r_pos`` -- along the EE's own axes, ``R_ee`` the
  orientation the observation reports -- and ``dR = Exp(ee_rot_scale * r_rot)``, a rotation about the
  target's own axes;
* without ``residual.accumulate`` (A) the offset is that increment, re-sent every step: ``o = dp``,
  ``O = dR``;
* with ``residual.accumulate`` (B) the increments add up, ``o += dp``, ``O = O dR``, and the offset
  goes back to zero whenever the base starts a new chunk (every ``action_horizon`` steps, and at an
  episode's start): a correction persists until the base re-plans. The offset is part of the agent's
  and the critic's observation (:meth:`EEResidual.offset_obs`), in the EE frame, scaled by its largest
  reach in one chunk;
* the env gets ``[p_base + o, rot6d(R_base O), gripper_base]``.

The critics' executed action (``residual.critic_action: full``) is the offset after the step, ``(o, O)``, as
:meth:`EEResidual.offset_obs` puts it: with ``normalize(a_base)`` in their input it determines the target sent
(and a demo's needed offset is stored exactly, however far beyond the agent's reach), in the agent's own
6 dims, as the learner requires.

One map for the env wrapper (``my_project/residual.py``), the critics' executed action
(``residual.critic_action: full``) and the demo conversion (``my_project/residual_demos.py``, through
:meth:`EEResidual.invert`). Plain torch.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch

from my_project.models.action_frame import mat_to_rot6d, rot6d_to_mat

__all__ = ["EEResidual", "WIDTH", "OFFSET_WIDTH", "ee_rotation", "exp_so3", "log_so3"]

#: the agent's action, and the accumulated offset's observation
WIDTH, OFFSET_WIDTH = 6, 6


#: inside the square roots below: keeps every gradient finite at the identity (a norm's is 0 / 0 there)
_EPS = 1e-12


def exp_so3(v: torch.Tensor) -> torch.Tensor:
    """``(..., 3)`` axis-angle (rad) -> ``(..., 3, 3)`` by Rodrigues, with gradients finite everywhere (the
    actor's action goes through it into the critics, :func:`~my_project.residual.ee_full_action_transform`)."""
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    zero = torch.zeros_like(x)
    K = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(*v.shape[:-1], 3, 3)
    t2 = (v * v).sum(-1)[..., None, None]
    t = torch.sqrt(t2 + _EPS)
    small = t2 < 1e-8
    a = torch.where(small, 1 - t2 / 6, torch.sin(t) / t)
    b = torch.where(small, 0.5 - t2 / 24, (1 - torch.cos(t)) / (t2 + _EPS))
    return torch.eye(3, dtype=v.dtype, device=v.device).expand(K.shape) + a * K + b * (K @ K)


def log_so3(R: torch.Tensor) -> torch.Tensor:
    """``(..., 3, 3)`` -> axis-angle ``(..., 3)`` for rotations under pi, with gradients finite everywhere there:
    ``v = vee(R - R^T) / 2 = sin(theta) axis``, ``theta = atan2(|v|, (tr R - 1) / 2)``, ``theta / sin(theta) v``.
    (No square root at zero, unlike a quaternion route; the offsets this maps stay far below pi.)"""
    v = 0.5 * torch.stack((R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]), dim=-1)
    c = 0.5 * (R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0)
    s2 = (v * v).sum(-1)
    s = torch.sqrt(s2 + _EPS)
    theta = torch.atan2(s, c)
    scale = torch.where(s2 < 1e-8, 1 + s2 / 6, theta / s)  # theta / sin(theta) -> 1 at the identity
    return v * scale[..., None]


def ee_rotation(obs: torch.Tensor) -> torch.Tensor:
    """An FBAbs observation (raw units; ``ee_pos 3, ee_rot6d 6`` first) -> the EE's ``(..., 3, 3)``."""
    return rot6d_to_mat(obs[..., 3:9])


class EEResidual:
    def __init__(self, pos_scale: float, rot_scale_deg: float, accumulate: bool, horizon: int) -> None:
        self.pos_scale = float(pos_scale)
        self.rot_scale = math.radians(float(rot_scale_deg))
        self.accumulate = bool(accumulate)
        self.horizon = int(horizon)
        #: the offset's largest reach in steps: a whole chunk when it accumulates, one step otherwise
        self.reach = self.horizon if self.accumulate else 1

    def identity(self, n: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        """No offset: ``(o (n, 3), O (n, 3, 3))``."""
        return torch.zeros(n, 3, device=device), torch.eye(3, device=device).expand(n, 3, 3).clone()

    def step(self, base: torch.Tensor, R_ee: torch.Tensor, o: torch.Tensor, O: torch.Tensor, r: torch.Tensor):
        """``base`` ``(N, 10)``, the offset ``(o, O)`` before this step, ``r`` ``(N, 6)`` -> ``(sent (N, 10), o, O)``
        after it."""
        dp = self.pos_scale * (R_ee @ r[..., 0:3].unsqueeze(-1)).squeeze(-1)
        dR = exp_so3(self.rot_scale * r[..., 3:6])
        if self.accumulate:
            o, O = o + dp, O @ dR
        else:
            o, O = dp, dR
        sent = torch.cat((base[..., 0:3] + o, mat_to_rot6d(rot6d_to_mat(base[..., 3:9]) @ O), base[..., 9:]), dim=-1)
        return sent, o, O

    def invert(self, base: torch.Tensor, target: torch.Tensor, R_ee: torch.Tensor, o: torch.Tensor, O: torch.Tensor):
        """The ``r`` that turns ``base`` into the absolute ``target`` (``(N, 10)``, its gripper aside) from the offset
        ``(o, O)`` before the step -- :meth:`step`'s inverse, exact when ``r`` comes out in [-1, 1] (and the
        rotation increment under pi) -- and the offset after it: ``(r (N, 6), o, O)``."""
        o_new = target[..., 0:3] - base[..., 0:3]
        O_new = rot6d_to_mat(base[..., 3:9]).transpose(-1, -2) @ rot6d_to_mat(target[..., 3:9])
        if self.accumulate:
            dp, dR = o_new - o, O.transpose(-1, -2) @ O_new
        else:
            dp, dR = o_new, O_new
        r_pos = (R_ee.transpose(-1, -2) @ dp.unsqueeze(-1)).squeeze(-1) / self.pos_scale
        r_rot = log_so3(dR) / self.rot_scale
        return torch.cat((r_pos, r_rot), dim=-1), o_new, O_new

    def offset_obs(self, R_ee: torch.Tensor, o: torch.Tensor, O: torch.Tensor) -> torch.Tensor:
        """An offset as the agent and the critics see it, ``(N, 6)``: ``R_ee^T o`` and ``log(O)``, each over its
        largest reach (:attr:`reach` steps of the step scale: a whole chunk of +1 is 1 when it accumulates, one
        step of +1 is 1 when it does not)."""
        pos = (R_ee.transpose(-1, -2) @ o.unsqueeze(-1)).squeeze(-1) / (self.reach * self.pos_scale)
        rot = log_so3(O) / (self.reach * self.rot_scale)
        return torch.cat((pos, rot), dim=-1)

    def from_offset_obs(self, R_ee: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """:meth:`offset_obs`'s inverse: ``(o, O)``."""
        o = (R_ee @ (x[..., 0:3] * (self.reach * self.pos_scale)).unsqueeze(-1)).squeeze(-1)
        return o, exp_so3(x[..., 3:6] * (self.reach * self.rot_scale))
