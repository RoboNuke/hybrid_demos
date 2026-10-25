"""``model.actor.action_frame: ee_at_obs``: the policy's chunk relative to the EE pose at its observation.

The env keeps the absolute target action of the FBAbs tasks, ``[goal_pos 3, goal_rot6d 6, gripper]``
(10, robot-base frame). The network instead predicts every step of its chunk in the frame of the EE
pose the observation it acted on reports, ``(p, R)`` = the observation's ``ee_pos`` / ``ee_rot6d``
(its first 9 dims in every FBAbs layout):

    ``[dpos 3, drot 3, gripper]`` (7) with ``dpos = R^T (goal_pos - p)`` in m and
    ``drot = log(R^T R_goal)`` an axis-angle in rad, neither scaled; the gripper as it is.

All the chunk's steps share that one pose (not the pose each step ends up at), so the chunk is a
trajectory anchored where the robot was when it was planned; zeros are "go back to / stay at that
pose". :func:`from_ee_frame` turns a predicted chunk into the env's absolute targets with the same
pose; :func:`to_ee_frame` builds the training labels from ``action.abs``; :func:`reframe` moves a
chunk planned at one pose to another (the warm start's unexecuted tail).

Plain torch, no Isaac Lab. rot6d is a matrix's first two rows (``furniture_lab``'s convention).
"""

from __future__ import annotations

from typing import Tuple

import torch

__all__ = [
    "FRAMES", "ABSOLUTE", "EE_AT_OBS", "NET_DIM", "ENV_DIM", "METRIC_GROUPS", "ee_pose", "to_ee_frame",
    "from_ee_frame", "reframe", "rot6d_to_mat", "mat_to_rot6d", "axis_angle_to_mat", "mat_to_axis_angle",
]

ABSOLUTE, EE_AT_OBS = "absolute", "ee_at_obs"
FRAMES = (ABSOLUTE, EE_AT_OBS)
#: the network's action in the EE frame, and the env's absolute target it becomes
NET_DIM, ENV_DIM = 7, 10
#: ``bc.action_metric_groups`` on the network's action
METRIC_GROUPS = {"pos": [0, 1, 2], "rot": [3, 4, 5], "gripper": [6]}


def rot6d_to_mat(d6: torch.Tensor) -> torch.Tensor:
    """``(..., 6)`` first two rows -> ``(..., 3, 3)`` by Gram-Schmidt."""
    b1 = torch.nn.functional.normalize(d6[..., 0:3], dim=-1)
    a2 = d6[..., 3:6]
    b2 = torch.nn.functional.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    return torch.stack((b1, b2, torch.cross(b1, b2, dim=-1)), dim=-2)


def mat_to_rot6d(R: torch.Tensor) -> torch.Tensor:
    return R[..., :2, :].reshape(*R.shape[:-2], 6)


def axis_angle_to_mat(v: torch.Tensor) -> torch.Tensor:
    """``(..., 3)`` axis-angle (rad) -> ``(..., 3, 3)`` (Rodrigues); zero is the identity."""
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    zero = torch.zeros_like(x)
    K = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(*v.shape[:-1], 3, 3)
    theta = v.norm(dim=-1)[..., None, None]
    small = theta < 1e-6
    t = torch.where(small, torch.ones_like(theta), theta)
    a = torch.where(small, 1 - theta**2 / 6, torch.sin(t) / t)
    b = torch.where(small, 0.5 - theta**2 / 24, (1 - torch.cos(t)) / t**2)
    return torch.eye(3, dtype=v.dtype, device=v.device).expand(K.shape) + a * K + b * (K @ K)


def _mat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """``(..., 3, 3)`` -> ``(..., 4)`` wxyz with ``w >= 0`` (Shepperd's method)."""
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    q_abs = torch.sqrt(torch.clamp(torch.stack(
        (1 + m00 + m11 + m22, 1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22), dim=-1), min=0.0))
    candidates = torch.stack((
        torch.stack((q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01), dim=-1),
        torch.stack((m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20), dim=-1),
        torch.stack((m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21), dim=-1),
        torch.stack((m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2), dim=-1),
    ), dim=-2)
    candidates = candidates / (2.0 * q_abs[..., None].clamp(min=0.1))  # no host tensor: CUDA-graph safe
    best = q_abs.argmax(dim=-1)
    q = torch.gather(candidates, -2, best[..., None, None].expand(*best.shape, 1, 4)).squeeze(-2)
    q = torch.where(q[..., :1] < 0, -q, q)
    return q / q.norm(dim=-1, keepdim=True)


def mat_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    """``(..., 3, 3)`` -> its axis-angle ``(..., 3)``, angle in [0, pi] rad; the identity -> 0."""
    q = _mat_to_quat(R)
    w, v = q[..., :1], q[..., 1:4]
    s = v.norm(dim=-1, keepdim=True)
    scale = torch.where(s > 1e-8, 2 * torch.atan2(s, w) / s.clamp(min=1e-8), 2 / w.clamp(min=1e-8))
    return v * scale


def ee_pose(obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """An FBAbs observation ``(..., obs_dim)`` (raw units) -> its EE ``(p (..., 3), R (..., 3, 3))``."""
    return obs[..., 0:3], rot6d_to_mat(obs[..., 3:9])


def to_ee_frame(p: torch.Tensor, R: torch.Tensor, actions_abs: torch.Tensor) -> torch.Tensor:
    """Absolute targets ``(B, H, 10)`` -> ``(B, H, 7)`` in the frame of the pose ``p (B, 3)``, ``R (B, 3, 3)``."""
    Rt = R.transpose(-1, -2)[:, None]
    dpos = (Rt @ (actions_abs[..., 0:3] - p[:, None]).unsqueeze(-1)).squeeze(-1)
    drot = mat_to_axis_angle(Rt @ rot6d_to_mat(actions_abs[..., 3:9]))
    return torch.cat((dpos, drot, actions_abs[..., 9:10]), dim=-1)


def from_ee_frame(p: torch.Tensor, R: torch.Tensor, actions_rel: torch.Tensor) -> torch.Tensor:
    """``(B, H, 7)`` in the frame of ``(p, R)`` -> the env's absolute targets ``(B, H, 10)``."""
    goal = p[:, None] + (R[:, None] @ actions_rel[..., 0:3].unsqueeze(-1)).squeeze(-1)
    R_goal = R[:, None] @ axis_angle_to_mat(actions_rel[..., 3:6])
    return torch.cat((goal, mat_to_rot6d(R_goal), actions_rel[..., 6:7]), dim=-1)


def reframe(actions_rel: torch.Tensor, obs_old: torch.Tensor, obs_new: torch.Tensor) -> torch.Tensor:
    """A chunk ``(B, H, 7)`` planned in the EE frame of ``obs_old`` -> the same targets in ``obs_new``'s."""
    p_old, R_old = ee_pose(obs_old)
    p_new, R_new = ee_pose(obs_new)
    return to_ee_frame(p_new, R_new, from_ee_frame(p_old, R_old, actions_rel))
