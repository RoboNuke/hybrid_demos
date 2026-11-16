"""Rotation and pose helpers shared by the residual wrappers (``my_project/wrappers``). Plain torch; every function keeps
its gradients finite (a critic's action can go through them).

A rotation is read and written in one of :data:`ROT_PARAMS`: axis-angle (``rotvec``, rad), roll-pitch-yaw (``euler``,
``R = Rz(yaw) Ry(pitch) Rx(roll)``), ``rot6d`` (the matrix's first two rows, Gram-Schmidt back) or a unit quaternion
(``quat_xyzw`` / ``quat_wxyz``).
"""

from __future__ import annotations

from typing import Tuple

import torch

from my_project.models.action_frame import mat_to_rot6d, rot6d_to_mat

__all__ = ["ROT_PARAMS", "rot_width", "rot_to_mat", "mat_to_rot", "exp_so3", "log_so3", "euler_rpy", "rpy_of",
           "mat_to_rot6d", "rot6d_to_mat", "rot6d_angle_deg", "relative_pose"]

ROT_PARAMS = ("rotvec", "euler", "rot6d", "quat_xyzw", "quat_wxyz")

#: inside the square roots below: keeps every gradient finite at the identity (a norm's is 0 / 0 there)
_EPS = 1e-12


def rot_width(param: str) -> int:
    """Numbers per rotation in ``param``."""
    return {"rotvec": 3, "euler": 3, "rot6d": 6, "quat_xyzw": 4, "quat_wxyz": 4}[param]


def exp_so3(v: torch.Tensor) -> torch.Tensor:
    """``(..., 3)`` axis-angle (rad) -> ``(..., 3, 3)`` by Rodrigues, with gradients finite everywhere."""
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
    """``(..., 3, 3)`` -> axis-angle ``(..., 3)`` for rotations under pi, gradients finite there:
    ``v = vee(R - R^T) / 2 = sin(theta) axis``, ``theta = atan2(|v|, (tr R - 1) / 2)``, ``theta / sin(theta) v``."""
    v = 0.5 * torch.stack((R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]), dim=-1)
    c = 0.5 * (R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0)
    s2 = (v * v).sum(-1)
    s = torch.sqrt(s2 + _EPS)
    theta = torch.atan2(s, c)
    scale = torch.where(s2 < 1e-8, 1 + s2 / 6, theta / s)  # theta / sin(theta) -> 1 at the identity
    return v * scale[..., None]


def euler_rpy(v: torch.Tensor) -> torch.Tensor:
    """``(..., 3)`` roll, pitch, yaw (rad) -> ``R = Rz(yaw) Ry(pitch) Rx(roll)`` ``(..., 3, 3)``."""
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
    """:func:`euler_rpy`'s inverse for pitch inside (-90, 90) deg."""
    roll = torch.atan2(R[..., 2, 1], R[..., 2, 2])
    pitch = torch.asin((-R[..., 2, 0]).clamp(-1.0, 1.0))
    yaw = torch.atan2(R[..., 1, 0], R[..., 0, 0])
    return torch.stack((roll, pitch, yaw), dim=-1)


def _quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    """``(..., 4)`` ``(w, x, y, z)`` (normalized here) -> ``(..., 3, 3)``."""
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    w, x, y, z = q.unbind(-1)
    rows = (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )
    return torch.stack([torch.stack(row, dim=-1) for row in rows], dim=-2)


def _mat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """``(..., 3, 3)`` -> ``(..., 4)`` ``(w, x, y, z)``, ``w >= 0`` (Shepperd's method, the largest pivot)."""
    m00, m11, m22 = R[..., 0, 0], R[..., 1, 1], R[..., 2, 2]
    trace = m00 + m11 + m22
    cands = torch.stack((
        torch.stack((1 + trace, R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]), -1),
        torch.stack((R[..., 2, 1] - R[..., 1, 2], 1 + m00 - m11 - m22, R[..., 0, 1] + R[..., 1, 0], R[..., 0, 2] + R[..., 2, 0]), -1),
        torch.stack((R[..., 0, 2] - R[..., 2, 0], R[..., 0, 1] + R[..., 1, 0], 1 - m00 + m11 - m22, R[..., 1, 2] + R[..., 2, 1]), -1),
        torch.stack((R[..., 1, 0] - R[..., 0, 1], R[..., 0, 2] + R[..., 2, 0], R[..., 1, 2] + R[..., 2, 1], 1 - m00 - m11 + m22), -1),
    ), dim=-2)
    pivot = torch.stack((1 + trace, 1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22), -1).argmax(-1)
    q = torch.gather(cands, -2, pivot[..., None, None].expand(*pivot.shape, 1, 4)).squeeze(-2)
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return torch.where(q[..., :1] < 0, -q, q)


def rot_to_mat(v: torch.Tensor, param: str) -> torch.Tensor:
    """A rotation written in ``param`` -> ``(..., 3, 3)``."""
    if param == "rotvec":
        return exp_so3(v)
    if param == "euler":
        return euler_rpy(v)
    if param == "rot6d":
        return rot6d_to_mat(v)
    if param == "quat_xyzw":
        return _quat_to_mat(torch.cat((v[..., 3:4], v[..., 0:3]), dim=-1))
    if param == "quat_wxyz":
        return _quat_to_mat(v)
    raise ValueError(f"unknown rotation parametrisation {param!r}; one of {ROT_PARAMS}")


def mat_to_rot(R: torch.Tensor, param: str) -> torch.Tensor:
    """``(..., 3, 3)`` -> the rotation written in ``param``."""
    if param == "rotvec":
        return log_so3(R)
    if param == "euler":
        return rpy_of(R)
    if param == "rot6d":
        return mat_to_rot6d(R)
    if param in ("quat_xyzw", "quat_wxyz"):
        q = _mat_to_quat(R)
        return torch.cat((q[..., 1:4], q[..., 0:1]), dim=-1) if param == "quat_xyzw" else q
    raise ValueError(f"unknown rotation parametrisation {param!r}; one of {ROT_PARAMS}")


def rot6d_angle_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Angle (deg) between the rotations two ``(N, 6)`` rot6d vectors decode to."""
    R = rot6d_to_mat(a).transpose(-1, -2) @ rot6d_to_mat(b)
    cos = ((R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.arccos(cos))


def relative_pose(p: torch.Tensor, R: torch.Tensor, p_frame: torch.Tensor, R_frame: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """A pose ``(p, R)`` against a frame ``(p_frame, R_frame)``, in the frame's axes: ``(R_f^T (p - p_f), R_f^T R)``."""
    Rt = R_frame.transpose(-1, -2)
    return (Rt @ (p - p_frame).unsqueeze(-1)).squeeze(-1), Rt @ R
