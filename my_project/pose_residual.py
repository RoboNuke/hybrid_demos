"""``residual.action_space: pose``: the residual on a furniture base's absolute pose targets, one map for every frame.

The base gives absolute targets ``[pos 3, rot6d 6, gripper 1]`` in the robot-base frame B; the env observation is the
FBAbs ResiP one (63). Everything here is set by independent config axes:

* **frame** (``action_frame`` / ``observation_frame``): the frame F poses are expressed in -- ``absolute`` B itself,
  ``chunk`` the EE pose at the base's chunk start, ``relative`` the current EE pose;
* **rotation** (``action_rotation`` / ``observation_rotation``): the frame's axes A -- ``absolute`` B's axes,
  ``relative`` F's own (the same thing for the absolute frame);
* **rotation_param**: how a rotation is written as numbers -- ``rotvec`` (axis-angle, 3), ``euler`` (roll, pitch, yaw,
  ``Rz Ry Rx``, 3) or ``rot6d`` (the first two rows, 6, Gram-Schmidt back);
* **combine**: ``add`` (``step = d + scale * r``) or ``multiply`` (``step = r * d``), per dim of the parametrised step;
* **action_target**: ``none`` -- ``d`` is the base's target expressed in the frame, ``express(base; F, A)``, and the sent
  target is the step taken back out, ``unexpress(step; F, A)``; ``target`` -- a chained target that starts at the EE at
  each chunk start: ``d`` is the base's planned increment in A's axes, ``increment(base_t, base_{t-1}; A)`` (at a chunk
  start ``increment(base_0, ee; A)``), and the sent target is the previous one advanced by the step,
  ``advance(prev_sent, step; A)``. So ``r`` at its neutral value (0 for add, 1 for multiply) sends the base's own target.

The gripper is always the base's. ``r`` is ``3 + 3`` wide (``3 + 6`` with ``rot6d``) and comes from the learner raw
(its ``action_range`` sets what r can be). A frame transform is ``express(X; F, A) = (A^T (p_X - p_F), A^T R_X)``; an
increment ``(A^T (p_1 - p_0), A^T R_1 R_0^T A)``.

The agent observes the env observation re-expressed in the observation frame (``residual.observation_drop`` then
removes columns), the base's ``d`` and gripper, and -- ``target`` -- the current target expressed in the observation
frame (parametrised). The critics observe the observation part (and the current target) and take the resulting target
relative to the EE and the gripper (7), rebuilt from a fixed side input (:data:`SIDE_WIDTH`). Plain torch, gradients
finite (``log_so3`` / ``exp_so3``): the critics' action goes through it. :meth:`PoseResidual.step_inputs` is the one
definition of a step's inputs, for the env wrapper and the demo loader.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch

from my_project.models.action_frame import mat_to_rot6d, rot6d_to_mat

__all__ = ["PoseResidual", "pose_from", "exp_so3", "log_so3", "euler_rpy", "rpy_of", "param_width", "ee_frame",
           "SIDE_WIDTH", "CRITIC_ACTION_WIDTH", "OBS_WIDTHS", "FRAMES", "ROTATIONS", "PARAMS", "COMBINES", "TARGETS"]

FRAMES = ("absolute", "chunk", "relative")
ROTATIONS = ("absolute", "relative")
PARAMS = ("rotvec", "euler", "rot6d")
COMBINES = ("add", "multiply")
TARGETS = ("none", "target")

#: the FBAbs observation (61): ee_pos 3, ee_rot6d 6, ee_pos_vel 3, ee_ori_vel 3, gripper_width 1, parts_poses_b 5 x 9;
#: the ResiP one (63) adds obstacle_xy 2
OBS_WIDTHS = (61, 63)
PARTS = slice(16, 61)
#: the critics' side input: [anchor pose 9, frame axes rot6d 6, d padded to 9, ee pose 9, gripper 1, routed 1]
SIDE_WIDTH = 35
#: the critics' action: the resulting target relative to the EE (6) and the gripper (1)
CRITIC_ACTION_WIDTH = 7

#: inside the square roots below: keeps every gradient finite at the identity (a norm's is 0 / 0 there)
_EPS = 1e-12


# ------------------------------------------------------------------ rotations
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
    ``v = vee(R - R^T) / 2 = sin(theta) axis``, ``theta = atan2(|v|, (tr R - 1) / 2)``, ``theta / sin(theta) v``.
    Near pi the axis is ill-conditioned: an absolute orientation (``action_frame: absolute`` without a target) is best
    written ``rot6d``."""
    v = 0.5 * torch.stack((R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]), dim=-1)
    c = 0.5 * (R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0)
    s2 = (v * v).sum(-1)
    s = torch.sqrt(s2 + _EPS)
    theta = torch.atan2(s, c)
    scale = torch.where(s2 < 1e-8, 1 + s2 / 6, theta / s)  # theta / sin(theta) -> 1 at the identity
    return v * scale[..., None]


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


def param_width(param: str) -> int:
    """Numbers per rotation in ``param``."""
    return 6 if param == "rot6d" else 3


def to_param(R: torch.Tensor, param: str) -> torch.Tensor:
    if param == "rotvec":
        return log_so3(R)
    if param == "euler":
        return rpy_of(R)
    return mat_to_rot6d(R)


def from_param(v: torch.Tensor, param: str) -> torch.Tensor:
    if param == "rotvec":
        return exp_so3(v)
    if param == "euler":
        return euler_rpy(v)
    return rot6d_to_mat(v)


def identity_param(param: str, like: torch.Tensor) -> torch.Tensor:
    """The identity rotation in ``param``, shaped as ``like``'s rows: zeros, or rot6d's ``[1, 0, 0, 0, 1, 0]``."""
    rows = like.shape[:-1]
    if param != "rot6d":
        return like.new_zeros(*rows, 3)
    return mat_to_rot6d(torch.eye(3, dtype=like.dtype, device=like.device)).expand(*rows, 6)


# ------------------------------------------------------------------ poses
def _split(pose: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return pose[..., 0:3], rot6d_to_mat(pose[..., 3:9])


def _pose(p: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    return torch.cat((p, mat_to_rot6d(R)), dim=-1)


def _apply(A: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return (A @ v.unsqueeze(-1)).squeeze(-1)


def express(X: torch.Tensor, F: torch.Tensor, A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pose ``X`` (``(..., >=9)``) in the frame at ``F``'s position with axes ``A``: ``(A^T (p_X - p_F), A^T R_X)``."""
    p_X, R_X = _split(X)
    At = A.transpose(-1, -2)
    return _apply(At, p_X - F[..., 0:3]), At @ R_X


def unexpress(dp: torch.Tensor, dR: torch.Tensor, F: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """:func:`express`'s inverse -> the pose ``(..., 9)``: ``(p_F + A dp, A dR)``."""
    return _pose(F[..., 0:3] + _apply(A, dp), A @ dR)


def increment(X1: torch.Tensor, X0: torch.Tensor, A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """The motion from pose ``X0`` to ``X1`` in axes ``A``: ``(A^T (p_1 - p_0), A^T R_1 R_0^T A)``."""
    p1, R1 = _split(X1)
    p0, R0 = _split(X0)
    At = A.transpose(-1, -2)
    return _apply(At, p1 - p0), At @ R1 @ R0.transpose(-1, -2) @ A


def advance(P: torch.Tensor, dp: torch.Tensor, dR: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Pose ``P`` moved by an increment in axes ``A`` -> ``(p_P + A dp, A dR A^T R_P)``."""
    p, R = _split(P)
    return _pose(p + _apply(A, dp), A @ dR @ A.transpose(-1, -2) @ R)


def ee_frame(targets: torch.Tensor, ee: torch.Tensor) -> torch.Tensor:
    """Poses ``(N, >=9)`` against the EE pose ``ee`` ``(N, 9)`` -> ``(N, 6)``: ``R_ee^T (p - p_ee)`` m, ``log(R_ee^T R)`` rad
    (the critics' action form)."""
    dp, dR = express(targets, ee, rot6d_to_mat(ee[..., 3:9]))
    return torch.cat((dp, log_so3(dR)), dim=-1)


def _identity_pose(like: torch.Tensor) -> torch.Tensor:
    eye = mat_to_rot6d(torch.eye(3, dtype=like.dtype, device=like.device)).expand(*like.shape[:-1], 6)
    return torch.cat((like.new_zeros(*like.shape[:-1], 3), eye), dim=-1)


def pose_from(rc, r_range=None, gripper: int = 9) -> "PoseResidual":
    """The config's map (``residual.action_space: pose``), one definition for the env wrapper and the demo loader.
    ``r_range`` the learner's ``action_range`` (the multiplier's bounds a demo's inverted r is clamped to)."""
    return PoseResidual(
        action_frame=str(rc.action_frame), action_rotation=str(rc.action_rotation),
        observation_frame=str(rc.observation_frame), observation_rotation=str(rc.observation_rotation),
        rotation_param=str(rc.rotation_param), combine=str(rc.combine), action_target=str(rc.action_target),
        step_scale_pos=float(rc.step_scale_pos), step_scale_rot=math.radians(float(rc.step_scale_rot_deg)),
        critic_observes_target=bool(rc.critic_observes_target), r_range=r_range, gripper=gripper,
    )


class PoseResidual:
    """See the module docstring. ``step_scale_pos`` m and ``step_scale_rot`` (rad; with ``rot6d`` the same number per
    rot6d component) per unit ``r`` for ``add``; ``r_range`` ``(lo, hi)`` the bounds a demo's inverted multiplier is
    clamped to (``multiply``); ``gripper`` the action's gripper dim; ``eps_pos`` m / ``eps_rot`` rad: a demo's base
    motion this small on a dim gives no ratio (the multiplier 1)."""

    def __init__(self, *, action_frame: str = "absolute", action_rotation: str = "absolute",
                 observation_frame: str = "absolute", observation_rotation: str = "absolute",
                 rotation_param: str = "rotvec", combine: str = "add", action_target: str = "none",
                 step_scale_pos: float = 0.01, step_scale_rot: float = math.radians(5.0),
                 critic_observes_target: bool = True, r_range: Optional[Tuple[float, float]] = None, gripper: int = 9,
                 eps_pos: float = 5e-4, eps_rot: float = math.radians(0.5)) -> None:
        for name, value, allowed in (("action_frame", action_frame, FRAMES), ("action_rotation", action_rotation, ROTATIONS),
                                     ("observation_frame", observation_frame, FRAMES),
                                     ("observation_rotation", observation_rotation, ROTATIONS),
                                     ("rotation_param", rotation_param, PARAMS), ("combine", combine, COMBINES),
                                     ("action_target", action_target, TARGETS)):
            if value not in allowed:
                raise ValueError(f"residual.{name} must be one of {allowed}, got {value!r}")
        self.action_frame, self.action_rotation = action_frame, action_rotation
        self.observation_frame, self.observation_rotation = observation_frame, observation_rotation
        self.param, self.combine = rotation_param, combine
        self.target = action_target == "target"
        if combine == "multiply":
            displacement = self.target or (action_frame != "absolute" and action_rotation == "relative")
            if not displacement:
                raise ValueError("residual.combine: multiply scales a motion; without action_target: target the base's "
                                 "target expressed in this frame is a pose, not a motion: use action_target: target, or "
                                 "action_frame chunk / relative with action_rotation: relative")
            if rotation_param == "rot6d":
                raise ValueError("residual.combine: multiply scales each number of the motion; rot6d's identity is not "
                                 "zero, so its numbers are no motion: use rotation_param rotvec or euler")
        self.scale_pos, self.scale_rot = float(step_scale_pos), float(step_scale_rot)
        if combine == "add" and (self.scale_pos <= 0 or self.scale_rot <= 0):
            raise ValueError("residual.step_scale_pos and step_scale_rot_deg must be > 0 for combine: add")
        self.r_range = None if r_range is None else (float(r_range[0]), float(r_range[1]))
        self.gripper = int(gripper)
        self.eps_pos, self.eps_rot = float(eps_pos), float(eps_rot)
        #: the agent's per-step action: position 3, rotation in the parametrisation
        self.width = 3 + param_width(rotation_param)
        #: the observed base motion: d and the base's gripper command
        self.observed_width = self.width + 1
        #: target: the current target expressed in the observation frame, parametrised (the agent's)
        self.current_width = self.width if self.target else 0
        #: the critics' copy of it: ``critic_observes_target`` false leaves it out (a critic trained before they saw it)
        self.critic_observes_target = bool(critic_observes_target)
        self.critic_current_width = self.current_width if self.critic_observes_target else 0
        self.side_width = SIDE_WIDTH
        #: device constants by (device, dtype), made once: a tensor built from a Python list is a host-to-device copy --
        #: a host synchronization every env step, and illegal inside a captured CUDA graph (the critics' update)
        self._constants: Dict[tuple, Dict[str, torch.Tensor]] = {}

    # ------------------------------------------------------------ frames
    def _frame(self, frame: str, rotation: str, ee: torch.Tensor, ee_chunk: Optional[torch.Tensor]):
        """``(F pose (N, 9), A (N, 3, 3))`` of a frame: its origin pose and its axes."""
        if frame == "absolute":
            F = _identity_pose(ee[..., 0:9])
        elif frame == "chunk":
            if ee_chunk is None:
                raise ValueError("the chunk frame needs the EE pose at the chunk's start")
            F = ee_chunk[..., 0:9]
        else:
            F = ee[..., 0:9]
        A = rot6d_to_mat(F[..., 3:9]) if rotation == "relative" else torch.eye(3, dtype=ee.dtype, device=ee.device).expand(*ee.shape[:-1], 3, 3)
        return F, A

    @property
    def scale(self):
        return [self.scale_pos] * 3 + [self.scale_rot] * param_width(self.param)

    def _constant(self, like: torch.Tensor) -> Dict[str, torch.Tensor]:
        key = (like.device, like.dtype)
        found = self._constants.get(key)
        if found is None:
            found = {
                "scale": torch.tensor(self.scale, dtype=like.dtype, device=like.device),
                "identity": torch.cat((like.new_zeros(3), identity_param(self.param, like.new_zeros(1, 1))[0]), dim=-1),
                "eps": torch.tensor([self.eps_pos] * 3 + [self.eps_rot] * (self.width - 3), dtype=like.dtype, device=like.device),
            }
            self._constants[key] = found
        return found

    def _scale(self, like: torch.Tensor) -> torch.Tensor:
        return self._constant(like)["scale"]

    def _identity_motion(self, like: torch.Tensor) -> torch.Tensor:
        return self._constant(like)["identity"].expand(*like.shape[:-1], self.width)

    # ------------------------------------------------------------ one step
    def motion(self, base, ee, ee_chunk=None, prev_sent=None, prev_base=None, chunk_start=None):
        """``(d (N, width), anchor (N, 9), A (N, 3, 3))``: what r acts on, the pose it is taken from and the axes.
        ``none``: ``d = express(base; F, A)``, the anchor F. ``target``: ``d = increment(base, base_prev; A)`` from the
        last sent target, both the EE at a chunk start (``chunk_start`` ``(N,)`` bool)."""
        F, A = self._frame(self.action_frame, self.action_rotation, ee, ee_chunk)
        if not self.target:
            dp, dR = express(base, F, A)
            return torch.cat((dp, to_param(dR, self.param)), dim=-1), F, A
        cs = chunk_start.reshape(-1, 1)
        anchor = torch.where(cs, ee[..., 0:9], prev_sent[..., 0:9])
        prev = torch.where(cs, ee[..., 0:9], prev_base[..., 0:9])
        dp, dR = increment(base, prev, A)
        return torch.cat((dp, to_param(dR, self.param)), dim=-1), anchor, A

    def step(self, d: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """The parametrised step: ``d + scale * r`` (add) or ``r * d`` (multiply), per dim."""
        if r.shape[-1] != self.width:
            raise ValueError(f"the pose residual is {self.width} per step, got {r.shape[-1]}")
        return d + self._scale(d) * r if self.combine == "add" else r * d

    def pose_of(self, step: torch.Tensor, anchor: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """A parametrised step taken back to a pose ``(N, 9)`` from its anchor."""
        dR = from_param(step[..., 3:], self.param)
        if self.target:
            return advance(anchor, step[..., 0:3], dR, A)
        return unexpress(step[..., 0:3], dR, anchor, A)

    def sent(self, base, d, anchor, A, r, routed):
        """The sent targets ``(N, 10)``: the step's pose and the base's gripper; an env not ``routed`` (``(N,)`` bool: its
        phase has no residual) sends ``base`` itself."""
        pose = self.pose_of(self.step(d, r), anchor, A)
        out = torch.cat((pose, base[..., self.gripper : self.gripper + 1]), dim=-1)
        return torch.where(routed.reshape(-1, 1), out, base)

    def invert(self, executed, d, anchor, A) -> torch.Tensor:
        """A demo frame's ``r`` ``(N, width)``: the executed target's own motion from the anchor ``d_X``, then
        ``(d_X - d) / scale`` (add; not clamped, as the per-dim space stores it) or ``d_X / d`` per dim (multiply; 1
        where the base's motion on a dim is under ``eps_pos`` / ``eps_rot``, clamped to ``r_range``)."""
        if self.target:
            dp, dR = increment(executed, anchor, A)
        else:
            dp, dR = express(executed, anchor, A)
        d_x = torch.cat((dp, to_param(dR, self.param)), dim=-1)
        if self.combine == "add":
            return (d_x - d) / self._scale(d)
        big = d.abs() >= self._constant(d)["eps"]
        ratio = torch.where(big, d_x / torch.where(big, d, torch.ones_like(d)), torch.ones_like(d))
        return ratio if self.r_range is None else ratio.clamp(*self.r_range)

    def motion_scale(self, d: torch.Tensor) -> torch.Tensor:
        """The base's motion in r units per dim (``exploration_std_mode: adaptive``): ``|d - identity| / scale`` for
        add (the r that would double it), 1 for multiply (r = 1 is the base's own motion)."""
        if self.combine == "multiply":
            return torch.ones_like(d)
        return (d - self._identity_motion(d)).abs() / self._scale(d)

    def step_inputs(self, obs, base, keep, routed, ee_chunk=None, prev_sent=None, prev_base=None,
                    chunk_start=None) -> Dict[str, torch.Tensor]:
        """One step's inputs, raw, for the env wrapper and the demo loader: ``policy`` (the re-expressed observation
        part, the base's ``d`` and gripper, then -- target -- the current target), ``critic`` (the observation part and
        -- target, with ``critic_observes_target`` -- the current target), ``side`` (:meth:`critic_action`'s), ``d``, ``anchor``, ``A`` (what
        :meth:`sent` takes) and ``motion_scale``. ``obs`` ``(N, 61 | 63)`` raw, ``base`` ``(N, 10)``, ``keep`` the observation
        columns kept (or None), ``routed`` ``(N,)`` bool."""
        ee = obs[..., 0:9]
        part = self.observation(obs, ee_chunk)
        if keep is not None:
            part = part[..., keep]
        d, anchor, A = self.motion(base, ee, ee_chunk, prev_sent, prev_base, chunk_start)
        grip = base[..., self.gripper : self.gripper + 1]
        policy, critic = torch.cat((part, d, grip), dim=-1), part
        if self.target:
            F, A_obs = self._frame(self.observation_frame, self.observation_rotation, ee, ee_chunk)
            cp, cR = express(anchor, F, A_obs)
            current = torch.cat((cp, to_param(cR, self.param)), dim=-1)
            policy = torch.cat((policy, current), dim=-1)
            if self.critic_observes_target:
                critic = torch.cat((part, current), dim=-1)
        # the critics' side input; an env whose phase has no residual sends the base itself, which its side input
        # rebuilds whatever r is: without a target the step is d itself; with one the anchor is the base and d no motion
        on = routed.reshape(-1, 1)
        side_anchor, side_A, side_d = anchor, A, d
        if self.target:
            eye = torch.eye(3, dtype=A.dtype, device=A.device).expand_as(A)
            side_anchor = torch.where(on, anchor, base[..., 0:9])
            side_A = torch.where(on[..., None], A, eye)
            side_d = torch.where(on, d, self._identity_motion(d))
        pad = side_d.new_zeros(*side_d.shape[:-1], 9 - self.width)
        flag = routed.to(obs.dtype).reshape(-1, 1)
        side = torch.cat((side_anchor, mat_to_rot6d(side_A), side_d, pad, ee, grip, flag), dim=-1)
        return {"policy": policy, "critic": critic, "side": side, "d": d, "anchor": anchor, "A": A,
                "motion_scale": self.motion_scale(d)}

    def critic_action(self, side: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """The critics' action ``(N, 7)`` from the side input and ``r``: the resulting target relative to the EE and the
        base's gripper (the base's own target where the phase has no residual)."""
        anchor, A = side[..., 0:9], rot6d_to_mat(side[..., 9:15])
        d, ee = side[..., 15 : 15 + self.width], side[..., 24:33]
        grip, routed = side[..., 33:34], side[..., 34:35] > 0.5
        step = torch.where(routed, self.step(d, r[..., : self.width]), d)
        return torch.cat((ee_frame(self.pose_of(step, anchor, A), ee), grip), dim=-1)

    # ------------------------------------------------------------ the observation
    def observation(self, obs: torch.Tensor, ee_chunk: Optional[torch.Tensor] = None) -> torch.Tensor:
        """The FBAbs observation ``(N, 61)`` or ResiP one ``(N, 63)``, raw, re-expressed in the observation frame (width
        unchanged): the EE's pose and every part's ``[A^T (p - p_F), rot6d(A^T R)]``, the EE's linear and angular velocity
        in A's axes, the finger separation and (ResiP) the obstacle's two columns as they are."""
        if obs.shape[-1] not in OBS_WIDTHS:
            raise ValueError(f"the pose residual re-expresses the FBAbs observation ({OBS_WIDTHS} wide), got {obs.shape[-1]}")
        F, A = self._frame(self.observation_frame, self.observation_rotation, obs[..., 0:9], ee_chunk)
        At = A.transpose(-1, -2)
        ee_p, ee_R = express(obs[..., 0:9], F, A)
        vel = [_apply(At, obs[..., a : a + 3]) for a in (9, 12)]
        parts = obs[..., PARTS].reshape(*obs.shape[:-1], 5, 9)
        Fp, Ap = F.unsqueeze(-2).expand(*parts.shape[:-1], 9), A.unsqueeze(-3).expand(*parts.shape[:-1], 3, 3)
        pp, pR = express(parts, Fp, Ap)
        rel = torch.cat((pp, mat_to_rot6d(pR)), dim=-1).reshape(*obs.shape[:-1], 45)
        return torch.cat((ee_p, mat_to_rot6d(ee_R), *vel, obs[..., 15:16], rel, obs[..., 61:]), dim=-1)
