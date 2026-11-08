"""``residual.action_frame: multiply``: the residual scales the base's motion per dim.

The agent's ``r`` (6, ``[-1, 1]`` per dim from its tanh) is a multiplier per pose dim, ``m = low + (r + 1) / 2 (high -
low)`` (``residual.multiply_min`` at ``r = -1``, ``multiply_max`` at ``r = 1``); the actor starts at ``m = 1``
(:attr:`MultiplyResidual.neutral`, its ``tanh`` value; ``model.actor.output_bias = atanh(neutral)``). The gripper is
always the base's. Everything the residual hands the learner is raw (m, rad, the gripper command): the learner's
``state_preprocessor`` normalizes it.

``residual.multiply_reference`` picks what ``m`` multiplies:

* ``increment`` (the base's planned step, robot-base frame B): the base executes a chunk of absolute targets; within it
  the step is ``d_t = base_t (-) base_{t-1}`` (``[p_t - p_{t-1}, log(R_t R_{t-1}^T)]``), chained onto the current target,
  ``sent_t = ref_t (+) m * d_t`` (``[p + m d_pos, Exp(m d_rot) R]``) with ``ref_t = sent_{t-1}``. At a chunk's first step
  (the base just replanned from the real state) the current target is the EE: ``ref = ee``, ``d = base_0 (-) ee``.
  So ``m = 1`` on every step is the base's own trajectory, ``m = 0`` holds the current target, and a held step never
  makes the next one bigger. The agent observes the step (6, B) with the base's gripper (1) and the current target
  relative to the EE (6, EE frame: zeros at a chunk's first step).
* ``ee`` (the base's target relative to the EE): ``d = [R_ee^T (p_base - p_ee), log(R_ee^T R_base)]`` (EE frame),
  ``sent = ee (+)_E m * d``; the agent observes ``d`` with the base's gripper.

The agent's observation starts with the env observation relative to the EE (:meth:`MultiplyResidual.observation`,
FBRel's conventions; ``residual.observation_drop`` then removes columns). The critics observe that observation part
alone and take the resulting target relative to the EE, ``[ee_frame(sent), gripper]`` (7), rebuilt from the side input
(:meth:`MultiplyResidual.critic_action`): a step in a phase without a residual executed the base (``m = 1``). Plain torch,
gradients finite (``log_so3`` / ``exp_so3``): the critics' action goes through it. :meth:`MultiplyResidual.step_inputs`
is the one definition of a step's inputs, for the env wrapper and the demo loader (``my_project/demo_loader.py``).
"""

from __future__ import annotations

import math
from typing import Dict

import torch

from my_project.models.action_frame import mat_to_rot6d, rot6d_to_mat
from my_project.residual_ee import exp_so3, log_so3

__all__ = ["multiply_from", "MultiplyResidual", "WIDTH", "OBSERVED_WIDTH", "CURRENT_WIDTH", "CRITIC_ACTION_WIDTH",
           "side_width"]

#: the agent's per-step action: a multiplier per pose dim (position 3, rotation 3)
WIDTH = 6
#: the observed base motion: the step (or EE-relative target) 6 and the base's gripper command 1
OBSERVED_WIDTH = 7
#: multiply_reference: increment: the current target relative to the EE (6)
CURRENT_WIDTH = 6
#: the critics' action: the resulting target relative to the EE (6) and the gripper (1)
CRITIC_ACTION_WIDTH = 7
#: the FBAbs ResiP observation (63) this map knows: ee_pos 3, ee_rot6d 6, ee_pos_vel 3, ee_ori_vel 3, gripper_width 1,
#: parts_poses_b 5 x 9, obstacle_xy 2
OBS_WIDTH = 63
PARTS = slice(16, 61)
REFERENCES = ("ee", "increment")


def side_width(reference: str) -> int:
    """The critics' side input width: ``ee`` ``[d 6, gripper 1, routed 1]``; ``increment`` ``[ref 9, d 6, ee 9, gripper 1,
    routed 1]`` (poses ``[p 3, rot6d 6]`` in B)."""
    return 8 if reference == "ee" else 26


def multiply_from(rc, gripper: int = 9) -> "MultiplyResidual":
    """The config's map (``action_frame: multiply``), one definition for the env wrapper and the demo loader."""
    return MultiplyResidual(rc.multiply_min, rc.multiply_max, str(rc.multiply_reference), gripper)


class MultiplyResidual:
    """``low`` / ``high``: the multiplier at ``r = -1`` / ``1``; ``reference``: ``increment`` or ``ee`` (the module
    docstring); ``gripper`` the action's gripper dim; ``eps_pos`` m / ``eps_rot`` rad: a demo's base step this small on a
    dim gives no ratio (the multiplier 1)."""

    def __init__(self, low: float, high: float, reference: str = "increment", gripper: int = 9,
                 eps_pos: float = 5e-4, eps_rot: float = math.radians(0.5)) -> None:
        self.low, self.high = float(low), float(high)
        if not self.low < 1.0 < self.high:
            raise ValueError(f"the multiplier's range [{self.low}, {self.high}] must hold 1 (the base's own action) inside")
        if reference not in REFERENCES:
            raise ValueError(f"multiply_reference must be one of {REFERENCES}, got {reference!r}")
        self.reference = reference
        self.gripper = int(gripper)
        self.eps_pos, self.eps_rot = float(eps_pos), float(eps_rot)
        self.side_width = side_width(reference)

    # ------------------------------------------------------------ the multiplier
    @property
    def half_span(self) -> float:
        """The multiplier per unit of ``r``."""
        return (self.high - self.low) / 2.0

    @property
    def neutral(self) -> float:
        """The ``r`` (in ``[-1, 1]``) whose multiplier is 1."""
        return self.unit(1.0)

    def multiplier(self, r: torch.Tensor) -> torch.Tensor:
        return self.low + (r.clamp(-1.0, 1.0) + 1.0) * self.half_span

    def unit(self, m):
        """:meth:`multiplier`'s inverse (unclamped)."""
        return (m - self.low) / self.half_span - 1.0

    # ------------------------------------------------------------ poses
    @staticmethod
    def ee_frame(targets: torch.Tensor, ee: torch.Tensor) -> torch.Tensor:
        """Poses ``(N, >=9)`` against the EE pose ``ee`` ``(N, 9)`` (raw ``ee_pos 3, ee_rot6d 6``) -> ``(N, 6)``:
        ``R_ee^T (p - p_ee)`` m, ``log(R_ee^T R)`` rad."""
        R_eeT = rot6d_to_mat(ee[..., 3:9]).transpose(-1, -2)
        dp = (R_eeT @ (targets[..., 0:3] - ee[..., 0:3]).unsqueeze(-1)).squeeze(-1)
        return torch.cat((dp, log_so3(R_eeT @ rot6d_to_mat(targets[..., 3:9]))), dim=-1)

    @staticmethod
    def compose(d: torch.Tensor, ee: torch.Tensor) -> torch.Tensor:
        """:meth:`ee_frame`'s inverse: ``d`` ``(N, 6)`` on the EE pose -> the pose ``(N, 9)``."""
        R_ee = rot6d_to_mat(ee[..., 3:9])
        p = ee[..., 0:3] + (R_ee @ d[..., 0:3].unsqueeze(-1)).squeeze(-1)
        return torch.cat((p, mat_to_rot6d(R_ee @ exp_so3(d[..., 3:6]))), dim=-1)

    @staticmethod
    def world_delta(target: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """``target (-) ref`` for poses ``(N, >=9)`` in B: ``[p - p_ref, log(R R_ref^T)]`` ``(N, 6)``."""
        R_ref = rot6d_to_mat(ref[..., 3:9])
        return torch.cat((target[..., 0:3] - ref[..., 0:3], log_so3(rot6d_to_mat(target[..., 3:9]) @ R_ref.transpose(-1, -2))), dim=-1)

    @staticmethod
    def world_compose(d: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """:meth:`world_delta`'s inverse: ``ref (+) d`` -> the pose ``(N, 9)``."""
        R = exp_so3(d[..., 3:6]) @ rot6d_to_mat(ref[..., 3:9])
        return torch.cat((ref[..., 0:3] + d[..., 0:3], mat_to_rot6d(R)), dim=-1)

    def _grip(self, base: torch.Tensor) -> torch.Tensor:
        return base[..., self.gripper : self.gripper + 1]

    # ------------------------------------------------------------ one step
    def motion(self, base, ee, prev_sent=None, prev_base=None, chunk_start=None):
        """``(d, ref)``: what ``m`` multiplies and the pose it applies to. ``increment``: ``base (-) ee`` from the EE at a
        chunk start (``(N,)`` bool), else ``base (-) prev_base`` from ``prev_sent`` (B). ``ee``: the EE-relative target
        (EE frame), ``ref`` the EE."""
        if self.reference == "ee":
            return self.ee_frame(base, ee), ee[..., 0:9]
        cs = chunk_start.reshape(-1, 1)
        d = torch.where(cs, self.world_delta(base, ee), self.world_delta(base, prev_base))
        return d, torch.where(cs, ee[..., 0:9], prev_sent[..., 0:9])

    def sent(self, base, d, ref, r, routed):
        """The sent targets ``(N, 10)``: ``ref`` moved by ``m * d`` (B for ``increment``, the EE frame for ``ee``), the
        base's gripper; an env not ``routed`` (``(N,)`` bool: its phase has no residual) sends ``base`` itself."""
        if r.shape[-1] != WIDTH:
            raise ValueError(f"the multiply residual is {WIDTH} per step, got {r.shape[-1]}")
        step = self.multiplier(r) * d
        pose = self.compose(step, ref) if self.reference == "ee" else self.world_compose(step, ref)
        return torch.where(routed.reshape(-1, 1), torch.cat((pose, self._grip(base)), dim=-1), base)

    def step_inputs(self, obs, base, keep, routed, prev_sent=None, prev_base=None, chunk_start=None) -> Dict[str, torch.Tensor]:
        """One step's inputs, raw, for the env wrapper and the demo loader: ``policy`` (the EE-relative observation part,
        then the base's motion 7, then -- increment -- the current target 6), ``critic`` (the observation part alone),
        ``side`` (:meth:`critic_action`'s), ``d`` and ``ref`` (what :meth:`sent` takes). ``obs`` ``(N, 63)`` raw, ``base``
        ``(N, 10)``, ``keep`` the observation columns kept (or None), ``routed`` ``(N,)`` bool."""
        ee = obs[..., 0:9]
        part = self.observation(obs)
        if keep is not None:
            part = part[..., keep]
        d, ref = self.motion(base, ee, prev_sent, prev_base, chunk_start)
        grip = self._grip(base)
        flag = routed.to(obs.dtype).reshape(-1, 1)
        if self.reference == "ee":
            policy = torch.cat((part, d, grip), dim=-1)
            side = torch.cat((d, grip, flag), dim=-1)
        else:
            current = self.ee_frame(ref, ee)  # the current target relative to the EE: zeros at a chunk start
            policy = torch.cat((part, d, grip, current), dim=-1)
            # an env whose phase has no residual sends the base itself: its side input rebuilds exactly that (the base's
            # pose and no motion), whatever r is
            on = routed.reshape(-1, 1)
            side = torch.cat((torch.where(on, ref, base[..., 0:9]), torch.where(on, d, torch.zeros_like(d)), ee, grip, flag), dim=-1)
        return {"policy": policy, "critic": part, "side": side, "d": d, "ref": ref}

    def critic_action(self, side: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """The critics' action ``(N, 7)`` from the side input and ``r``: the resulting target relative to the EE and the
        base's gripper (``m = 1`` where the phase has no residual)."""
        if self.reference == "ee":
            d, grip, routed = side[..., 0:6], side[..., 6:7], side[..., 7:8] > 0.5
            return torch.cat((torch.where(routed, self.multiplier(r[..., :WIDTH]) * d, d), grip), dim=-1)
        ref, d, ee, grip, routed = side[..., 0:9], side[..., 9:15], side[..., 15:24], side[..., 24:25], side[..., 25:26] > 0.5
        step = torch.where(routed, self.multiplier(r[..., :WIDTH]) * d, d)
        return torch.cat((self.ee_frame(self.world_compose(step, ref), ee), grip), dim=-1)

    def invert(self, d: torch.Tensor, demo: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """A demo frame's ``r`` ``(N, 6)``: per dim the demo's motion over the base's, ``(demo (-) ref) / d`` (the EE frame
        for ``ee``), clamped to ``[low, high]``; 1 where the base's motion on that dim is under ``eps_pos`` / ``eps_rot``."""
        d_demo = self.ee_frame(demo, ref) if self.reference == "ee" else self.world_delta(demo, ref)
        eps = torch.tensor([self.eps_pos] * 3 + [self.eps_rot] * 3, dtype=d.dtype, device=d.device)
        big = d.abs() >= eps
        ratio = torch.where(big, d_demo / torch.where(big, d, torch.ones_like(d)), torch.ones_like(d))
        return self.unit(ratio.clamp(self.low, self.high))

    @staticmethod
    def observation(obs: torch.Tensor) -> torch.Tensor:
        """The FBAbs ResiP observation ``(N, 63)``, raw, relative to the EE (FBRel's ``ee_relative_states``, the same
        width): the EE's pose as it is (robot-base frame; ``residual.observation_drop`` removes it), its linear and angular
        velocity in its own frame, the finger separation, each part ``[R_ee^T (p - p_ee), rot6d(R_ee^T R)]``; the
        obstacle's two columns as they are (the caller drops them)."""
        if obs.shape[-1] != OBS_WIDTH:
            raise ValueError(f"the multiply residual's EE-relative observation reads the {OBS_WIDTH}-wide FBAbs ResiP observation, got {obs.shape[-1]}")
        R_eeT = rot6d_to_mat(obs[..., 3:9]).transpose(-1, -2)
        vel = [(R_eeT @ obs[..., a : a + 3].unsqueeze(-1)).squeeze(-1) for a in (9, 12)]
        parts = obs[..., PARTS].reshape(*obs.shape[:-1], 5, 9)
        Rt = R_eeT.unsqueeze(-3)
        p = (Rt @ (parts[..., 0:3] - obs[..., None, 0:3]).unsqueeze(-1)).squeeze(-1)
        R = mat_to_rot6d(Rt @ rot6d_to_mat(parts[..., 3:9]))
        rel = torch.cat((p, R), dim=-1).reshape(*obs.shape[:-1], 45)
        return torch.cat((obs[..., 0:9], *vel, obs[..., 15:16], rel, obs[..., 61:63]), dim=-1)
