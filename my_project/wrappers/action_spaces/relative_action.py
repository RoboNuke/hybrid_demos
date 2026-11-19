"""``relative_action.*``: the action as a change from the robot's current state (:class:`RelativeAction`, the
:class:`~my_project.wrappers.chain.ConversionWrapper` around it), for an env whose action is absolute (DexMimicGen's
hand targets and finger servo angles). The same definition as ``data_analysis/demo_residual_analysis.ipynb``'s
``relative_to_ee`` (``data_analysis/residual_plots/models.yaml``), so a noise scale fitted there means the same here:

* a pose (per hand): the target's position minus the hand's observed position (world axes), and its rotation
  ``R_target R_hand^T`` (world axes) in the action's own parametrisation, ``R_hand`` the observed rotation times
  ``obs_rot_offset`` (the observed frame -> the target's frame);
* a finger column (a position servo's target angle): the command minus the previous step's sent command (0 at an
  episode's first step, robosuite's initial gripper goal).

Every other action dim passes as it is (an OSC delta is relative already). The action keeps its width and dim order, so
the base's own scale per dim still applies to it (``additive_residual.scale_fraction``). A residual above that leaves
the base's relative action unchanged sends the base's absolute action exactly. Observation columns are read as they
reach this wrapper: it sits right above the base policy, so they are the env's own.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch

from my_project.wrappers.chain import Conversion, ConversionWrapper
from my_project.wrappers.pose_math import mat_to_rot, rot_to_mat

__all__ = ["RelativePose", "RelativeAction", "RelativeActionWrapper"]


class RelativePose:
    """One hand: its target's position (``action_pos``) and rotation (``action_rot``, optional) columns, and its observed
    position and rotation in the observation."""

    def __init__(self, action_pos: Sequence[int], obs_pos: Sequence[int], action_rot: Optional[Sequence[int]] = None,
                 action_rot_param: str = "rotvec", obs_rot: Optional[Sequence[int]] = None,
                 obs_rot_param: str = "quat_xyzw", obs_rot_offset: Optional[Sequence[Sequence[float]]] = None) -> None:
        self.action_pos, self.obs_pos = slice(*map(int, action_pos)), slice(*map(int, obs_pos))
        self.action_rot = None if action_rot is None else slice(*map(int, action_rot))
        self.obs_rot = None if obs_rot is None else slice(*map(int, obs_rot))
        self.action_rot_param, self.obs_rot_param = str(action_rot_param), str(obs_rot_param)
        self.offset = None if obs_rot_offset is None else torch.tensor([[float(v) for v in row] for row in obs_rot_offset])

    def hand_R(self, policy: torch.Tensor) -> torch.Tensor:
        R = rot_to_mat(policy[..., self.obs_rot], self.obs_rot_param)
        if self.offset is not None:
            R = R @ self.offset.to(R.device, R.dtype)
        return R


class RelativeAction(Conversion):
    def __init__(self, poses: List[RelativePose], fingers: Sequence[int] = ()) -> None:
        self.poses = poses
        self.fingers = [int(f) for f in fingers]
        self._index: Dict[Any, torch.Tensor] = {}
        #: ``(lo, hi)`` on the absolute action the env executes (an additive residual's ResFiT clamp), or None
        self.executed_clip = None

    def build(self, obs_width: int, critic_width: int, action_dim: int) -> None:
        covered = list(self.fingers)
        for p in self.poses:
            covered += list(range(p.action_pos.start, p.action_pos.stop))
            if p.action_rot is not None:
                covered += list(range(p.action_rot.start, p.action_rot.stop))
            stops = [p.obs_pos.stop] + ([p.obs_rot.stop] if p.obs_rot is not None else [])
            if max(stops) > obs_width:
                raise ValueError(f"relative_action: a pose's observation columns run past the {obs_width}-wide observation")
        if len(set(covered)) != len(covered):
            raise ValueError("relative_action: the poses and fingers overlap in the action")
        if covered and max(covered) >= action_dim:
            raise ValueError(f"relative_action: an action column runs past the {action_dim}-wide action")
        self.obs_width, self.critic_width, self.action_dim = obs_width, critic_width, action_dim

    def _fingers(self, like: torch.Tensor) -> torch.Tensor:
        found = self._index.get(like.device)
        if found is None:
            found = torch.tensor(self.fingers, dtype=torch.long, device=like.device)
            self._index[like.device] = found
        return found

    def initial_state(self, n: int, device: Any) -> Dict[str, torch.Tensor]:
        #: each finger's previous sent command (0 before an episode's first step)
        return {"held": torch.zeros((n, len(self.fingers)), device=device)}

    def observe(self, policy, critic, base, state, *, fresh, new_chunk):
        held = torch.where(fresh.to(torch.bool).reshape(-1, 1), torch.zeros_like(state["held"]), state["held"])
        ctx: Dict[str, Any] = {"held": held, "policy": policy}
        return policy, critic, self._relative(base, ctx), ctx, state

    def _relative(self, absolute: torch.Tensor, ctx) -> torch.Tensor:
        out = absolute.clone()
        policy = ctx["policy"]
        for p in self.poses:
            out[..., p.action_pos] = absolute[..., p.action_pos] - policy[..., p.obs_pos]
            if p.action_rot is not None:
                R = rot_to_mat(absolute[..., p.action_rot], p.action_rot_param)
                out[..., p.action_rot] = mat_to_rot(R @ p.hand_R(policy).transpose(-1, -2), p.action_rot_param)
        if self.fingers:
            index = self._fingers(absolute)
            out[..., index] = absolute.index_select(-1, index) - ctx["held"]
        return out

    def _absolute(self, relative: torch.Tensor, ctx) -> torch.Tensor:
        out = relative.clone()
        policy = ctx["policy"]
        for p in self.poses:
            out[..., p.action_pos] = relative[..., p.action_pos] + policy[..., p.obs_pos]
            if p.action_rot is not None:
                R = rot_to_mat(relative[..., p.action_rot], p.action_rot_param)
                out[..., p.action_rot] = mat_to_rot(R @ p.hand_R(policy), p.action_rot_param)
        if self.fingers:
            index = self._fingers(relative)
            out[..., index] = relative.index_select(-1, index) + ctx["held"]
        return out

    def _sent(self, absolute: torch.Tensor, state) -> Dict[str, torch.Tensor]:
        state = dict(state)
        if self.fingers:
            state["held"] = absolute.index_select(-1, self._fingers(absolute))
        return state

    def act(self, action, ctx, state):
        absolute = self._absolute(action, ctx)
        if self.executed_clip is not None:
            lo, hi = self.executed_clip
            absolute = torch.maximum(torch.minimum(absolute, hi.to(absolute.device)), lo.to(absolute.device))
        # the fingers' reference for the next step is the command the env executes (the notebook's previous executed one)
        return absolute, self._sent(absolute, state)

    def invert(self, executed, ctx, state):
        return self._relative(executed, ctx), self._sent(executed, state)


class RelativeActionWrapper(ConversionWrapper):
    def __init__(self, env: Any, cfg: Any) -> None:
        rc = cfg.relative_action
        poses = [RelativePose(p.action_pos, p.obs_pos, p.action_rot, p.action_rot_param, p.obs_rot, p.obs_rot_param,
                              p.obs_rot_offset) for p in rc.poses]
        super().__init__(env, RelativeAction(poses, rc.fingers))
        print(f"[relative_action] {len(poses)} poses relative to the hand's observed pose, fingers {list(rc.fingers)} "
              "relative to the previous sent command; every other dim as it is", flush=True)
