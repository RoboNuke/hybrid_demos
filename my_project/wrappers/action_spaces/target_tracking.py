"""``target_tracking.*``: the env is sent an absolute target per tracked body; the agent's action is each target's change
(:class:`TargetTracking`, the :class:`~my_project.wrappers.chain.ConversionWrapper` around it).

Per target (an arm's pose, a finger's joints): it starts at the body's current value when an episode starts (and, with
``reset_at_chunk``, whenever the base re-plans). Every step:

* the base's change ``d``: the base's target against the previous base target (against the body's current value at a
  start) -- positions ``p_base - p_prev``, rotations ``log(R_base R_prev^T)`` (axis-angle, base frame's axes); this is
  the base action the wrappers above see;
* the sent target: the current target moved by the agent's change ``s``, ``p + s_pos``, ``Exp(s_rot) R``; a residual
  above that leaves ``s = d`` sends the base's own trajectory.

The action this wrapper takes is, per target in order, its position change then (a pose) its rotation change (3), then
the action dims no target covers, as they are. ``target_in_policy`` / ``target_in_critic`` append every current target,
relative to its body (``R^T (p_target - p)``, ``log(R^T R_target)``; a joint target minus the joints), to the agent's /
the critics' observation columns.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch

from my_project.wrappers.chain import Conversion, ConversionWrapper
from my_project.wrappers.pose_math import exp_so3, log_so3, mat_to_rot, relative_pose, rot_to_mat

__all__ = ["Target", "TargetTracking", "TargetTrackingWrapper"]


class Target:
    """One tracked body: its position-like columns in the action and the observation, and optionally its rotation's."""

    def __init__(self, action_pos: Sequence[int], obs_pos: Sequence[int], action_rot: Optional[Sequence[int]] = None,
                 action_rot_param: str = "rot6d", obs_rot: Optional[Sequence[int]] = None, obs_rot_param: str = "rot6d") -> None:
        self.action_pos, self.obs_pos = slice(*map(int, action_pos)), slice(*map(int, obs_pos))
        self.action_rot = None if action_rot is None else slice(*map(int, action_rot))
        self.obs_rot = None if obs_rot is None else slice(*map(int, obs_rot))
        self.action_rot_param, self.obs_rot_param = str(action_rot_param), str(obs_rot_param)
        self.k = self.action_pos.stop - self.action_pos.start
        #: its part of the agent's action and of the appended observation: the position change, then the rotation's
        self.width = self.k + (3 if self.action_rot is not None else 0)


class TargetTracking(Conversion):
    def __init__(self, targets: List[Target], reset_at_chunk: bool = True, target_in_policy: bool = True,
                 target_in_critic: bool = False) -> None:
        self.targets = targets
        self.reset_at_chunk = bool(reset_at_chunk)
        self.in_policy, self.in_critic = bool(target_in_policy), bool(target_in_critic)

    def build(self, obs_width: int, critic_width: int, action_dim: int) -> None:
        covered = []
        for t in self.targets:
            covered += list(range(t.action_pos.start, t.action_pos.stop))
            if t.action_rot is not None:
                covered += list(range(t.action_rot.start, t.action_rot.stop))
            stops = [t.obs_pos.stop] + ([t.obs_rot.stop] if t.obs_rot is not None else [])
            if max(stops) > obs_width:
                raise ValueError(f"target_tracking: a target's observation columns run past the {obs_width}-wide observation")
        if max(covered) >= action_dim:
            raise ValueError(f"target_tracking: a target's action columns run past the {action_dim}-wide action")
        #: the action dims no target covers, carried as they are after the targets' changes
        self.passthrough = [a for a in range(action_dim) if a not in covered]
        self.change_width = sum(t.width for t in self.targets)
        self.inner_action_dim = action_dim
        self.obs_width = obs_width + (self.change_width if self.in_policy else 0)
        self.critic_width = critic_width + (self.change_width if self.in_critic else 0)
        self.action_dim = self.change_width + len(self.passthrough)
        self._index = {}

    def _pass(self, like: torch.Tensor) -> torch.Tensor:
        found = self._index.get(like.device)
        if found is None:
            found = torch.tensor(self.passthrough, dtype=torch.long, device=like.device)
            self._index[like.device] = found
        return found

    def initial_state(self, n: int, device: Any) -> Dict[str, torch.Tensor]:
        state = {}
        for i, t in enumerate(self.targets):
            state[f"sent_p{i}"] = torch.zeros((n, t.k), device=device)
            state[f"base_p{i}"] = torch.zeros((n, t.k), device=device)
            if t.action_rot is not None:
                state[f"sent_R{i}"] = torch.eye(3, device=device).expand(n, 3, 3).clone()
                state[f"base_R{i}"] = torch.eye(3, device=device).expand(n, 3, 3).clone()
        return state

    def observe(self, policy, critic, base, state, *, fresh, new_chunk):
        start = fresh.to(torch.bool).reshape(-1, 1)
        if self.reset_at_chunk:
            start = start | new_chunk.to(torch.bool).reshape(-1, 1)
        changes, currents, ctx = [], [], {"base": base}
        for i, t in enumerate(self.targets):
            cur_p = policy[..., t.obs_pos]
            base_p = base[..., t.action_pos]
            anchor_p = torch.where(start, cur_p, state[f"sent_p{i}"])
            prev_p = torch.where(start, cur_p, state[f"base_p{i}"])
            ctx[f"anchor_p{i}"], ctx[f"base_p{i}"] = anchor_p, base_p
            parts, rel = [base_p - prev_p], [anchor_p - cur_p]
            if t.action_rot is not None:
                cur_R = rot_to_mat(policy[..., t.obs_rot], t.obs_rot_param)
                base_R = rot_to_mat(base[..., t.action_rot], t.action_rot_param)
                s3 = start.unsqueeze(-1)
                anchor_R = torch.where(s3, cur_R, state[f"sent_R{i}"])
                prev_R = torch.where(s3, cur_R, state[f"base_R{i}"])
                ctx[f"anchor_R{i}"], ctx[f"base_R{i}"] = anchor_R, base_R
                parts.append(log_so3(base_R @ prev_R.transpose(-1, -2)))
                rp, rR = relative_pose(anchor_p, anchor_R, cur_p, cur_R)
                rel = [rp, log_so3(rR)]
            changes += parts
            currents += rel
        change = torch.cat(changes + [base.index_select(-1, self._pass(base))], dim=-1)
        current = torch.cat(currents, dim=-1)
        if self.in_policy:
            policy = torch.cat((policy, current), dim=-1)
        if self.in_critic:
            critic = torch.cat((critic, current), dim=-1)
        return policy, critic, change, ctx, state

    def _targets(self, change: torch.Tensor, ctx) -> Dict[str, torch.Tensor]:
        """Each target moved by ``change`` (this wrapper's action form) from the current one."""
        out, col = {}, 0
        for i, t in enumerate(self.targets):
            out[f"p{i}"] = ctx[f"anchor_p{i}"] + change[..., col : col + t.k]
            col += t.k
            if t.action_rot is not None:
                out[f"R{i}"] = exp_so3(change[..., col : col + 3]) @ ctx[f"anchor_R{i}"]
                col += 3
        return out

    def _next_state(self, state, targets, ctx) -> Dict[str, torch.Tensor]:
        state = dict(state)
        for i, t in enumerate(self.targets):
            state[f"sent_p{i}"], state[f"base_p{i}"] = targets[f"p{i}"], ctx[f"base_p{i}"]
            if t.action_rot is not None:
                state[f"sent_R{i}"], state[f"base_R{i}"] = targets[f"R{i}"], ctx[f"base_R{i}"]
        return state

    def act(self, action, ctx, state):
        targets = self._targets(action, ctx)
        inner = action.new_zeros(*action.shape[:-1], self.inner_action_dim)
        for i, t in enumerate(self.targets):
            inner[..., t.action_pos] = targets[f"p{i}"]
            if t.action_rot is not None:
                inner[..., t.action_rot] = mat_to_rot(targets[f"R{i}"], t.action_rot_param)
        inner[..., self._pass(action)] = action[..., self.change_width :]
        return inner, self._next_state(state, targets, ctx)

    def invert(self, executed, ctx, state):
        changes, targets = [], {}
        for i, t in enumerate(self.targets):
            p = executed[..., t.action_pos]
            changes.append(p - ctx[f"anchor_p{i}"])
            targets[f"p{i}"] = p
            if t.action_rot is not None:
                R = rot_to_mat(executed[..., t.action_rot], t.action_rot_param)
                changes.append(log_so3(R @ ctx[f"anchor_R{i}"].transpose(-1, -2)))
                targets[f"R{i}"] = R
        change = torch.cat(changes + [executed.index_select(-1, self._pass(executed))], dim=-1)
        # the chain goes on from the target the env executed
        return change, self._next_state(state, targets, ctx)


class TargetTrackingWrapper(ConversionWrapper):
    def __init__(self, env: Any, cfg: Any) -> None:
        tc = cfg.target_tracking
        targets = [Target(t.action_pos, t.obs_pos, t.action_rot, t.action_rot_param, t.obs_rot, t.obs_rot_param)
                   for t in tc.targets]
        super().__init__(env, TargetTracking(targets, tc.reset_at_chunk, tc.target_in_policy, tc.target_in_critic))
        print(f"[target_tracking] {len(targets)} tracked targets (reset {'at every chunk' if tc.reset_at_chunk else 'per episode'}"
              f"); action: their changes ({self.conversion.change_width}) then dims {self.conversion.passthrough} as they are; "
              f"current targets in the agent's observation {tc.target_in_policy}, the critics' {tc.target_in_critic}", flush=True)
