"""``drop_columns.*``: observation columns and action dims removed (:class:`DropColumns`, the
:class:`~my_project.wrappers.chain.ConversionWrapper` around it). A dropped action dim is the base's: the wrapper below
gets the base's own value there, and the dim leaves the base action above."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import torch

from my_project.wrappers.chain import Conversion, ConversionWrapper

__all__ = ["DropColumns", "DropColumnsWrapper"]


class DropColumns(Conversion):
    """``observation``: ``[start, stop)`` column pairs, out of both observations (a critic observation of its own width
    -- the env's own critic group -- keeps its columns); ``action``: dims."""

    def __init__(self, observation: Sequence[Sequence[int]] = (), action: Sequence[int] = ()) -> None:
        self.drop_obs = [(int(a), int(b)) for a, b in observation]
        self.drop_act = sorted(int(a) for a in action)

    def build(self, obs_width: int, critic_width: int, action_dim: int) -> None:
        gone = set()
        for start, stop in self.drop_obs:
            if stop > obs_width:
                raise ValueError(f"drop_columns.observation [{start}, {stop}) runs past the {obs_width}-wide observation")
            gone.update(range(start, stop))
        if any(a >= action_dim for a in self.drop_act):
            raise ValueError(f"drop_columns.action {self.drop_act} runs past the {action_dim}-wide action")
        self.keep_obs = [c for c in range(obs_width) if c not in gone]
        self.keep_act = [a for a in range(action_dim) if a not in self.drop_act]
        if not self.keep_act:
            raise ValueError("drop_columns.action drops every action dim")
        self.drop_critic = critic_width == obs_width
        self.obs_width = len(self.keep_obs)
        self.critic_width = len(self.keep_obs) if self.drop_critic else critic_width
        self.action_dim = len(self.keep_act)
        self._index: Dict[Any, Dict[str, torch.Tensor]] = {}

    def _idx(self, like: torch.Tensor) -> Dict[str, torch.Tensor]:
        # built once per device: a tensor from a Python list is a host-to-device copy (a synchronization every step)
        found = self._index.get(like.device)
        if found is None:
            found = {"obs": torch.tensor(self.keep_obs, dtype=torch.long, device=like.device),
                     "act": torch.tensor(self.keep_act, dtype=torch.long, device=like.device)}
            self._index[like.device] = found
        return found

    def observe(self, policy, critic, base, state, *, fresh, new_chunk):
        idx = self._idx(policy)
        critic = critic.index_select(-1, idx["obs"]) if self.drop_critic else critic
        return policy.index_select(-1, idx["obs"]), critic, base.index_select(-1, idx["act"]), {"base": base}, state

    def act(self, action, ctx, state):
        inner = ctx["base"].clone()
        inner[..., self._idx(action)["act"]] = action
        return inner, state

    def invert(self, executed, ctx, state):
        return executed.index_select(-1, self._idx(executed)["act"]), state


class DropColumnsWrapper(ConversionWrapper):
    def __init__(self, env: Any, cfg: Any) -> None:
        dc = cfg.drop_columns
        super().__init__(env, DropColumns([list(p) for p in dc.observation], list(dc.action)))
        print(f"[drop_columns] observation columns {[list(p) for p in dc.observation]} and action dims {list(dc.action)} "
              f"dropped (the base keeps those dims): observation {self.obs_width}, action {self.action_dim}", flush=True)
