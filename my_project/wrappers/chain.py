"""What every residual wrapper above the base policy shares (:class:`ChainWrapper`), and the conversion wrappers' one
shape (:class:`ConversionWrapper` around a :class:`Conversion`).

A conversion (drop columns, target tracking, EE-relative observation) is plain torch with no env: it takes the step's
observation columns (the agent's and the critics') and the base action as the wrapper below produced them, and gives
its own; it turns its own action back into the wrapper below's to step the env, and -- for demos -- an action the env
executed into its own. The wrapper around it hooks the env's observations, sets the spaces' widths, and keeps the
step's context between the observation and the action. The demo conversion (:mod:`my_project.wrappers.residual_policy.
demos`) runs the same conversions offline, so a demo goes through exactly the code the env runs.

Laid out by :class:`~my_project.wrappers.residual_policy.base_policy.BasePolicy`: the agent's observation is
``[observation columns, base action]`` (the base action right after the ``obs_width`` columns, ``action_dim`` wide; a
history or wrench wrapper outside the residual appends after it), the critics' the observation columns.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import gymnasium as gym
import torch

__all__ = ["ChainWrapper", "Conversion", "ConversionWrapper"]

State = Dict[str, torch.Tensor]


class ChainWrapper(gym.Wrapper):
    """A residual wrapper above the base policy. ``obs_width`` / ``critic_width``: its observation columns; ``action_dim``
    its action; ``base_action`` this step's base action in its action space."""

    def __init__(self, env: Any) -> None:
        super().__init__(env)
        from my_project.envhooks import hooks_for

        from my_project.wrappers.residual_policy.base_policy import BasePolicy, find_wrapper

        if not isinstance(env, (BasePolicy, ChainWrapper)):
            raise TypeError(f"{type(self).__name__} goes right above the base policy or another residual wrapper, got "
                            f"{type(env).__name__}")
        self.inner = env
        self.base_policy = env if isinstance(env, BasePolicy) else env.base_policy
        self.hooks = hooks_for(env.unwrapped)
        self.num_envs, self.device = self.base_policy.num_envs, self.base_policy.device
        self.obs_width, self.critic_width, self.action_dim = env.obs_width, env.critic_width, env.action_dim
        self._base_action = None

    @property
    def base_action(self) -> torch.Tensor:
        return self.inner.base_action if self._base_action is None else self._base_action

    def chain(self):
        """The residual wrappers from the base policy up to this one, innermost first."""
        out, node = [], self
        while isinstance(node, ChainWrapper):
            out.append(node)
            node = node.inner
        out.append(node)
        return list(reversed(out))


class Conversion:
    """One conversion's maps (the module docstring). ``build`` sets the widths it gives from the ones it gets."""

    obs_width: int
    critic_width: int
    action_dim: int

    def build(self, obs_width: int, critic_width: int, action_dim: int) -> None:
        raise NotImplementedError

    def initial_state(self, n: int, device: Any) -> State:
        return {}

    def observe(self, policy: torch.Tensor, critic: torch.Tensor, base: torch.Tensor, state: State, *, fresh: torch.Tensor,
                new_chunk: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any], State]:
        """``(policy columns, critic columns, base action, context, state)`` from the wrapper below's."""
        raise NotImplementedError

    def act(self, action: torch.Tensor, ctx: Dict[str, Any], state: State) -> Tuple[torch.Tensor, State]:
        """This wrapper's action -> the wrapper below's, and the state after the step."""
        raise NotImplementedError

    def invert(self, executed: torch.Tensor, ctx: Dict[str, Any], state: State) -> Tuple[torch.Tensor, State]:
        """An action the wrapper below executed (a demo's) -> this wrapper's, and the state after the step."""
        raise NotImplementedError


class ConversionWrapper(ChainWrapper):
    """A :class:`Conversion` on the env: hooks its observations and steps with its action map."""

    def __init__(self, env: Any, conversion: Conversion) -> None:
        super().__init__(env)
        inner_obs, inner_critic, inner_act = self.obs_width, self.critic_width, self.action_dim
        conversion.build(inner_obs, inner_critic, inner_act)
        self.conversion = conversion
        self.obs_width, self.critic_width, self.action_dim = conversion.obs_width, conversion.critic_width, conversion.action_dim
        self.hooks.widen((self.obs_width + self.action_dim) - (inner_obs + inner_act), self.critic_width - inner_critic)
        if self.action_dim != inner_act:
            self.hooks.set_agent_action_width(self.action_dim)
        self.state = conversion.initial_state(self.num_envs, self.device)
        self.ctx: Dict[str, Any] = {}
        self._inner_obs = inner_obs
        self.hooks.hook_observations(self._observations)

    def _observations(self, observations):
        columns = observations["policy"][..., : self._inner_obs]
        bp = self.base_policy
        policy, critic, base, self.ctx, self.state = self.conversion.observe(
            columns, observations["critic"], self.inner.base_action, self.state, fresh=bp.fresh, new_chunk=bp.new_chunk)
        self._base_action = base
        observations["policy"] = torch.cat((policy, base), dim=-1)
        observations["critic"] = critic
        return observations

    def step(self, action):
        inner, self.state = self.conversion.act(action, self.ctx, self.state)
        return self.env.step(inner)
