"""How a project wrapper reaches into the env underneath it, for both env families.

A wrapper that appends to the observation (history, residual) needs three things from the env:
where the observation is produced (to post-process it), how wide the spaces are (to widen
them), and which envs just reset (to start a fresh history). A Direct env (Forge) exposes
``_get_observations``, an integer ``cfg.observation_space`` / ``state_space`` and
``_configure_gym_env_spaces``; a manager-based env (Isaac-Furniture) has an observation
manager and gym spaces built per group. :func:`hooks_for` hands back the matching adapter, so
the wrappers themselves have one implementation.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

import torch

__all__ = ["EnvHooks", "DirectHooks", "ManagerHooks", "hooks_for"]


class EnvHooks:
    """The adapter interface."""

    family: str

    def __init__(self, unwrapped: Any) -> None:
        self.unwrapped = unwrapped

    @property
    def device(self):
        return self.unwrapped.device

    @property
    def num_envs(self) -> int:
        return int(self.unwrapped.num_envs)

    def obs_width(self) -> int:
        raise NotImplementedError

    def action_width(self) -> int:
        raise NotImplementedError

    def has_critic(self) -> bool:
        raise NotImplementedError

    def add_obs_dims(self, n: int) -> None:
        """Widen the policy observation (and the critic state, if there is one) by ``n``."""
        raise NotImplementedError

    def critic_width(self) -> int:
        """The critic state's width (the env must have one, :meth:`has_critic`)."""
        raise NotImplementedError

    def widen(self, policy: int, critic: int) -> None:
        """Widen the policy observation by ``policy`` and the critic state (which must exist) by ``critic``."""
        raise NotImplementedError

    def create_critic(self, width: int) -> None:
        """Give an env without a critic state one, ``width`` wide (its values are the hooked observation
        function's ``"critic"`` entry, which the caller must produce from then on)."""
        raise NotImplementedError

    def set_agent_action_width(self, n: int) -> None:
        """Announce an ``n``-wide action in ``[-1, 1]`` to the agent (skrl reads the unwrapped env's action
        space), for a wrapper that turns it into the env's own action before stepping."""
        raise NotImplementedError(f"{type(self).__name__} cannot change the agent's action width")

    def hook_observations(self, fn: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]) -> None:
        """Run ``fn`` on the env's observation dict every time it is produced."""
        raise NotImplementedError

    def fresh(self) -> torch.Tensor:
        """``(num_envs,)`` bool: envs whose episode began this step."""
        return self.unwrapped.episode_length_buf == 0

    @property
    def observation_space(self):
        return self.unwrapped.observation_space

    @property
    def action_space(self):
        return self.unwrapped.action_space


class DirectHooks(EnvHooks):
    """Forge / any ``DirectRLEnv``."""

    family = "forge"

    def obs_width(self) -> int:
        return int(self.unwrapped.cfg.observation_space)

    def action_width(self) -> int:
        return int(self.unwrapped.cfg.action_space)

    def has_critic(self) -> bool:
        return bool(getattr(self.unwrapped.cfg, "state_space", 0))

    def add_obs_dims(self, n: int) -> None:
        u = self.unwrapped
        u.cfg.observation_space += int(n)
        if self.has_critic():
            u.cfg.state_space += int(n)
        self._rebuild()

    def critic_width(self) -> int:
        return int(self.unwrapped.cfg.state_space)

    def widen(self, policy: int, critic: int) -> None:
        u = self.unwrapped
        u.cfg.observation_space += int(policy)
        u.cfg.state_space += int(critic)
        self._rebuild()

    def create_critic(self, width: int) -> None:
        self.unwrapped.cfg.state_space = int(width)
        self._rebuild()

    def _rebuild(self) -> None:
        u = self.unwrapped
        native = u.actions  # the rebuild re-allocates the action buffer at the cfg's width
        u._configure_gym_env_spaces()
        u.actions = native

    def set_agent_action_width(self, n: int) -> None:
        u = self.unwrapped
        u.cfg.action_space = int(n)
        native = u.actions  # the rebuild re-allocates the action buffer at the cfg's width
        u._configure_gym_env_spaces()
        u.actions = native

    def hook_observations(self, fn) -> None:
        u = self.unwrapped
        original = u._get_observations
        u._get_observations = lambda: fn(original())


class ManagerHooks(EnvHooks):
    """``ManagerBasedRLEnv`` (Isaac-Furniture-*)."""

    family = "furniture"

    def obs_width(self) -> int:
        return int(self.unwrapped.single_observation_space["policy"].shape[-1])

    def action_width(self) -> int:
        return int(self.unwrapped.single_action_space.shape[-1])

    def has_critic(self) -> bool:
        return "critic" in self.unwrapped.single_observation_space.spaces

    def add_obs_dims(self, n: int) -> None:
        import gymnasium as gym
        from gymnasium.vector.utils import batch_space

        u = self.unwrapped
        for group in ("policy", "critic"):
            if group in u.single_observation_space.spaces:
                old = u.single_observation_space[group]
                shape = (int(old.shape[-1]) + int(n),)
                u.single_observation_space[group] = gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=shape)
        u.observation_space = batch_space(u.single_observation_space, u.num_envs)

    def critic_width(self) -> int:
        return int(self.unwrapped.single_observation_space["critic"].shape[-1])

    def _set_width(self, group: str, width: int) -> None:
        import gymnasium as gym
        from gymnasium.vector.utils import batch_space

        u = self.unwrapped
        u.single_observation_space[group] = gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(int(width),))
        u.observation_space = batch_space(u.single_observation_space, u.num_envs)

    def widen(self, policy: int, critic: int) -> None:
        self._set_width("policy", self.obs_width() + int(policy))
        self._set_width("critic", self.critic_width() + int(critic))

    def create_critic(self, width: int) -> None:
        self._set_width("critic", width)

    def set_agent_action_width(self, n: int) -> None:
        import gymnasium as gym
        from gymnasium.vector.utils import batch_space

        u = self.unwrapped
        u.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(int(n),))
        u.action_space = batch_space(u.single_action_space, u.num_envs)

    def hook_observations(self, fn) -> None:
        manager = self.unwrapped.observation_manager
        original = manager.compute
        # Isaac Lab calls compute(update_history=...): pass its arguments through
        manager.compute = lambda *args, **kwargs: fn(original(*args, **kwargs))


def hooks_for(unwrapped: Any) -> EnvHooks:
    if hasattr(unwrapped, "observation_manager"):
        return ManagerHooks(unwrapped)
    if hasattr(unwrapped, "_get_observations") and hasattr(unwrapped.cfg, "observation_space"):
        return DirectHooks(unwrapped)
    raise TypeError(f"{type(unwrapped).__name__}: neither a DirectRLEnv nor a ManagerBasedRLEnv")
