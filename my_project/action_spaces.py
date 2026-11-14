"""The residual's action spaces (``residual.action_space``), one class each, one definition for the env wrapper
(``my_project/residual.py``) and the demo loader (``my_project/demo_loader.py``).

Every space maps one step the same three ways:

* :meth:`ActionSpace.observe` -- the raw env observation (and the base's action this step) to what the agent and the
  critics see (``policy``, ``critic``: None leaves the env's own) and the critics' side input (``side``), from the
  space's per-env state (the chain's current target, the chunk's EE pose) as it stands before the step; also
  ``motion_scale``, the base's motion in r units (DDPG's ``exploration_std_mode: adaptive``);
* :meth:`ActionSpace.send` -- the agent's ``r`` to the action the env executes, and the state after the step;
* :meth:`ActionSpace.invert` -- an executed action (a demo's) to the ``r`` that sends it, and the state after the step:
  :meth:`send`'s inverse, exact inside the space's range (``forward(invert(x)) == x``).

``per_dim`` (:class:`PerDimSpace`) adds ``scale * r`` per action dim in the env's units (any task); ``pose``
(:class:`PoseSpace`, :mod:`my_project.pose_residual`) acts on a furniture base's absolute pose targets. The state is a dict
of per-env tensors the space owns (:meth:`ActionSpace.initial_state`); ``observe`` resets it where an episode starts
(``fresh``) or the base replans (``new_chunk``). Everything is raw (m, rad, the env's own action units): the learner's
input normalization is the model's business. Plain torch.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from my_project.residual import critic_state, residual_observation

__all__ = ["ActionSpace", "PerDimSpace", "PoseSpace"]

State = Dict[str, torch.Tensor]
Step = Dict[str, Any]


class ActionSpace:
    """The shared observation pieces (``residual.observe_base_action``, ``observation_drop``, ``critic_state``) and the
    interface. ``keep`` the observation columns the residual sees (None: all); ``critic_mode`` "residual" (the
    residual's own critic state) or "observation"; ``env_critic`` whether the env has a critic state of its own."""

    #: the agent's per-step action width
    width: int

    def __init__(self, *, keep=None, observe_base: bool = True, critic_mode: str = "residual", env_critic: bool = False,
                 steps_per_chunk: int = 1) -> None:
        self.keep = keep
        self.observe_base = bool(observe_base)
        self.critic_mode = str(critic_mode)
        self.env_critic = bool(env_critic)
        self.steps_per_chunk = int(steps_per_chunk)

    # ------------------------------------------------------------------ interface
    def initial_state(self, n: int, device) -> State:
        return {}

    def observe(self, obs: torch.Tensor, base: torch.Tensor, state: State, *, observed: Optional[torch.Tensor] = None,
                env_critic_obs: Optional[torch.Tensor] = None, fresh: Optional[torch.Tensor] = None,
                new_chunk: Optional[torch.Tensor] = None, routed: Optional[torch.Tensor] = None,
                scale_rows: Optional[torch.Tensor] = None) -> Tuple[Step, State]:
        raise NotImplementedError

    def send(self, r: torch.Tensor, step: Step, state: State) -> Tuple[torch.Tensor, State]:
        raise NotImplementedError

    def invert(self, executed: torch.Tensor, step: Step, state: State) -> Tuple[torch.Tensor, State]:
        raise NotImplementedError


class PerDimSpace(ActionSpace):
    """``action_space: per_dim``: ``sent = base + scale * r`` per action dim (``scale_rows`` per env with
    ``residual.phase_scales``), clamped to ``clip`` ``(low, high)`` (``residual.executed_clip``) when set. The agent sees
    the kept observation columns and the observed base action(s); the critics the residual's critic state, or the
    observation part with the base action(s) as the side input (``critic_state: observation``)."""

    def __init__(self, scale: torch.Tensor, clip=None, phase_scales: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.scale = scale
        self.clip = clip
        self.phase_scales = bool(phase_scales)
        self.width = int(scale.numel())

    def observe(self, obs, base, state, *, observed=None, env_critic_obs=None, fresh=None, new_chunk=None, routed=None,
                scale_rows=None):
        observed = base if observed is None else observed
        n = obs.shape[0]
        policy, _ = residual_observation(obs, observed, self.observe_base, self.keep)
        critic = None
        if self.critic_mode == "observation":
            if self.observe_base or self.keep is not None:
                critic = obs if self.keep is None else obs[..., self.keep]
        else:
            critic = critic_state(obs, observed, env_critic_obs if self.env_critic else None, self.keep)
        side = observed.reshape(n, -1)
        scale = scale_rows if self.phase_scales else self.scale
        if self.phase_scales:
            # the critics' scale from r to the executed action: this transition's phase's (residual_critic_action)
            side = torch.cat((side, scale_rows), dim=-1)
        # the base action in r units (exploration_std_mode: adaptive); a dim with no scale gets none
        safe = torch.where(scale > 0, scale, torch.ones_like(scale))
        motion = torch.where(scale > 0, base.abs() / safe, torch.zeros_like(base))
        step = {"policy": policy, "critic": critic, "side": side, "base": base, "scale": scale, "motion_scale": motion}
        return step, state

    def send(self, r, step, state):
        width = step["base"].shape[-1]
        correction = step["scale"] * r[:, :width]
        sent = step["base"] + correction
        if self.clip is not None:
            sent = torch.maximum(torch.minimum(sent, self.clip[1]), self.clip[0])
        step["correction"] = correction
        return sent, state

    def invert(self, executed, step, state):
        return (executed - step["base"]) / step["scale"], state


class PoseSpace(ActionSpace):
    """``action_space: pose`` (:class:`~my_project.pose_residual.PoseResidual`): its state is the chain's current target
    and the base's previous target (``action_target: target``; the EE at an episode's start) and the EE pose at the
    chunk's start (the ``chunk`` frame), reset where the base replans."""

    def __init__(self, pose, action_dim: int, **kwargs) -> None:
        super().__init__(**kwargs)
        if not self.observe_base:
            raise NotImplementedError("action_space: pose observes the base's motion: set residual.observe_base_action: true")
        if self.critic_mode != "observation" or self.env_critic:
            raise NotImplementedError("action_space: pose builds the critics' observation itself: critic_state: observation, "
                                      "on an env without a critic state of its own")
        self.pose = pose
        self.action_dim = int(action_dim)
        self.width = pose.width

    def initial_state(self, n, device):
        return {"prev_sent": torch.zeros((n, self.action_dim), device=device),
                "prev_base": torch.zeros((n, self.action_dim), device=device),
                "ee_chunk": torch.zeros((n, 9), device=device)}

    def observe(self, obs, base, state, *, observed=None, env_critic_obs=None, fresh=None, new_chunk=None, routed=None,
                scale_rows=None):
        if new_chunk is None:
            raise ValueError("action_space: pose resets its chain and its chunk frame where the base replans: pass new_chunk")
        ee = obs[:, 0:9]
        state = dict(state)
        if fresh is not None:
            # a new episode's current target is the EE (it has sent nothing yet)
            new = fresh.to(torch.bool).reshape(-1, 1)
            at_ee = torch.cat((ee, base[:, 9:]), dim=-1)
            state["prev_sent"] = torch.where(new, at_ee, state["prev_sent"])
            state["prev_base"] = torch.where(new, base, state["prev_base"])
        state["ee_chunk"] = torch.where(new_chunk.to(torch.bool).reshape(-1, 1), ee, state["ee_chunk"])
        if routed is None:
            routed = torch.ones(obs.shape[0], dtype=torch.bool, device=obs.device)
        inputs = self.pose.step_inputs(obs, base, self.keep, routed, ee_chunk=state["ee_chunk"],
                                       prev_sent=state["prev_sent"], prev_base=state["prev_base"], chunk_start=new_chunk)
        step = {**inputs, "base": base, "routed": routed}
        return step, state

    def send(self, r, step, state):
        sent = self.pose.sent(step["base"], step["d"], step["anchor"], step["A"], r[:, : self.width], step["routed"])
        return sent, dict(state, prev_sent=sent, prev_base=step["base"])

    def invert(self, executed, step, state):
        r = self.pose.invert(executed, step["d"], step["anchor"], step["A"])
        # the chain carries the executed target on (a step in a phase without a residual executed the base)
        return r, dict(state, prev_sent=executed, prev_base=step["base"])
