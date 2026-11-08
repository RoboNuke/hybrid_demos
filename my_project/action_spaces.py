"""The residual's action spaces (``residual.action_frame``), one class each, one definition for the env wrapper
(``my_project/residual.py``) and the demo loader (``my_project/demo_loader.py``).

Every space maps one step the same three ways:

* :meth:`ActionSpace.observe` -- the raw env observation (and the base's action this step) to what the agent and the
  critics see (``policy``, ``critic``: None leaves the env's own) and the critics' side input (``side``), from the
  space's per-env state (offsets, the chain's current target) as it stands before the step;
* :meth:`ActionSpace.send` -- the agent's ``r`` to the action the env executes, and the state after the step;
* :meth:`ActionSpace.invert` -- an executed action (a demo's) to the ``r`` that sends it, and the state after the step:
  :meth:`send`'s inverse, exact inside the space's range (``forward(invert(x)) == x``).

The state is a dict of per-env tensors the space owns (:meth:`ActionSpace.initial_state`); ``observe`` resets it where
an episode starts (``fresh``) or the base replans (``new_chunk``), as the space needs. Everything is raw (m, rad, the
env's own action units): the learner's ``state_preprocessor`` normalizes the inputs. Plain torch.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from my_project.residual import critic_state, residual_observation

__all__ = ["ActionSpace", "AbsoluteSpace", "TargetSpace", "EESpace", "MultiplySpace"]

State = Dict[str, torch.Tensor]
Step = Dict[str, Any]


def _rows(mask: Optional[torch.Tensor], like: torch.Tensor) -> torch.Tensor:
    return mask.to(torch.bool).reshape((-1,) + (1,) * (like.dim() - 1))


class ActionSpace:
    """The shared observation pieces (``residual.observe_base_action``, ``observation_drop``, ``critic_state``,
    ``ee_relative_actions``) and the interface. ``keep`` the observation columns the residual sees (None: all);
    ``critic_mode`` "residual" (the residual's own critic state) or "observation"; ``env_critic`` whether the env has a
    critic state of its own; ``ee_rel`` an :class:`~my_project.residual_target.EERelative` or None."""

    #: the agent's per-step action width
    width: int

    def __init__(self, *, keep=None, observe_base: bool = True, critic_mode: str = "residual", env_critic: bool = False,
                 ee_rel=None, steps_per_chunk: int = 1) -> None:
        self.keep = keep
        self.observe_base = bool(observe_base)
        self.critic_mode = str(critic_mode)
        self.env_critic = bool(env_critic)
        self.ee_rel = ee_rel
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

    # ------------------------------------------------------------------ shared pieces
    def _part(self, obs: torch.Tensor) -> torch.Tensor:
        return obs if self.keep is None else obs[..., self.keep]

    def _common(self, obs: torch.Tensor, observed: torch.Tensor, env_critic_obs) -> Step:
        """The agent's observation, the critics' (None: the env's own) and the side input, before a space's own
        appendices: the base observation's kept columns, then the observed base action(s) (EE-relative with
        ``ee_relative_actions``); the critic state the residual's own (``critic_mode: residual``) or the observation
        part."""
        n = obs.shape[0]
        critic = None
        if self.ee_rel is not None:
            part = self._part(obs)
            block = observed.reshape(n, self.steps_per_chunk, -1)
            policy = torch.cat((part, self.ee_rel(block, obs[:, 0:9])), dim=-1) if self.observe_base else part
            if self.critic_mode == "observation" and self.observe_base:
                critic = part
        else:
            policy, _ = residual_observation(obs, observed, self.observe_base, self.keep)
            if self.critic_mode == "observation" and (self.observe_base or self.keep is not None):
                critic = self._part(obs)
        side = observed.reshape(n, -1)
        if self.ee_rel is not None:
            side = torch.cat((side, obs[:, 0:9]), dim=-1)
        residual_critic = (
            None if self.critic_mode == "observation"
            else critic_state(obs, observed, env_critic_obs if self.env_critic else None, self.keep)
        )
        return {"policy": policy, "critic": critic, "side": side, "residual_critic": residual_critic}

    @staticmethod
    def _finish(step: Step) -> Step:
        if step["residual_critic"] is not None:
            step["critic"] = step["residual_critic"]
        return step


class AbsoluteSpace(ActionSpace):
    """``action_frame: absolute``: ``sent = base + scale * r`` (per dim; ``scale_rows`` per env with
    ``residual.phase_scales``), clamped to ``clip`` ``(low, high)`` (``residual.executed_clip``) when set."""

    def __init__(self, scale: torch.Tensor, clip=None, phase_scales: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.scale = scale
        self.clip = clip
        self.phase_scales = bool(phase_scales)
        self.width = int(scale.numel())

    def observe(self, obs, base, state, *, observed=None, env_critic_obs=None, fresh=None, new_chunk=None, routed=None,
                scale_rows=None):
        observed = base if observed is None else observed
        step = self._common(obs, observed, env_critic_obs)
        if self.phase_scales:
            # the critics' scale from r to the executed action: this transition's phase's (residual_critic_action)
            step["side"] = torch.cat((step["side"], scale_rows), dim=-1)
        step["base"] = base
        step["scale"] = scale_rows if self.phase_scales else self.scale
        return self._finish(step), state

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


class TargetSpace(ActionSpace):
    """``action_frame: target`` (:class:`~my_project.residual_target.TargetResidual`); with ``accumulate``
    (``residual.target_accumulate``) the residual increments an offset the base's targets carry, reset when the base
    replans, observed by the agent and the critics (over ``reach`` steps) and carried in the side input."""

    def __init__(self, target, accumulate: bool = False, reach: int = 1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.target = target
        self.accumulate = bool(accumulate)
        self.reach = int(reach)
        self.width = int(target.width)

    def initial_state(self, n, device):
        if not self.accumulate:
            return {}
        off_pos, off_rot = self.target.identity(n, device)
        return {"off_pos": off_pos, "off_rot": off_rot}

    def observe(self, obs, base, state, *, observed=None, env_critic_obs=None, fresh=None, new_chunk=None, routed=None,
                scale_rows=None):
        observed = base if observed is None else observed
        step = self._common(obs, observed, env_critic_obs)
        if self.accumulate:
            # the offset goes back to zero when the base replans; it is state: the agent, the critics and the side
            # input all carry it
            zero_p, eye = self.target.identity(1, obs.device)
            state = {"off_pos": torch.where(_rows(new_chunk, state["off_pos"]), zero_p, state["off_pos"]),
                     "off_rot": torch.where(_rows(new_chunk, state["off_rot"]), eye, state["off_rot"])}
            off = self.target.offset_obs(state["off_pos"], state["off_rot"], self.reach)
            step["policy"] = torch.cat((step["policy"], off), dim=-1)
            step["critic"] = torch.cat((self._part(obs), off), dim=-1)
            step["side"] = torch.cat((step["side"], off), dim=-1)
        step["base"] = base
        return self._finish(step), state

    def send(self, r, step, state):
        r = r[:, : self.width]
        if self.accumulate:
            sent, off_pos, off_rot = self.target.accumulate(step["base"], r, state["off_pos"], state["off_rot"])
            return sent, {"off_pos": off_pos, "off_rot": off_rot}
        return self.target.step(step["base"], r), state

    def invert(self, executed, step, state):
        if self.accumulate:
            r, off_pos, off_rot = self.target.invert_accumulate(step["base"], executed, state["off_pos"], state["off_rot"])
            return r, {"off_pos": off_pos, "off_rot": off_rot}
        return self.target.invert(step["base"], executed), state


class EESpace(ActionSpace):
    """``action_frame: ee`` (:class:`~my_project.residual_ee.EEResidual`): the EE's orientation is the observation's;
    with ``accumulate`` the offset resets when the base replans and is observed by the agent and the critics."""

    def __init__(self, ee, **kwargs) -> None:
        super().__init__(**kwargs)
        self.ee = ee
        from my_project.residual_ee import WIDTH

        self.width = WIDTH

    def initial_state(self, n, device):
        o, O = self.ee.identity(n, device)
        return {"o": o, "O": O, "R_ee": torch.eye(3, device=device).expand(n, 3, 3).clone()}

    def observe(self, obs, base, state, *, observed=None, env_critic_obs=None, fresh=None, new_chunk=None, routed=None,
                scale_rows=None):
        from my_project.residual_ee import ee_rotation

        observed = base if observed is None else observed
        step = self._common(obs, observed, env_critic_obs)
        state = dict(state, R_ee=ee_rotation(obs))
        if self.ee.accumulate:  # the offset goes back to zero when the base re-plans
            zero_o, eye = self.ee.identity(1, obs.device)
            state["o"] = torch.where(_rows(new_chunk, state["o"]), zero_o, state["o"])
            state["O"] = torch.where(_rows(new_chunk, state["O"]), eye, state["O"])
            offset = self.ee.offset_obs(state["R_ee"], state["o"], state["O"])
            step["policy"] = torch.cat((step["policy"], offset), dim=-1)
            step["residual_critic"] = torch.cat((step["residual_critic"], offset), dim=-1)
        step["base"] = base
        return self._finish(step), state

    def send(self, r, step, state):
        sent, o, O = self.ee.step(step["base"], state["R_ee"], state["o"], state["O"], r[:, : self.width])
        return sent, dict(state, o=o, O=O)

    def invert(self, executed, step, state):
        r, o, O = self.ee.invert(step["base"], executed, state["R_ee"], state["o"], state["O"])
        return r, dict(state, o=o, O=O)


class MultiplySpace(ActionSpace):
    """``action_frame: multiply`` (:class:`~my_project.residual_multiply.MultiplyResidual`): ``increment`` chains each
    step onto the current target (the last sent one; the EE at an episode's start and at every chunk start)."""

    def __init__(self, multiply, action_dim: int, **kwargs) -> None:
        super().__init__(**kwargs)
        if not self.observe_base:
            raise NotImplementedError("action_frame: multiply observes the base's motion: set residual.observe_base_action: true")
        self.multiply = multiply
        self.action_dim = int(action_dim)
        self.increment = multiply.reference == "increment"
        from my_project.residual_multiply import WIDTH

        self.width = WIDTH

    def initial_state(self, n, device):
        if not self.increment:
            return {}
        return {"prev_sent": torch.zeros((n, self.action_dim), device=device),
                "prev_base": torch.zeros((n, self.action_dim), device=device)}

    def observe(self, obs, base, state, *, observed=None, env_critic_obs=None, fresh=None, new_chunk=None, routed=None,
                scale_rows=None):
        if self.increment and fresh is not None:
            # a new episode's current target is the EE (it has sent nothing yet)
            new = fresh.to(torch.bool).reshape(-1, 1)
            at_ee = torch.cat((obs[:, 0:9], base[:, 9:]), dim=-1)
            state = {"prev_sent": torch.where(new, at_ee, state["prev_sent"]),
                     "prev_base": torch.where(new, base, state["prev_base"])}
        inputs = self.multiply.step_inputs(obs, base, self.keep, routed, prev_sent=state.get("prev_sent"),
                                           prev_base=state.get("prev_base"), chunk_start=new_chunk)
        step = {"policy": inputs["policy"], "critic": inputs["critic"], "side": inputs["side"], "d": inputs["d"],
                "ref": inputs["ref"], "base": base, "routed": routed}
        return step, state

    def send(self, r, step, state):
        sent = self.multiply.sent(step["base"], step["d"], step["ref"], r[:, : self.width], step["routed"])
        return sent, ({"prev_sent": sent, "prev_base": step["base"]} if self.increment else state)

    def invert(self, executed, step, state):
        r = self.multiply.invert(step["d"], executed, step["ref"])
        # the chain carries the executed target on (a step in a phase without a residual executed the base)
        return r, ({"prev_sent": executed, "prev_base": step["base"]} if self.increment else state)
