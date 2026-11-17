"""``multiply_residual.*``: the agent's ``r`` multiplies the base action, ``sent = r * base`` on every dim of the action
space the wrapper below gives (:class:`MultiplyResidual`). r is the multiplier itself: its range and start are the
learner's (``ddpg.action_range``, ``model.actor.output_bias``); the base alone is r = 1 (the neutral action). A dim it
must not scale (the gripper) is dropped below (``drop_columns.action``). The critics take the sent action, ``r *
base``, rebuilt from the side input (the base action); the adaptive noise factor is 1 (r = 1 is the base's own action).

``r_range_clip`` (not 0) limits the change as an additive residual with ``r`` in ``+-r_range_clip`` is limited: the sent
action clamped to ``base +- r_range_clip x scale`` per dim, ``scale`` the additive residual's (``additive_residual.scale``
or ``scale_fraction x`` the base's demo half range, :func:`~my_project.wrappers.residual_policy.additive.flat_scale`); the
critics' action is clamped the same way. With ``r_range_clip_actor`` the limit clamps the policy's own action instead:
the actor's ``r`` (its ``mid + half * tanh``, unchanged) clamped so ``r * base`` stays within ``base +- limit``, i.e. r
within ``1 +- limit / |base|`` per row and dim (:meth:`MultiplyResidual.action_bound`; straight-through: the actor's
gradient passes the clamp); the exploration noise is added
after the clamp, so the sent action is ``clamped r * base + noise * base``, and neither it nor the critics' action is
clamped again.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from my_project.wrappers.residual_policy.combine import SIDE_INPUT, CombineWrapper

__all__ = ["MultiplyResidual", "MultiplyCriticAction", "clamp_change"]


def clamp_change(sent: torch.Tensor, base: torch.Tensor, limit: Optional[torch.Tensor]) -> torch.Tensor:
    """``sent`` clamped to ``base +- limit`` per dim; as it is with no limit."""
    if limit is None:
        return sent
    limit = limit.to(sent.device)
    return torch.maximum(torch.minimum(sent, base + limit), base - limit)


class MultiplyCriticAction:
    """``(critic_inputs, r, side) -> r * base`` (clamped to ``base +- limit``), the critics' action, from the side input
    :data:`SIDE_INPUT`."""

    def __init__(self, width: int, limit: Optional[torch.Tensor] = None) -> None:
        self.side_inputs = {SIDE_INPUT: int(width)}
        self.limit = limit

    def __call__(self, critic_inputs: torch.Tensor, actions: torch.Tensor, side) -> torch.Tensor:
        base = side[SIDE_INPUT]
        return clamp_change(actions * base, base, self.limit)


class MultiplyResidual(CombineWrapper):
    neutral = 1.0

    def __init__(self, env: Any, cfg: Any) -> None:
        super().__init__(env, cfg)
        from my_project.cfg import q_section

        floor = [float(v) for v in cfg.multiply_residual.invert_floor]
        if len(floor) not in (1, self.action_dim):
            raise ValueError(f"multiply_residual.invert_floor: one number or {self.action_dim} (one per dim), got {len(floor)}")
        self.floor = torch.tensor(floor * (self.action_dim if len(floor) == 1 else 1), device=self.device)
        rng = getattr(q_section(cfg), "action_range", None)
        self.r_range: Optional[tuple] = None if rng is None else (float(rng[0]), float(rng[1]))
        #: the most the sent action may move from the base per dim (r_range_clip x the additive residual's scale)
        self.limit: Optional[torch.Tensor] = None
        clip = float(cfg.multiply_residual.r_range_clip)
        if clip != 0:
            from my_project.wrappers.residual_policy.additive import flat_scale

            self.limit = clip * flat_scale(self, cfg)
        #: r_range_clip_actor: the limit bounds the actor (action_bound), not the sent action
        self.bound_actor = bool(cfg.multiply_residual.r_range_clip_actor)
        print(f"[multiply_residual] sent = r * base over the {self.action_dim}-wide action, r in {self.r_range} "
              f"(the base alone at 1)"
              + ("" if self.limit is None else f", {'the actor' if self.bound_actor else 'clamped'} to base +- "
                 f"{[round(v, 5) for v in self.limit.tolist()]} (r_range_clip {clip} x the additive scale)"
                 + (", the noise unclamped" if self.bound_actor else "")), flush=True)

    @property
    def sent_limit(self) -> Optional[torch.Tensor]:
        """The clamp on the sent (and the critics') action: the limit, unless it bounds the actor instead."""
        return None if self.bound_actor else self.limit

    def combine(self, r, side):
        return clamp_change(r * side, side, self.sent_limit)

    def invert(self, executed, side):
        """``executed / base`` per dim, 1 where ``|base|`` is under the floor, clamped to the learner's range."""
        big = side.abs() >= self.floor
        ratio = torch.where(big, executed / torch.where(big, side, torch.ones_like(side)), torch.ones_like(side))
        return ratio if self.r_range is None else ratio.clamp(*self.r_range)

    def step_metrics(self, r, metrics) -> None:
        metrics["residual/multiplier"] = r.mean(dim=-1)

    def critic_action(self) -> MultiplyCriticAction:
        return MultiplyCriticAction(self.action_dim, self.sent_limit)

    def action_bound(self):
        """With ``r_range_clip_actor``: ``observations -> (1 - limit / |base|, 1 + limit / |base|)`` per row and dim, the
        base read from the observation's base-action columns, the actor's r clamped to it so ``r * base`` moves at most
        ``limit`` from the base; else None."""
        if not self.bound_actor:
            return None
        start, stop = self.obs_width, self.obs_width + self.action_dim
        limit = self.limit.to(self.device)

        def window(observations: torch.Tensor):
            reach = limit / observations[..., start:stop].abs().clamp_min(1e-12)
            return 1.0 - reach, 1.0 + reach

        return window

    def noise_factor(self):
        start, stop = self.obs_width, self.obs_width + self.action_dim

        def factor(observations: torch.Tensor) -> torch.Tensor:
            return torch.ones_like(observations[..., start:stop])

        return factor
