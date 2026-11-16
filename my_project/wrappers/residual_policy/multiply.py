"""``multiply_residual.*``: the agent's ``r`` multiplies the base action, ``sent = r * base`` on every dim of the action
space the wrapper below gives (:class:`MultiplyResidual`). r is the multiplier itself: its range and start are the
learner's (``ddpg.action_range``, ``model.actor.output_bias``); the base alone is r = 1 (the neutral action). A dim it
must not scale (the gripper) is dropped below (``drop_columns.action``). The critics take the sent action, ``r *
base``, rebuilt from the side input (the base action); the adaptive noise factor is 1 (r = 1 is the base's own action).
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from my_project.wrappers.residual_policy.combine import SIDE_INPUT, CombineWrapper

__all__ = ["MultiplyResidual", "MultiplyCriticAction"]


class MultiplyCriticAction:
    """``(critic_inputs, r, side) -> r * base``, the critics' action, from the side input :data:`SIDE_INPUT`."""

    def __init__(self, width: int) -> None:
        self.side_inputs = {SIDE_INPUT: int(width)}

    def __call__(self, critic_inputs: torch.Tensor, actions: torch.Tensor, side) -> torch.Tensor:
        return actions * side[SIDE_INPUT]


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
        print(f"[multiply_residual] sent = r * base over the {self.action_dim}-wide action, r in {self.r_range} "
              f"(the base alone at 1)", flush=True)

    def combine(self, r, side):
        return r * side

    def invert(self, executed, side):
        """``executed / base`` per dim, 1 where ``|base|`` is under the floor, clamped to the learner's range."""
        big = side.abs() >= self.floor
        ratio = torch.where(big, executed / torch.where(big, side, torch.ones_like(side)), torch.ones_like(side))
        return ratio if self.r_range is None else ratio.clamp(*self.r_range)

    def step_metrics(self, r, metrics) -> None:
        metrics["residual/multiplier"] = r.mean(dim=-1)

    def critic_action(self) -> MultiplyCriticAction:
        return MultiplyCriticAction(self.action_dim)

    def noise_factor(self):
        start, stop = self.obs_width, self.obs_width + self.action_dim

        def factor(observations: torch.Tensor) -> torch.Tensor:
            return torch.ones_like(observations[..., start:stop])

        return factor
