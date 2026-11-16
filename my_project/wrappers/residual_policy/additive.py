"""``additive_residual.*``: the agent's ``r`` added to the base action, ``sent = base + scale * r`` on every dim of the
action space the wrapper below gives (:class:`AdditiveResidual`).

The scale per dim is ``additive_residual.scale``, or -- where the action is still the base's own (only the base policy,
dropped dims and the EE-relative observation below) -- one of ResFiT's rules on the base's actions:
``scale_fraction x`` the half range per dim (of the base normalizer, or of its demo dataset floored at
``half_range_floor``), or per routed phase its own scale from ``phase_stats`` (phase_v7: each phase's median demo
correction; a phase without a residual gets 0). ``clip`` clamps the sent action to ``center +- clip x half range``.

The critics take the sent action, ``base + scale * r``, rebuilt from the side input (the base action, then each env's
scale with phase scales). The adaptive noise factor is the base action in r units, ``|base| / scale``, read from the
agent's observation (the base action right after its observation columns); the neutral action is 0.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from my_project.wrappers.residual_policy.combine import SIDE_INPUT, CombineWrapper

__all__ = ["AdditiveResidual", "AdditiveCriticAction", "native_dims", "demo_action_range", "phase_scale_table"]


def native_dims(wrapper) -> Optional[list]:
    """The env's action dims the wrapper's action space keeps, in order, when every wrapper below leaves the action the
    base's own (drop columns, the EE-relative observation); None when one changes what an action is (target tracking)."""
    from my_project.wrappers.action_spaces.drop_columns import DropColumns
    from my_project.wrappers.observation_spaces.ee_relative import EERelative

    dims = list(range(wrapper.base_policy.action_dim))
    for node in wrapper.chain()[1:-1]:
        conversion = node.conversion
        if isinstance(conversion, DropColumns):
            dims = [dims[k] for k in conversion.keep_act]
        elif not isinstance(conversion, EERelative):
            return None
    return dims


def demo_action_range(base_dir: Any) -> Dict[str, torch.Tensor]:
    """The action's min and max per column over every frame of the base's demo dataset (its resolved config's
    ``bc.dataset``, ``bc.obs_key``, ``bc.action_key``): ``half_range_floor``'s half range."""
    from omegaconf import OmegaConf

    from robonuke_rl_core.evaluation import resolved_config_path

    from my_project.bc.dataset import load_lerobot

    bc = OmegaConf.load(resolved_config_path(base_dir)).bc
    demos = load_lerobot(str(bc.dataset), str(bc.obs_key), str(bc.action_key))
    return {"action_min": demos.action.min(0).values, "action_max": demos.action.max(0).values}


def phase_scale_table(path: Any, names, routes, width: int, device: Any) -> torch.Tensor:
    """Per learner label (``std_label_names``) its scale ``(L, width)`` from ``phase_stats``; a label without a residual
    gets 0 (no correction)."""
    stats = torch.load(path, map_location="cpu", weights_only=False)
    if list(stats.get("names", [])) != list(names):
        raise ValueError(f"additive_residual.phase_stats {path} is for labels {stats.get('names')}, the learner's are {list(names)}")
    missing = [r for r in routes if r not in stats["phases"]]
    if missing:
        raise ValueError(f"additive_residual.phase_stats {path} has no stats for routed phases {missing}")
    scale = torch.zeros((len(names), int(width)))
    for route in routes:
        scale[list(names).index(route)] = torch.as_tensor(stats["phases"][route]["scale"])
    return scale.to(device)


class AdditiveCriticAction:
    """``(critic_inputs, r, side) -> base + scale * r``, the critics' action, from the side input :data:`SIDE_INPUT`: the
    base action, then (``side_scale``, phase scales) each transition's own scale."""

    def __init__(self, width: int, scale: Optional[torch.Tensor] = None, side_scale: bool = False) -> None:
        self.width, self.scale, self.side_scale = int(width), scale, bool(side_scale)
        self.side_inputs = {SIDE_INPUT: self.width * (2 if self.side_scale else 1)}

    def __call__(self, critic_inputs: torch.Tensor, actions: torch.Tensor, side) -> torch.Tensor:
        base = side[SIDE_INPUT]
        w = self.width
        if self.side_scale:
            return base[..., :w] + base[..., w : 2 * w] * actions
        return base + self.scale.to(actions.device) * actions


class AdditiveResidual(CombineWrapper):
    neutral = 0.0

    def __init__(self, env: Any, cfg: Any) -> None:
        super().__init__(env, cfg)
        ac = cfg.additive_residual
        policy = self.base_policy.policy
        dims = native_dims(self)
        self.table = None
        self.clip = None
        if ac.scale is not None:
            if len(ac.scale) != self.action_dim:
                raise ValueError(f"additive_residual.scale has {len(ac.scale)} numbers; the action is {self.action_dim} wide")
            self.scale = torch.tensor([float(v) for v in ac.scale], device=self.device)
        else:
            if dims is None:
                raise ValueError("additive_residual: below it the action is no longer the base's own (target tracking): "
                                 "give additive_residual.scale, one number per dim")
            norm = policy.normalizer
            half = (norm.action_max - norm.action_min) / 2
            center = (norm.action_max + norm.action_min) / 2
            if ac.half_range_floor is not None:
                from my_project.wrappers.residual_policy.base_policy import base_run_dir

                stats = demo_action_range(base_run_dir(cfg))
                half = torch.clamp((stats["action_max"] - stats["action_min"]) / 2, min=float(ac.half_range_floor)).to(self.device)
                center = ((stats["action_max"] + stats["action_min"]) / 2).to(self.device)
            index = torch.tensor(dims, dtype=torch.long, device=self.device)
            self.scale = (float(ac.scale_fraction) * half).to(self.device).index_select(0, index)
            if ac.clip is not None:
                c, h = center.to(self.device).index_select(0, index), half.to(self.device).index_select(0, index)
                self.clip = (c - float(ac.clip) * h, c + float(ac.clip) * h)
            if ac.phase_stats is not None:
                table = phase_scale_table(ac.phase_stats, self.phase_names, self.routes, policy.action_dim, self.device)
                self.table = table.index_select(1, index)
        print(f"[additive_residual] sent = base + scale * r over the {self.action_dim}-wide action"
              + (f", per routed phase its scale from {ac.phase_stats}" if self.table is not None else f", scale {[round(v, 5) for v in self.scale.tolist()]}")
              + (f", clamped to the base's +-{ac.clip} x half range" if self.clip is not None else ""), flush=True)

    def uses_phase_rows(self) -> bool:
        return self.table is not None

    def side_of(self, base, phase):
        if self.table is None:
            return base
        return torch.cat((base, self.table[phase]), dim=-1)

    def _scale_of(self, side: torch.Tensor) -> torch.Tensor:
        return side[..., self.action_dim :] if self.table is not None else self.scale

    def combine(self, r, side):
        sent = side[..., : self.action_dim] + self._scale_of(side) * r
        if self.clip is not None:
            sent = torch.maximum(torch.minimum(sent, self.clip[1]), self.clip[0])
        return sent

    def invert(self, executed, side):
        """``(executed - base) / scale``, never clamped (the critics take it as it is); 0 where a dim's scale is 0."""
        scale = self._scale_of(side)
        on = scale > 0
        return torch.where(on, (executed - side[..., : self.action_dim]) / torch.where(on, scale, torch.ones_like(scale)),
                           torch.zeros_like(executed))

    def critic_action(self) -> AdditiveCriticAction:
        return AdditiveCriticAction(self.action_dim, None if self.table is not None else self.scale, self.table is not None)

    def noise_factor(self):
        """The base action in r units, ``|base| / scale``, from agent observations (the base action sits right after the
        observation columns; the history and the wrench, appended later, come after it)."""
        if self.table is not None:
            raise NotImplementedError("the adaptive noise factor reads one scale per dim; phase scales change it per env")
        scale, start, stop = self.scale, self.obs_width, self.obs_width + self.action_dim
        on = scale > 0
        safe = torch.where(on, scale, torch.ones_like(scale))

        def factor(observations: torch.Tensor) -> torch.Tensor:
            base = observations[..., start:stop]
            return torch.where(on, base.abs() / safe, torch.zeros_like(base))

        return factor
