"""``additive_residual.*``: the agent's ``r`` added to the base action, ``sent = base + scale * r`` on every dim of the
action space the wrapper below gives (:class:`AdditiveResidual`).

The scale per dim is ``additive_residual.scale`` -- the same in every phase, or per phase (each env its current phase's,
from the env's tracker) -- or, where the action is still the base's own (only the base policy, dropped dims and the
EE-relative observation below), one of ResFiT's rules on the base's actions:
``scale_fraction x`` the half range per dim (of the base normalizer, or of its demo dataset floored at
``half_range_floor``), or per routed phase its own scale from ``phase_stats`` (phase_v7: each phase's median demo
correction; a phase without a residual gets 0). ``clip`` clamps the sent action to ``center +- clip x half range``.

The critics take the sent action, ``base + scale * r``, rebuilt from the side input (the base action, then each env's
scale with phase scales); the neutral action is 0. ``residual_noise`` scales are in the sent action's units, divided by
the scale into r (``my_project/wrappers/residual_policy/noise_scale.py``).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from my_project.wrappers.residual_policy.combine import SIDE_INPUT, CombineWrapper

__all__ = ["AdditiveResidual", "AdditiveCriticAction", "native_dims", "demo_action_range", "phase_scale_table",
           "fraction_scale", "flat_scale"]


def native_dims(wrapper) -> Optional[list]:
    """The env's action dims the wrapper's action space keeps, in order, when every wrapper below keeps each dim the
    base's own up to a shift (drop columns, the EE-relative observation, the relative action: a dim's change has the
    same scale as the dim); None when one changes what an action is (target tracking)."""
    from my_project.wrappers.action_spaces.drop_columns import DropColumns
    from my_project.wrappers.action_spaces.relative_action import RelativeAction
    from my_project.wrappers.observation_spaces.ee_relative import EERelative

    dims = list(range(wrapper.base_policy.action_dim))
    for node in wrapper.chain()[1:-1]:
        conversion = node.conversion
        if isinstance(conversion, DropColumns):
            dims = [dims[k] for k in conversion.keep_act]
        elif not isinstance(conversion, (EERelative, RelativeAction)):
            return None
    return dims


def relative_action_of(wrapper):
    """The chain's :class:`~my_project.wrappers.action_spaces.relative_action.RelativeAction`, or None."""
    from my_project.wrappers.action_spaces.relative_action import RelativeAction

    found = [node.conversion for node in wrapper.chain()[1:-1] if isinstance(getattr(node, "conversion", None), RelativeAction)]
    return found[0] if found else None


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


def fraction_scale(wrapper, cfg: Any):
    """ResFiT's scale on the wrapper's action dims, ``additive_residual.scale_fraction x`` the half range per dim (of the
    base normalizer, or of its demo dataset floored at ``half_range_floor``), with that range's center and half range:
    ``(scale, center, half)``. Only where the action is still the base's own (:func:`native_dims`)."""
    ac = cfg.additive_residual
    dims = native_dims(wrapper)
    if dims is None:
        raise ValueError("additive_residual: below it the action is no longer the base's own (target tracking): "
                         "give additive_residual.scale, one number per dim")
    device = wrapper.device
    norm = wrapper.base_policy.policy.normalizer
    half = (norm.action_max - norm.action_min) / 2
    center = (norm.action_max + norm.action_min) / 2
    if ac.half_range_floor is not None:
        from my_project.wrappers.residual_policy.base_policy import base_run_dir

        stats = demo_action_range(base_run_dir(cfg))
        half = torch.clamp((stats["action_max"] - stats["action_min"]) / 2, min=float(ac.half_range_floor))
        center = (stats["action_max"] + stats["action_min"]) / 2
    index = torch.tensor(dims, dtype=torch.long, device=device)
    half, center = half.to(device).index_select(0, index), center.to(device).index_select(0, index)
    return float(ac.scale_fraction) * half, center, half


def flat_scale(wrapper, cfg: Any) -> torch.Tensor:
    """The additive residual's one scale for every phase (``additive_residual.scale`` as a list, else
    :func:`fraction_scale`'s); a per-phase scale (a mapping, ``phase_stats``) refused."""
    ac = cfg.additive_residual
    if ac.phase_stats is not None or (ac.scale is not None and hasattr(ac.scale, "keys")):
        raise ValueError("this reads the additive residual's one scale for every phase: additive_residual.scale per phase "
                         "or phase_stats is not supported here")
    if ac.scale is not None:
        if len(ac.scale) != wrapper.action_dim:
            raise ValueError(f"additive_residual.scale has {len(ac.scale)} numbers; the action is {wrapper.action_dim} wide")
        return torch.tensor([float(v) for v in ac.scale], device=wrapper.device)
    return fraction_scale(wrapper, cfg)[0]


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
    noise_kind = "additive"

    def __init__(self, env: Any, cfg: Any) -> None:
        super().__init__(env, cfg)
        ac = cfg.additive_residual
        policy = self.base_policy.policy
        self.table = None
        self.clip = None
        if ac.scale is not None and hasattr(ac.scale, "keys"):
            # per phase: each env its current phase's row (a phase without a residual 0)
            table = torch.zeros((len(self.phase_names), self.action_dim))
            for name, row in ac.scale.items():
                if name not in self.phase_names:
                    raise ValueError(f"additive_residual.scale names phase {name!r}; the learner's are {self.phase_names}")
                if len(row) != self.action_dim:
                    raise ValueError(f"additive_residual.scale.{name} has {len(row)} numbers; the action is {self.action_dim} wide")
                table[self.phase_names.index(name)] = torch.tensor([float(v) for v in row])
            self.table = table.to(self.device)
            self.scale = None
        elif ac.scale is not None:
            if len(ac.scale) != self.action_dim:
                raise ValueError(f"additive_residual.scale has {len(ac.scale)} numbers; the action is {self.action_dim} wide")
            self.scale = torch.tensor([float(v) for v in ac.scale], device=self.device)
        else:
            self.scale, center, half = fraction_scale(self, cfg)
            if ac.clip is not None:
                lo, hi = center - float(ac.clip) * half, center + float(ac.clip) * half
                relative = relative_action_of(self)
                if relative is not None:
                    # the clamp bounds the absolute action the env executes (ResFiT's executed clip), in the relative
                    # action's map back to absolute -- before it keeps the fingers' executed command -- on the dims this
                    # action keeps (the others unclamped)
                    width = self.base_policy.action_dim
                    full_lo = torch.full((width,), -float("inf"), device=self.device)
                    full_hi = torch.full((width,), float("inf"), device=self.device)
                    index = torch.tensor(native_dims(self), dtype=torch.long, device=self.device)
                    full_lo[index], full_hi[index] = lo, hi
                    relative.executed_clip = (full_lo, full_hi)
                else:
                    self.clip = (lo, hi)
            if ac.phase_stats is not None:
                index = torch.tensor(native_dims(self), dtype=torch.long, device=self.device)
                table = phase_scale_table(ac.phase_stats, self.phase_names, self.routes, policy.action_dim, self.device)
                self.table = table.index_select(1, index)
        print(f"[additive_residual] sent = base + scale * r over the {self.action_dim}-wide action"
              + (f", per phase its scale ({ac.phase_stats or 'additive_residual.scale'})" if self.table is not None
                 else f", scale {[round(v, 5) for v in self.scale.tolist()]}")
              + (f", clamped to the base's +-{ac.clip} x half range" if self.clip is not None else "")
              + (f", the executed (absolute) action clamped to the base's +-{ac.clip} x half range"
                 if getattr(relative_action_of(self), "executed_clip", None) is not None else ""), flush=True)

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

    def noise_target(self, r: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        """A demo frame's correction in ``residual_noise``'s units, the sent action's: ``scale * r``, the executed action
        minus the base's (unclamped)."""
        return self._scale_of(side) * r
