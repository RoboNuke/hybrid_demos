"""Every registration the residual wrappers need, in one call from an entry script's ``setup()``
(:func:`register_residual`): their config sections, and the package hooks that ask the env's residual wrapper --

* ``register_critic_action``: the critics take the sent action, rebuilt from the side input (the residual's
  :meth:`critic_action`);
* ``register_exploration_scale``: DDPG's adaptive noise factor, read from agent observations (:meth:`noise_factor`);
* ``register_neutral_action``: the r that sends the base's own action (0 added, 1 multiplied);
* ``register_demo_loader``: raw demo episodes converted at load (``my_project/wrappers/residual_policy/demos.py``).

Each hook answers None when the run has no residual (``base_policy.enabled`` false).
"""

from __future__ import annotations

from typing import Any

__all__ = ["register_residual", "residual_critic_action", "residual_exploration_scale", "residual_neutral_action"]


def _enabled(cfg: Any) -> bool:
    return bool(getattr(getattr(cfg, "base_policy", None), "enabled", False))


def residual_critic_action(cfg: Any, env: Any):
    """The critics' action: the sent action, from the residual's side input; None without a residual."""
    if not _enabled(cfg):
        return None
    from my_project.wrappers.residual_policy.combine import find_combine

    combine = find_combine(env)
    print(f"[residual] critics take the sent action ({type(combine).__name__}), rebuilt from the side input", flush=True)
    return combine.critic_action()


def residual_exploration_scale(cfg: Any, env: Any):
    """DDPG's adaptive noise factor, ``observations -> (rows, A)``; None without a residual."""
    if not _enabled(cfg):
        return None
    from my_project.wrappers.residual_policy.combine import find_combine

    return find_combine(env).noise_factor()


def residual_neutral_action(cfg: Any, env: Any):
    """The r that sends the base's own action; None without a residual."""
    if not _enabled(cfg):
        return None
    if cfg.multiply_residual.enabled:
        return 1.0
    return 0.0


def register_residual() -> None:
    """Register the residual wrappers' config sections and package hooks (every entry script's ``setup()``)."""
    from robonuke_rl_core.config import register_section
    from robonuke_rl_core.learners.sac import (register_critic_action, register_demo_loader, register_exploration_scale,
                                               register_neutral_action)

    from my_project.cfg import (AdditiveResidualCfg, BasePolicyCfg, DropColumnsCfg, EERelativeCfg, MultiplyResidualCfg,
                                TargetTrackingCfg)
    from my_project.wrappers.residual_policy.demos import residual_demo_loader

    register_section("base_policy", BasePolicyCfg)
    register_section("drop_columns", DropColumnsCfg)
    register_section("target_tracking", TargetTrackingCfg)
    register_section("ee_relative", EERelativeCfg)
    register_section("additive_residual", AdditiveResidualCfg)
    register_section("multiply_residual", MultiplyResidualCfg)
    register_critic_action(residual_critic_action)
    register_exploration_scale(residual_exploration_scale)
    register_neutral_action(residual_neutral_action)
    register_demo_loader(residual_demo_loader)
