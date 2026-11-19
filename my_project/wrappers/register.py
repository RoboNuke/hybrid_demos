"""Every registration the residual wrappers need, in one call from an entry script's ``setup()``
(:func:`register_residual`): their config sections, and the package hooks that ask the env's residual wrapper --

* ``register_critic_action``: the critics take the sent action, rebuilt from the side input (the residual's
  :meth:`critic_action`);
* ``register_noise_scale``: DDPG's noise scale per env and dim (``residual_noise``,
  ``my_project/wrappers/residual_policy/noise_scale.py``);
* ``register_neutral_action``: the r that sends the base's own action (0 added, 1 multiplied);
* ``register_action_bound``: the clamp on DDPG's deterministic action (:meth:`action_bound`; ``multiply_residual.r_range_clip_actor``);
* ``register_demo_loader``: raw demo episodes converted at load (``my_project/wrappers/residual_policy/demos.py``).

Each hook answers None when the run has no residual (``base_policy.enabled`` false).
"""

from __future__ import annotations

from typing import Any

__all__ = ["register_residual", "residual_action_bound", "residual_critic_action", "residual_noise_scale",
           "residual_neutral_action"]


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


def residual_noise_scale(cfg: Any, env: Any):
    """DDPG's noise scale, ``(observations, phase) -> (rows, A)`` in r units (``residual_noise``); None without a residual
    or with ``residual_noise.mode: none``."""
    if not _enabled(cfg):
        return None
    from my_project.wrappers.residual_policy.combine import find_combine
    from my_project.wrappers.residual_policy.noise_scale import build_noise_scale

    return build_noise_scale(cfg, find_combine(env))


def residual_action_bound(cfg: Any, env: Any):
    """The clamp on DDPG's deterministic action, ``observations -> (lo, hi)`` (``multiply_residual.r_range_clip_actor``);
    else None."""
    if not _enabled(cfg):
        return None
    from my_project.wrappers.residual_policy.combine import find_combine

    return find_combine(env).action_bound()


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
    from robonuke_rl_core.learners.sac import (register_action_bound, register_critic_action, register_demo_loader,
                                               register_neutral_action, register_noise_scale)

    from my_project.cfg import (AdditiveResidualCfg, BasePolicyCfg, DropColumnsCfg, EERelativeCfg, MultiplyResidualCfg,
                                RelativeActionCfg, ResidualNoiseCfg, TargetTrackingCfg)
    from my_project.wrappers.residual_policy.demos import residual_demo_loader

    register_section("base_policy", BasePolicyCfg)
    register_section("relative_action", RelativeActionCfg)
    register_section("drop_columns", DropColumnsCfg)
    register_section("target_tracking", TargetTrackingCfg)
    register_section("ee_relative", EERelativeCfg)
    register_section("additive_residual", AdditiveResidualCfg)
    register_section("multiply_residual", MultiplyResidualCfg)
    register_section("residual_noise", ResidualNoiseCfg)
    register_critic_action(residual_critic_action)
    register_noise_scale(residual_noise_scale)
    register_neutral_action(residual_neutral_action)
    register_action_bound(residual_action_bound)
    register_demo_loader(residual_demo_loader)
