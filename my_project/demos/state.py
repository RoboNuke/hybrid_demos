"""Snapshot a Forge env at an episode's start, and put an env back into that snapshot.

A seed cannot reproduce an episode: Forge draws observation noise every step (the draws
happen even when the noise is 0), so the random stream at a reset depends on everything that
ran before it. So the recorder saves the whole starting state and replay restores it:

* the scene (robot joints, peg and hole poses and velocities), via ``scene.get_state`` /
  ``scene.reset_to``, env-relative;
* the env's per-episode bookkeeping that the scene does not hold: Factory's finite-difference
  bases, hole estimate and action seeds, and Forge's per-episode draws (smoothing factor, gains,
  thresholds, dead zone, quaternion sign, force threshold) plus its force smoothing. The lists
  are the package's efficient-reset ones, which restore exactly this for a teleport.

One env. Everything is moved to the CPU to be saved.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from robonuke_rl_core.envs.forge.reset import FACTORY_ATTRS, FORCE_SMOOTHING, FORGE_ATTRS

__all__ = ["snapshot", "restore", "ATTRS"]

ATTRS = (*FACTORY_ATTRS, *FORGE_ATTRS, *FORCE_SMOOTHING)


def _map(value: Any, fn) -> Any:
    if torch.is_tensor(value):
        return fn(value)
    if isinstance(value, dict):
        return {key: _map(item, fn) for key, item in value.items()}
    return value


def snapshot(env: Any) -> Dict[str, Any]:
    """The state a just-reset env starts its episode from, on the CPU."""
    unwrapped = env.unwrapped
    missing = [name for name in ATTRS if not torch.is_tensor(getattr(unwrapped, name, None))]
    if missing:
        raise AttributeError(f"{type(unwrapped).__name__} has no tensor(s) {missing}: not a Forge env?")
    return {
        "scene": _map(unwrapped.scene.get_state(is_relative=True), lambda t: t.detach().cpu().clone()),
        "attrs": {name: getattr(unwrapped, name).detach().cpu().clone() for name in ATTRS},
    }


def restore(env: Any, state: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Put a reset env into ``state``; returns the fresh observations (``{"policy", "critic"}``).

    Call right after ``env.reset()``: the reset closed the gripper and settled the scene, and
    this then overwrites the poses and the bookkeeping with the recorded ones.
    """
    unwrapped = env.unwrapped
    device = unwrapped.device
    env_ids = torch.arange(unwrapped.num_envs, device=device)
    unwrapped.scene.reset_to(_map(state["scene"], lambda t: t.to(device)), env_ids=env_ids, is_relative=True)
    # the env's cached kinematics still describe the pre-restore poses
    unwrapped._compute_intermediate_values(dt=unwrapped.physics_dt)
    for name, value in state["attrs"].items():
        target = getattr(unwrapped, name)
        if target.shape != value.shape:
            raise ValueError(f"recorded {name} has shape {tuple(value.shape)}, the env {tuple(target.shape)}")
        target.copy_(value.to(device))
    return unwrapped._get_observations()
