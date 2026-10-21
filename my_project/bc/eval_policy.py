"""The diffusion BC policy for the package's eval and debug (``register_eval_policy``).

The run's own resolved config gives the architecture (``model.actor``), the checkpoint the
weights and normalizer stats. Acting goes through
:class:`~my_project.models.diffusion_policy.ChunkedBase`, so every env keeps its own chunk and
an env whose episode just began (``episode_length_buf == 0``: a round's reset, the viewer's
reset key, an auto-reset) starts fresh without any hook. Diffusion sampling is stochastic
whatever ``eval.deterministic`` says.

``bc.show_waypoints`` draws every env's current chunk as spheres in the scene, so they show in
the debug viewer and in recorded videos alike: the waypoint being executed this step yellow,
the chunk's first ``action_horizon`` (the ones executed before the next prediction) green while
still to come and grey once done, and the remaining ``horizon - action_horizon`` red (never
executed). Only positions are drawn, ``bc.waypoint_radius`` in size.
"""

from __future__ import annotations

import math
from typing import Any

import torch

__all__ = ["build_bc_eval_policy", "WaypointMarkers", "waypoint_roles", "waypoint_roles_base"]

#: sphere colours (RGB), in marker-index order: executing now, the executed part still to
#: come, the tail that is never executed, the executed part already done
COLORS = {
    "current": (0.95, 0.85, 0.1),
    "upcoming": (0.1, 0.8, 0.2),
    "planned": (0.9, 0.1, 0.1),
    "done": (0.5, 0.5, 0.5),
}
CURRENT, UPCOMING, PLANNED, DONE = range(4)


class WaypointMarkers:
    """Spheres at each env's chunk waypoints, coloured by their role in the chunk."""

    def __init__(self, unwrapped: Any, runner: Any, radius: float) -> None:
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        markers = {
            name: sim_utils.SphereCfg(radius=radius, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb))
            for name, rgb in COLORS.items()
        }
        self.markers = VisualizationMarkers(VisualizationMarkersCfg(prim_path="/Visuals/bc_waypoints", markers=markers))
        self.unwrapped = unwrapped
        self.runner = runner
        horizon, executed = runner.policy.horizon, runner.policy.action_horizon
        self.role = waypoint_roles_base(horizon, executed, unwrapped.device)

    def update(self) -> None:
        runner = self.runner
        positions = runner.chunk[:, :, 0:3] + self.unwrapped.scene.env_origins[:, None, :]
        roles = waypoint_roles(self.role, runner.index)
        self.markers.visualize(translations=positions.reshape(-1, 3), marker_indices=roles.reshape(-1))


def waypoint_roles_base(horizon: int, executed: int, device: Any = None) -> torch.Tensor:
    """``(horizon,)``: the first ``executed`` waypoints UPCOMING, the rest PLANNED."""
    role = torch.full((horizon,), PLANNED, dtype=torch.long, device=device)
    role[:executed] = UPCOMING
    return role


def waypoint_roles(base: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """``(num_envs, horizon)`` marker indices after :meth:`ChunkedBase.actions` moved each env's
    ``index`` past the waypoint it returned: that one CURRENT, the ones before it DONE."""
    n, horizon = index.shape[0], base.shape[0]
    current = (index - 1).clamp(0, horizon - 1)
    steps = torch.arange(horizon, device=base.device).expand(n, horizon)
    roles = torch.where(steps < current.unsqueeze(-1), torch.full_like(steps, DONE), base.expand(n, horizon))
    return torch.where(steps == current.unsqueeze(-1), torch.full_like(steps, CURRENT), roles)


def build_bc_eval_policy(cfg: Any, env: Any, checkpoint: Any, deterministic: bool):
    from my_project.models.diffusion_policy import ChunkedBase, DiffusionPolicy

    unwrapped = env.unwrapped
    state = torch.load(checkpoint, map_location=unwrapped.device, weights_only=False)
    policy = DiffusionPolicy(cfg.model.actor, state["obs_dim"], state["action_dim"]).to(unwrapped.device)
    policy.load_state_dict(state["model_state_dict"])
    obs_width = int(env.observation_space.shape[-1])
    act_width = int(env.action_space.shape[-1])
    if (obs_width, act_width) != (policy.obs_dim, policy.action_dim):
        raise ValueError(
            f"{checkpoint} was trained on obs {policy.obs_dim} / action {policy.action_dim}; this "
            f"env has obs {obs_width} / action {act_width}"
        )
    runner = ChunkedBase(policy, unwrapped.num_envs)
    markers = WaypointMarkers(unwrapped, runner, cfg.bc.waypoint_radius) if cfg.bc.show_waypoints else None

    def act(observations: torch.Tensor, states: Any) -> torch.Tensor:
        action = runner.actions(observations, unwrapped.episode_length_buf == 0)
        if markers is not None:
            markers.update()
        return action

    metadata = {
        "step": int(state.get("global_step", 0)),
        "mean_return": math.nan,  # a BC checkpoint records no return
        "agent_idx": 0,
        "best_success_rate": state.get("best_success_rate"),
    }
    return act, None, metadata
