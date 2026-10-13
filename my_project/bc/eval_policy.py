"""The diffusion BC policy for the package's eval and debug (``register_eval_policy``).

The run's own resolved config gives the architecture (``model.actor``), the checkpoint the
weights and normalizer stats. Acting goes through
:class:`~my_project.models.diffusion_policy.ChunkedBase`, so every env keeps its own chunk and
an env whose episode just began (``episode_length_buf == 0``: a round's reset, the viewer's
reset key, an auto-reset) starts fresh without any hook. Diffusion sampling is stochastic
whatever ``eval.deterministic`` says.

``bc.show_waypoints`` draws every env's current chunk as spheres in the scene, so they show in
the debug viewer and in recorded videos alike: the waypoint being executed this step purple,
the rest of the chunk's first ``action_horizon`` (the ones executed before the next prediction)
green, and the remaining ``horizon - action_horizon`` red. Only positions are drawn.
"""

from __future__ import annotations

import math
from typing import Any

import torch

__all__ = ["build_bc_eval_policy", "WaypointMarkers"]

#: sphere colours (RGB): executing now, the rest of the executed part, the planned tail
COLORS = {"current": (0.6, 0.1, 0.9), "executed": (0.1, 0.8, 0.2), "planned": (0.9, 0.1, 0.1)}
RADIUS = 0.0015


class WaypointMarkers:
    """Spheres at each env's chunk waypoints, coloured by their role in the chunk."""

    def __init__(self, unwrapped: Any, runner: Any) -> None:
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        markers = {
            name: sim_utils.SphereCfg(radius=RADIUS, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb))
            for name, rgb in COLORS.items()
        }
        self.markers = VisualizationMarkers(VisualizationMarkersCfg(prim_path="/Visuals/bc_waypoints", markers=markers))
        self.unwrapped = unwrapped
        self.runner = runner
        horizon, executed = runner.policy.horizon, runner.policy.action_horizon
        role = torch.full((horizon,), 2, dtype=torch.long, device=unwrapped.device)
        role[:executed] = 1
        self.role = role

    def update(self) -> None:
        runner = self.runner
        n, horizon = runner.chunk.shape[0], runner.chunk.shape[1]
        positions = runner.chunk[:, :, 0:3] + self.unwrapped.scene.env_origins[:, None, :]
        roles = self.role.expand(n, horizon).clone()
        current = (runner.index - 1).clamp(0, horizon - 1)  # actions() already moved past it
        roles[torch.arange(n, device=roles.device), current] = 0
        self.markers.visualize(translations=positions.reshape(-1, 3), marker_indices=roles.reshape(-1))


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
    markers = WaypointMarkers(unwrapped, runner) if cfg.bc.show_waypoints else None

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
