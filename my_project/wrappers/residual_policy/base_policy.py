"""The frozen diffusion policy under a residual (``base_policy.*``): :class:`BasePolicy`, innermost of the residual
wrappers.

Every step, when the env produces its observation (an observation hook, before the agent sees it), the base runs on
the env's own observation (:class:`~my_project.models.diffusion_policy.ChunkedBase`: each env its own clock, a new chunk
every ``action_horizon`` steps or when its episode starts) and gives this step's base action. The wrapper keeps the
step's ``base_obs``, ``base_action``, ``fresh`` (the envs whose episode began) and ``new_chunk`` (the envs whose base
re-planned) as fields every wrapper above reads, and lays the observations out the one way they all expect:

* the agent's observation: ``[observation columns, base action]`` -- the base's action last, in the action space of
  the wrapper that produced it (each wrapper above replaces it with its own);
* the critics' observation: the observation columns alone (an env with a critic group of its own keeps it).

It steps the env with the action it is given (the env's own) and logs the size of the correction, that action
against the base's (``residual/position_mm``, ``residual/rot_deg`` or ``residual/yaw_deg``, by the task's action layout,
:func:`~my_project.families.action_layout`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import torch

__all__ = ["BasePolicy", "base_source", "base_run_dir", "load_base_policy", "override_base", "find_wrapper"]


def base_source(cfg: Any) -> str:
    """Where the base BC run comes from, as the logs show it: ``entity/project/run`` on wandb or the local run dir."""
    bc = cfg.base_policy
    if bc.base_run:
        return str(bc.base_run)
    return f"{bc.base_entity or cfg.wandb.entity}/{bc.base_project}/{bc.base_run_name}"


def base_run_dir(cfg: Any) -> Path:
    """The base BC run as a local run directory: ``base_policy.base_run``, or the wandb run downloaded (its
    ``resolved_config.yaml`` and checkpoint) into the package's run cache."""
    bc = cfg.base_policy
    if bc.base_run:
        return Path(bc.base_run).expanduser()
    from robonuke_rl_core.evaluation import fetch_wandb_run

    return fetch_wandb_run(base_source(cfg), bc.base_checkpoint)


def load_base_policy(run_dir: str | Path, checkpoint: str, device: Any):
    """The frozen diffusion policy of a BC run: architecture from the run's resolved config, weights and normalizer
    stats from ``<run_dir>/checkpoints/<checkpoint>``; its training step and best success rate ride along as
    ``policy.checkpoint_step`` / ``policy.checkpoint_success_rate`` (-1 / nan when it has none)."""
    from omegaconf import OmegaConf

    from robonuke_rl_core.evaluation import resolved_config_path

    from my_project.models.cfg import DIFFUSION_ARCHITECTURES, actor_cfg_from
    from my_project.models.diffusion_policy import DiffusionPolicy

    run_dir = Path(run_dir).expanduser()
    path = run_dir / "checkpoints" / checkpoint
    if not path.is_file():
        available = sorted(p.name for p in (run_dir / "checkpoints").glob("*.pt"))
        raise FileNotFoundError(f"no base checkpoint {path}; {run_dir / 'checkpoints'} has {available}")
    resolved = OmegaConf.load(resolved_config_path(run_dir))
    if resolved.model.architecture not in DIFFUSION_ARCHITECTURES:
        raise ValueError(f"{run_dir} is not a diffusion policy run (model.architecture={resolved.model.architecture})")
    actor_cfg = actor_cfg_from(resolved.model)
    if actor_cfg.action_frame != "absolute":
        raise NotImplementedError(f"{run_dir}: model.actor.action_frame {actor_cfg.action_frame!r}: the base must act in the "
                                  "env's own action (action_frame: absolute)")
    state = torch.load(path, map_location=device, weights_only=False)
    policy = DiffusionPolicy(actor_cfg, state["obs_dim"], state["action_dim"]).to(device)
    policy.load_state_dict(state["model_state_dict"])
    policy.checkpoint_step = int(state.get("global_step", -1))
    policy.checkpoint_success_rate = float(state.get("best_success_rate", float("nan")))
    return policy.eval()


def override_base(policy, bc: Any, source: str = "the base"):
    """``base_policy.base_action_horizon`` / ``base_inference_steps`` / ``base_warm_start`` on a loaded base (null keeps
    the base run's own). One call for the env wrapper and the raw demos."""
    if bc.base_action_horizon is not None:
        if int(bc.base_action_horizon) > int(policy.horizon):
            raise ValueError(f"base_policy.base_action_horizon ({bc.base_action_horizon}) exceeds {source}'s horizon "
                             f"({policy.horizon}): it executes predicted actions")
        policy.action_horizon = int(bc.base_action_horizon)
    if getattr(bc, "base_warm_start", None) is not None:
        policy.cfg.warm_start = bool(bc.base_warm_start)
    if bc.base_inference_steps is not None:
        limit = int(policy.cfg.num_diffusion_iters)
        if int(bc.base_inference_steps) > limit:
            raise ValueError(f"base_policy.base_inference_steps ({bc.base_inference_steps}) exceeds {source}'s "
                             f"num_diffusion_iters ({limit})")
        policy.cfg.inference_steps = int(bc.base_inference_steps)
    return policy


def find_wrapper(env: Any, kind: type) -> Any:
    """The first ``kind`` wrapper in ``env``'s wrapper chain (skrl's ``_env`` or gym's ``env``), or None."""
    node, seen = env, set()
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        if isinstance(node, kind):
            return node
        node = getattr(node, "_env", None) or getattr(node, "env", None)
    return None


class BasePolicy(gym.Wrapper):
    """See the module docstring. ``obs_width`` / ``critic_width``: the observation columns of the agent's / the
    critics' observation (the agent's is ``obs_width + action_dim`` wide); ``action_dim`` the env's action."""

    def __init__(self, env: Any, cfg: Any) -> None:
        super().__init__(env)
        from my_project.envhooks import hooks_for
        from my_project.families import action_layout
        from my_project.models.diffusion_policy import ChunkedBase

        hooks = hooks_for(env.unwrapped)
        bc = cfg.base_policy
        self.source = base_source(cfg)
        policy = load_base_policy(base_run_dir(cfg), bc.base_checkpoint, hooks.device)
        self.obs_width, self.action_dim = hooks.obs_width(), hooks.action_width()
        if (self.obs_width, self.action_dim) != (policy.obs_dim, policy.action_dim):
            raise ValueError(f"the base policy ({self.source}) was trained on obs {policy.obs_dim} / action {policy.action_dim}; "
                             f"this env has obs {self.obs_width} / action {self.action_dim}")
        override_base(policy, bc, self.source)
        self.base = ChunkedBase(policy, hooks.num_envs)
        self.policy = policy
        self.layout = action_layout(cfg.task_name)
        self.num_envs, self.device = hooks.num_envs, hooks.device
        #: the env's own critic group (Forge's state_order) is kept; otherwise the critics see the observation columns
        self.own_critic = hooks.has_critic()
        self.critic_width = hooks.critic_width() if self.own_critic else self.obs_width
        if not self.own_critic:
            hooks.create_critic(self.obs_width)
        hooks.widen(self.action_dim, 0)
        self.base_obs = torch.zeros((self.num_envs, self.obs_width), device=self.device)
        self.base_action = torch.zeros((self.num_envs, self.action_dim), device=self.device)
        self.fresh = torch.ones((self.num_envs,), dtype=torch.bool, device=self.device)
        self.new_chunk = torch.ones((self.num_envs,), dtype=torch.bool, device=self.device)
        self._hooks = hooks
        hooks.hook_observations(self._observations)
        print(f"[base_policy] {self.source}/{bc.base_checkpoint} (step {policy.checkpoint_step}, best success rate "
              f"{policy.checkpoint_success_rate:.3f}); executes {policy.action_horizon} of {policy.horizon}, "
              f"{policy.cfg.inference_steps} DDIM steps, warm start {'on' if policy.cfg.warm_start else 'off'}", flush=True)

    def _observations(self, observations):
        obs = observations["policy"]
        fresh = self._hooks.fresh().to(torch.bool).reshape(-1)
        # the envs whose base starts a chunk this step (ChunkedBase.actions re-plans exactly those)
        self.new_chunk = fresh | (self.base.index >= self.policy.action_horizon)
        self.fresh = fresh
        self.base_obs = obs
        self.base_action = self.base.actions(obs, fresh)
        observations["policy"] = torch.cat((obs, self.base_action), dim=-1)
        if not self.own_critic:
            observations["critic"] = obs
        return observations

    def step(self, action):
        base = self.base_action  # env.step's observation hook replaces it with the next step's
        out = self.env.step(action)
        infos = out[4]
        if isinstance(infos, dict):
            self._log_correction(infos.setdefault("metrics_to_log", {}), action, base)
        return out

    def _log_correction(self, metrics, sent: torch.Tensor, base: torch.Tensor) -> None:
        """The sent action against the base's, by the task's action layout: position (mm, an action in controller units
        scaled to metres first), rotation (deg: axis-angle, the angle between two rot6d targets, or a quaternion delta's)
        or yaw (deg)."""
        from my_project.wrappers.pose_math import rot6d_angle_deg

        layout = self.layout
        correction = sent - base
        metrics["residual/position_mm"] = correction[:, layout["pos"]].norm(dim=-1) * (1000.0 * layout.get("pos_m_per_unit", 1.0))
        if "axis_angle" in layout:
            metrics["residual/rot_deg"] = correction[:, layout["axis_angle"]].norm(dim=-1) * (layout.get("rad_per_unit", 1.0) * 180.0 / torch.pi)
        elif "yaw" in layout:
            metrics["residual/yaw_deg"] = correction[:, layout["yaw"]].abs() * (180.0 / torch.pi)
        elif "rot6d" in layout:
            r6 = layout["rot6d"]
            metrics["residual/rot_deg"] = rot6d_angle_deg(base[:, r6], sent[:, r6])
        else:
            q = correction[:, layout["quat_xyzw"]]
            metrics["residual/rot_deg"] = 2.0 * torch.atan2(q[:, 0:3].norm(dim=-1), q[:, 3].abs()) * (180.0 / torch.pi)
