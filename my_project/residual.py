"""Residual RL on a frozen diffusion policy: the env runs the base, the agent learns a correction.

:class:`ResidualPolicyWrapper` sits outermost (``register_env_wrapper`` in every script's
``setup``), on top of the world-pose action. Every step:

1. The frozen base (:class:`~my_project.models.diffusion_policy.ChunkedBase`) gives this
   step's base action ``a_base = [x, y, z, yaw]`` from the current observation. Each env keeps
   its own chunk: a new one is predicted when ``action_horizon`` actions are used up or the
   env's episode just began.
2. The agent observes ``[normalize(obs), normalize(a_base)]``: the base's observation and the
   base action, with the **base policy's** min-max normalizer (as robust-rearrangement's
   residual does). The critic state gets ``normalize(a_base)`` appended too.
3. The agent's action ``r`` (``[-1, 1]`` per dim for a squashed policy) becomes
   ``a_base + residual.action_scale * r``, sent on to the world-pose action. With
   ``hybrid.enabled`` the agent's action also holds the controller's selection and force blocks
   after its 4 pose dims; the base has none, so they go to the controller as the agent gave them.

``residual/*`` per-env metrics report the size of the correction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import torch

__all__ = ["load_base_policy", "ResidualPolicyWrapper", "residual_scale", "residual_observation"]


def load_base_policy(run_dir: str | Path, checkpoint: str, device: Any):
    """The frozen diffusion policy of a BC run: architecture from the run's resolved config,
    weights and normalizer stats from ``<run_dir>/checkpoints/<checkpoint>``."""
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
    state = torch.load(path, map_location=device, weights_only=False)
    policy = DiffusionPolicy(actor_cfg, state["obs_dim"], state["action_dim"]).to(device)
    policy.load_state_dict(state["model_state_dict"])
    return policy.eval()


def residual_scale(policy, action_scale, fraction: float = 0.1) -> torch.Tensor:
    """``residual.action_scale``, or ``fraction`` of the base's normalized action range per
    dim (``residual.action_scale_fraction``)."""
    if action_scale is not None:
        return torch.tensor(list(action_scale), dtype=torch.float32, device=policy.device)
    normalizer = policy.normalizer
    return float(fraction) * (normalizer.action_max - normalizer.action_min) / 2


def residual_observation(normalizer, obs: torch.Tensor, base_action: torch.Tensor):
    """The agent's observation from the base's raw observation and its action:
    ``([normalize(obs), normalize(a_base)], normalize(a_base))`` — the second part is what the
    critic state gets appended. One definition for the env wrapper and the offline demo
    conversion (``my_project/residual_demos.py``)."""
    base_n = normalizer.normalize(base_action, "action")
    return torch.cat((normalizer.normalize(obs, "obs"), base_n), dim=-1), base_n


class ResidualPolicyWrapper(gym.Wrapper):
    def __init__(self, env: Any, cfg: Any) -> None:
        super().__init__(env)
        from my_project.envhooks import hooks_for
        from my_project.families import action_layout
        from my_project.models.diffusion_policy import ChunkedBase

        hooks = hooks_for(env.unwrapped)
        rc = cfg.residual
        policy = load_base_policy(rc.base_run, rc.base_checkpoint, hooks.device)
        obs_width = hooks.obs_width()
        #: the env's action past the pose (``hybrid``'s selection and force blocks)
        self.extra_dims = cfg.hybrid.extra_dims
        act_width = hooks.action_width() - self.extra_dims
        if (obs_width, act_width) != (policy.obs_dim, policy.action_dim):
            raise ValueError(
                f"the base policy ({rc.base_run}) was trained on obs {policy.obs_dim} / action "
                f"{policy.action_dim}; this env has obs {obs_width} / action {act_width} "
                f"(+{self.extra_dims} hybrid dims)"
            )
        self.base = ChunkedBase(policy, hooks.num_envs)
        self.normalizer = policy.normalizer
        self.scale = residual_scale(policy, rc.action_scale, rc.action_scale_fraction)
        self.base_action = torch.zeros((hooks.num_envs, policy.action_dim), device=hooks.device)
        self.layout = action_layout(cfg.task_name)

        # the agent's observation and critic state each gain the normalized base action
        hooks.add_obs_dims(policy.action_dim)
        self.observation_space = hooks.observation_space
        self.action_space = hooks.action_space
        self._hooks = hooks
        hooks.hook_observations(self._observations)
        print(f"[residual] base {rc.base_run}/{rc.base_checkpoint}; scale {self.scale.tolist()}", flush=True)

    def _observations(self, observations):
        obs = observations["policy"]
        self.base_action = self.base.actions(obs, self._hooks.fresh())
        observations["policy"], base_n = residual_observation(self.normalizer, obs, self.base_action)
        if "critic" in observations:
            observations["critic"] = torch.cat((observations["critic"], base_n), dim=-1)
        return observations

    def step(self, action):
        action = action.to(self.scale.device)
        width = self.base_action.shape[-1]
        correction = self.scale * action[:, :width]
        base = self.base_action  # env.step's observation hook replaces it with the next step's
        sent = base + correction
        if self.extra_dims:
            sent = torch.cat((sent, action[:, width:]), dim=-1)
        observations, rewards, terminated, truncated, infos = self.env.step(sent)
        if isinstance(infos, dict):
            metrics = infos.setdefault("metrics_to_log", {})
            metrics["residual/position_mm"] = correction[:, self.layout["pos"]].norm(dim=-1) * 1000.0
            if "yaw" in self.layout:
                metrics["residual/yaw_deg"] = correction[:, self.layout["yaw"]].abs() * (180.0 / torch.pi)
            elif "rot6d" in self.layout:  # absolute targets: the angle between the base's and the sent rotation
                r6 = self.layout["rot6d"]
                metrics["residual/rot_deg"] = rot6d_angle_deg(base[:, r6], sent[:, r6])
            else:  # a quaternion delta's size: the angle of the correction's rotation part
                q = correction[:, self.layout["quat_xyzw"]]
                metrics["residual/rot_deg"] = 2.0 * torch.atan2(q[:, 0:3].norm(dim=-1), q[:, 3].abs()) * (180.0 / torch.pi)
        return observations, rewards, terminated, truncated, infos


def _rot6d_to_mat(d6: torch.Tensor) -> torch.Tensor:
    """``(N, 6)`` first two rows -> ``(N, 3, 3)`` by Gram-Schmidt (the action term's own decode)."""
    b1 = torch.nn.functional.normalize(d6[:, 0:3], dim=-1)
    a2 = d6[:, 3:6]
    b2 = torch.nn.functional.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1)
    return torch.stack((b1, b2, torch.cross(b1, b2, dim=-1)), dim=-2)


def rot6d_angle_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Angle (deg) between the rotations two ``(N, 6)`` rot6d vectors decode to."""
    R = _rot6d_to_mat(a).transpose(-1, -2) @ _rot6d_to_mat(b)
    cos = ((R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.arccos(cos))
