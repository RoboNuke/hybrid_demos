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

    from my_project.models.cfg import DiffusionTransformerActorCfg
    from my_project.models.diffusion_policy import DiffusionPolicy

    run_dir = Path(run_dir).expanduser()
    path = run_dir / "checkpoints" / checkpoint
    if not path.is_file():
        available = sorted(p.name for p in (run_dir / "checkpoints").glob("*.pt"))
        raise FileNotFoundError(f"no base checkpoint {path}; {run_dir / 'checkpoints'} has {available}")
    resolved = OmegaConf.load(resolved_config_path(run_dir))
    if resolved.model.architecture != "diffusion_transformer":
        raise ValueError(f"{run_dir} is not a diffusion policy run (model.architecture={resolved.model.architecture})")
    actor_cfg = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(DiffusionTransformerActorCfg), resolved.model.actor))
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
        from my_project.models.diffusion_policy import ChunkedBase
        from my_project.wrappers import _rebuild_spaces

        unwrapped = env.unwrapped
        rc = cfg.residual
        policy = load_base_policy(rc.base_run, rc.base_checkpoint, unwrapped.device)
        obs_width = int(unwrapped.single_observation_space["policy"].shape[-1])
        #: the env's action past the pose (``hybrid``'s selection and force blocks)
        self.extra_dims = cfg.hybrid.extra_dims
        act_width = int(unwrapped.single_action_space.shape[-1]) - self.extra_dims
        if (obs_width, act_width) != (policy.obs_dim, policy.action_dim):
            raise ValueError(
                f"the base policy ({rc.base_run}) was trained on obs {policy.obs_dim} / action "
                f"{policy.action_dim}; this env has obs {obs_width} / action {act_width} "
                f"(+{self.extra_dims} hybrid dims)"
            )
        self.base = ChunkedBase(policy, unwrapped.num_envs)
        self.normalizer = policy.normalizer
        self.scale = residual_scale(policy, rc.action_scale, rc.action_scale_fraction)
        self.base_action = torch.zeros((unwrapped.num_envs, policy.action_dim), device=unwrapped.device)

        # the agent's observation and critic state each gain the normalized base action
        unwrapped.cfg.observation_space += policy.action_dim
        if getattr(unwrapped.cfg, "state_space", 0):
            unwrapped.cfg.state_space += policy.action_dim
        _rebuild_spaces(unwrapped)
        self.observation_space = unwrapped.observation_space
        self.action_space = unwrapped.action_space
        self._original_get_observations = unwrapped._get_observations
        unwrapped._get_observations = self._get_observations
        print(f"[residual] base {rc.base_run}/{rc.base_checkpoint}; scale {self.scale.tolist()}", flush=True)

    def _get_observations(self):
        observations = self._original_get_observations()
        unwrapped = self.env.unwrapped
        obs = observations["policy"]
        fresh = unwrapped.episode_length_buf == 0
        self.base_action = self.base.actions(obs, fresh)
        observations["policy"], base_n = residual_observation(self.normalizer, obs, self.base_action)
        if "critic" in observations:
            observations["critic"] = torch.cat((observations["critic"], base_n), dim=-1)
        return observations

    def step(self, action):
        action = action.to(self.scale.device)
        width = self.base_action.shape[-1]
        correction = self.scale * action[:, :width]
        sent = self.base_action + correction
        if self.extra_dims:
            sent = torch.cat((sent, action[:, width:]), dim=-1)
        observations, rewards, terminated, truncated, infos = self.env.step(sent)
        if isinstance(infos, dict):
            metrics = infos.setdefault("metrics_to_log", {})
            metrics["residual/position_mm"] = correction[:, 0:3].norm(dim=-1) * 1000.0
            metrics["residual/yaw_deg"] = correction[:, 3].abs() * (180.0 / torch.pi)
        return observations, rewards, terminated, truncated, infos
