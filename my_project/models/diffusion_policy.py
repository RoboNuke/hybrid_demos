"""The diffusion policy: the noise-prediction transformer plus everything around it.

A port of robust-rearrangement's state-based ``DiffusionPolicy`` / ``Actor``
(``src/behavior/diffusion.py``, ``src/behavior/base.py``): DDPM noising for the training loss,
DDIM denoising at inference, min-max normalization to [-1, 1], and action chunking (predict
``horizon`` actions, execute the first ``action_horizon``, predict again).

Kept from the reference on purpose:

* **Warm start** (``model.actor.warm_start``): each chunk starts from the previous chunk's
  unexecuted tail, noised to ``warm_start_timestep``, rather than from pure noise. The DDIM
  loop still runs every inference step (from ~90 down), as the reference does. The buffer is
  zeros before the first chunk and after :meth:`reset`.
* **Observation clipping** at inference only (``model.actor.obs_clip``): normalized
  observations are clipped to [-obs_clip, obs_clip] when acting, never in training.

Changed from the reference:

* The training target follows ``prediction_type`` (noise, clean sample, or velocity); the
  reference always regresses the noise, which is only right for ``epsilon`` (the default).
* :meth:`predict_chunk` (the sampled-action metric) never warm-starts. The reference's metric
  went through the acting path, so it warm-started from whatever chunk the previous batch
  left behind -- an unrelated sample. Here the metric measures the model alone.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from .transformer import DiffusionTransformer

__all__ = ["MinMaxNormalizer", "DiffusionPolicy", "ChunkedBase"]


class MinMaxNormalizer(nn.Module):
    """Per-dimension min-max scaling to [-1, 1], one (min, max) pair per key.

    A column that never changes gets a range of ``[value - 1, value + 1]``, as in the
    reference, so it maps to 0 instead of dividing by zero. The stats are buffers: they
    travel with the policy's ``state_dict`` and never train.
    """

    KEYS = ("obs", "action")

    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        for key, dim in (("obs", obs_dim), ("action", action_dim)):
            self.register_buffer(f"{key}_min", torch.full((dim,), -1.0))
            self.register_buffer(f"{key}_max", torch.full((dim,), 1.0))

    @torch.no_grad()
    def fit(self, key: str, data: torch.Tensor) -> None:
        """Set ``key``'s stats from ``data`` of shape ``(N, dim)``."""
        low, high = data.min(dim=0).values, data.max(dim=0).values
        constant = high == low
        low, high = low.clone(), high.clone()
        low[constant] -= 1
        high[constant] += 1
        getattr(self, f"{key}_min").copy_(low)
        getattr(self, f"{key}_max").copy_(high)

    def normalize(self, x: torch.Tensor, key: str) -> torch.Tensor:
        low, high = getattr(self, f"{key}_min"), getattr(self, f"{key}_max")
        return 2 * (x - low) / (high - low) - 1

    def denormalize(self, x: torch.Tensor, key: str) -> torch.Tensor:
        low, high = getattr(self, f"{key}_min"), getattr(self, f"{key}_max")
        return (x + 1) / 2 * (high - low) + low


class DiffusionPolicy(nn.Module):
    """``obs_dim`` / ``action_dim`` come from the dataset; ``actor_cfg`` is ``model.actor``."""

    def __init__(self, actor_cfg: Any, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.cfg = actor_cfg
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.horizon = actor_cfg.horizon
        self.n_obs_steps = actor_cfg.n_obs_steps
        self.action_horizon = actor_cfg.action_horizon

        self.model = DiffusionTransformer(
            input_dim=self.action_dim,
            output_dim=self.action_dim,
            horizon=actor_cfg.horizon,
            n_obs_steps=actor_cfg.n_obs_steps,
            cond_dim=self.obs_dim,
            n_layer=actor_cfg.n_layer,
            n_head=actor_cfg.n_head,
            n_emb=actor_cfg.n_emb,
            p_drop_emb=actor_cfg.p_drop_emb,
            p_drop_attn=actor_cfg.p_drop_attn,
            causal_attn=actor_cfg.causal_attn,
            time_as_cond=actor_cfg.time_as_cond,
            n_cond_layers=actor_cfg.n_cond_layers,
        )
        self.normalizer = MinMaxNormalizer(self.obs_dim, self.action_dim)

        schedule = dict(
            num_train_timesteps=actor_cfg.num_diffusion_iters,
            beta_schedule=actor_cfg.beta_schedule,
            clip_sample=actor_cfg.clip_sample,
            prediction_type=actor_cfg.prediction_type,
        )
        self.train_noise_scheduler = DDPMScheduler(**schedule)
        self.inference_noise_scheduler = DDIMScheduler(**schedule)
        self.eta = 0.0  # deterministic DDIM, as in the reference

        # acting state: the observation history, the queued actions, the warm-start buffer
        self._observations: deque = deque(maxlen=self.n_obs_steps)
        self._actions: deque = deque()
        self._prev_naction: Optional[torch.Tensor] = None

    @property
    def device(self) -> torch.device:
        return self.normalizer.obs_min.device

    # ------------------------------------------------------------------ training
    def compute_loss(
        self, nobs: torch.Tensor, naction: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """The denoising loss on a normalized batch.

        ``nobs``: ``(B, n_obs_steps, obs_dim)``; ``naction``: ``(B, horizon, action_dim)``.
        Returns the scalar loss and its detached parts for logging.
        """
        noise = torch.randn(naction.shape, device=naction.device)
        timesteps = torch.randint(
            0,
            self.train_noise_scheduler.config.num_train_timesteps,
            (naction.shape[0],),
            device=naction.device,
        ).long()
        noisy_action = self.train_noise_scheduler.add_noise(naction, noise, timesteps)
        prediction = self.model(noisy_action, timesteps, global_cond=nobs.float())

        kind = self.train_noise_scheduler.config.prediction_type
        if kind == "epsilon":
            target = noise
        elif kind == "sample":
            target = naction
        else:  # v_prediction
            target = self.train_noise_scheduler.get_velocity(naction, noise, timesteps)
        loss = F.mse_loss(prediction, target)
        return loss, {"bc_loss": loss.detach()}

    # ------------------------------------------------------------------ inference
    def _denoise(self, nobs: torch.Tensor, start: Optional[torch.Tensor]) -> torch.Tensor:
        """DDIM from pure noise (``start`` None) or from ``start`` noised to the warm-start step."""
        B = nobs.shape[0]
        shape = (B, self.horizon, self.action_dim)
        noise = torch.randn(shape, device=nobs.device)
        scheduler = self.inference_noise_scheduler
        scheduler.set_timesteps(self.cfg.inference_steps)
        if start is None:
            naction = noise * scheduler.init_noise_sigma
        else:
            t = torch.full((B,), self.cfg.warm_start_timestep, device=nobs.device).long()
            naction = scheduler.add_noise(start, noise, t)
        for k in scheduler.timesteps:
            prediction = self.model(sample=naction, timestep=k.to(nobs.device), global_cond=nobs)
            naction = scheduler.step(
                model_output=prediction, timestep=k, sample=naction, eta=self.eta
            ).prev_sample
        return naction

    @torch.no_grad()
    def predict_chunk(self, nobs: torch.Tensor) -> torch.Tensor:
        """A full predicted chunk in action units, from pure noise: ``(B, horizon, A)``."""
        return self.normalizer.denormalize(self._denoise(nobs, None), "action")

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """One env step: ``obs`` ``(B, obs_dim)`` in raw units -> action ``(B, action_dim)``.

        Keeps the last ``n_obs_steps`` observations (the first one repeated to fill the
        history), and predicts a new chunk only when the queued actions run out.
        """
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(
                f"observation has {obs.shape[-1]} dims, the policy was trained on {self.obs_dim}"
            )
        self._observations.append(obs)
        while len(self._observations) < self.n_obs_steps:
            self._observations.append(obs)
        if not self._actions:
            history = torch.stack(list(self._observations), dim=1)
            nobs = self.normalizer.normalize(history, "obs")
            nobs = nobs.clamp(-self.cfg.obs_clip, self.cfg.obs_clip)  # inference only
            naction = self._sample_for_acting(nobs)
            action = self.normalizer.denormalize(naction, "action")
            for i in range(self.action_horizon):
                self._actions.append(action[:, i, :])
        return self._actions.popleft()

    def _sample_for_acting(self, nobs: torch.Tensor) -> torch.Tensor:
        if not self.cfg.warm_start:
            return self._denoise(nobs, None)
        B = nobs.shape[0]
        if self._prev_naction is None or self._prev_naction.shape[0] != B:
            self._prev_naction = torch.zeros(
                (B, self.horizon, self.action_dim), device=nobs.device
            )
        naction = self._denoise(nobs, self._prev_naction)
        # the unexecuted tail seeds the next chunk; the last action_horizon slots keep their
        # previous values, exactly as the reference leaves them
        keep = self.horizon - self.action_horizon
        self._prev_naction[:, :keep] = naction[:, self.action_horizon :]
        return naction

    def reset(self) -> None:
        """Forget the observation history, the queued actions and the warm-start buffer."""
        self._observations.clear()
        self._actions.clear()
        self._prev_naction = None


class ChunkedBase:
    """A frozen :class:`DiffusionPolicy` acting for ``num_envs`` envs, each on its own clock.

    :meth:`DiffusionPolicy.act` keeps one queue for the whole batch, which is right only while
    every env starts and replans together. Here every env has its own observation history,
    chunk, position in the chunk and warm-start buffer, so an env that resets mid-run replans
    alone. One env's step is exactly what ``act`` does for it: a fresh episode fills the
    history with its first observation and starts from an empty warm-start buffer; a new chunk
    (``horizon`` actions) is predicted when the previous ``action_horizon`` are used up; the
    unexecuted tail seeds the next chunk when ``warm_start`` is on. Envs that need a chunk on
    the same step are denoised in one batch.
    """

    def __init__(self, policy: DiffusionPolicy, num_envs: int) -> None:
        self.policy = policy.eval()
        for param in self.policy.parameters():
            param.requires_grad_(False)
        device = policy.device
        n, h, a = int(num_envs), policy.horizon, policy.action_dim
        self.num_envs = n
        self.chunk = torch.zeros((n, h, a), device=device)
        self.index = torch.full((n,), policy.action_horizon, dtype=torch.long, device=device)
        self.prev_naction = torch.zeros((n, h, a), device=device)
        self.history = torch.zeros((n, policy.n_obs_steps, policy.obs_dim), device=device)
        self._rows = torch.arange(n, device=device)

    @torch.no_grad()
    def actions(self, obs: torch.Tensor, fresh: torch.Tensor) -> torch.Tensor:
        """``obs`` ``(num_envs, obs_dim)`` raw; ``fresh`` marks envs whose episode just began.

        Returns this step's base action per env ``(num_envs, action_dim)``, in action units.
        """
        p = self.policy
        if obs.shape != (self.num_envs, p.obs_dim):
            raise ValueError(f"base policy: expected obs {(self.num_envs, p.obs_dim)}, got {tuple(obs.shape)}")
        fresh = fresh.to(torch.bool).reshape(-1)
        self.history = torch.cat((self.history[:, 1:], obs.unsqueeze(1)), dim=1)
        if bool(fresh.any()):
            self.history[fresh] = obs[fresh].unsqueeze(1).expand(-1, p.n_obs_steps, -1)
            self.prev_naction[fresh] = 0.0
            self.index[fresh] = p.action_horizon
        need = (self.index >= p.action_horizon).nonzero(as_tuple=False).reshape(-1)
        if need.numel():
            nobs = p.normalizer.normalize(self.history[need], "obs").clamp(-p.cfg.obs_clip, p.cfg.obs_clip)
            start = self.prev_naction[need] if p.cfg.warm_start else None
            naction = p._denoise(nobs, start)
            if p.cfg.warm_start:
                keep = p.horizon - p.action_horizon
                self.prev_naction[need, :keep] = naction[:, p.action_horizon :]
            self.chunk[need] = p.normalizer.denormalize(naction, "action")
            self.index[need] = 0
        action = self.chunk[self._rows, self.index]
        self.index += 1
        return action
