"""The `model` section for ``model.architecture: diffusion_transformer``.

The diffusion policy (Chi et al., "Diffusion Policy", RSS 2023,
https://arxiv.org/abs/2303.04137) as robust-rearrangement runs it
(Ankile et al., https://github.com/ankile/robust-rearrangement): a noise-prediction
transformer plus the diffusion process and the action-chunking rules around it. The defaults
are that repo's ``config/actor/diffusion.yaml``, ``diffusion_model/transformer.yaml`` and
``config/base.yaml`` horizons. Registered in the entry scripts' ``setup()`` with
``register_architecture("diffusion_transformer", DiffusionTransformerModelCfg)``.

**Everything the policy and the trainer both read lives here** (horizons, the noise
schedule); the `bc` section holds only what the training loop alone uses, and reads these.

The network is a plain single-agent module for this project's own learner, not the
package's SAC / PPO pipeline (no ``VmapEnsemble``, no ``MODEL_BUILDERS`` entry). There is
no ``critic`` group. The action and observation widths come from the dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["DiffusionTransformerActorCfg", "DiffusionTransformerModelCfg"]

#: diffusers beta schedules the DDPM / DDIM schedulers accept
BETA_SCHEDULES = ("linear", "scaled_linear", "squaredcos_cap_v2")
#: what the network predicts: the added noise, the clean sample, or the v-parameterization
PREDICTION_TYPES = ("epsilon", "sample", "v_prediction")


@dataclass
class DiffusionTransformerActorCfg:
    """The noise-prediction network, the diffusion process, and action chunking."""

    # ---- horizons (the dataset windows and the policy's queues both follow these)
    #: action steps predicted per denoising pass (robust-rearrangement's ``pred_horizon``)
    horizon: int = 32
    #: observation steps the prediction is conditioned on (``obs_horizon``)
    n_obs_steps: int = 1
    #: predicted actions executed before predicting again (``action_horizon``)
    action_horizon: int = 8

    # ---- the network (field names follow DiffusionTransformer.__init__)
    n_layer: int = 8
    n_head: int = 4
    n_emb: int = 256
    #: dropout on the token embeddings
    p_drop_emb: float = 0.0
    #: dropout inside the attention / feed-forward blocks
    p_drop_attn: float = 0.3
    causal_attn: bool = True
    #: the diffusion timestep is a conditioning token (True) rather than a trunk token
    time_as_cond: bool = True
    #: > 0: a transformer encoder over the conditioning tokens; 0: an MLP
    n_cond_layers: int = 0

    # ---- the diffusion process
    #: forward-process steps used in training (DDPM) and as DDIM's reference grid
    num_diffusion_iters: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    prediction_type: str = "epsilon"
    #: clip the denoised sample to [-1, 1] at each step (the normalized action range)
    clip_sample: bool = True
    #: DDIM denoising steps at inference
    inference_steps: int = 16

    # ---- inference
    #: start each chunk from the previous chunk's unexecuted tail, noised to
    #: ``warm_start_timestep``, instead of pure noise (robust-rearrangement's behaviour)
    warm_start: bool = True
    warm_start_timestep: int = 50
    #: normalized observations are clipped to [-obs_clip, obs_clip] at inference only
    obs_clip: float = 3.0


@dataclass
class DiffusionTransformerModelCfg:
    """The `model` section when ``architecture`` is ``diffusion_transformer``."""

    architecture: str = "diffusion_transformer"
    actor: DiffusionTransformerActorCfg = field(default_factory=DiffusionTransformerActorCfg)

    def validate(self, cfg: Any) -> None:
        actor = self.actor
        for name in (
            "horizon", "n_obs_steps", "action_horizon", "n_layer", "n_head", "n_emb",
            "num_diffusion_iters", "inference_steps",
        ):
            if getattr(actor, name) < 1:
                raise ValueError(f"model.actor.{name} must be >= 1, got {getattr(actor, name)}")
        if actor.action_horizon > actor.horizon:
            raise ValueError(
                f"model.actor.action_horizon ({actor.action_horizon}) cannot exceed "
                f"model.actor.horizon ({actor.horizon}): it executes predicted actions"
            )
        if actor.n_cond_layers < 0:
            raise ValueError(f"model.actor.n_cond_layers must be >= 0, got {actor.n_cond_layers}")
        if actor.n_emb % actor.n_head:
            raise ValueError(
                f"model.actor.n_emb ({actor.n_emb}) must be divisible by model.actor.n_head "
                f"({actor.n_head})"
            )
        if actor.n_emb % 2 or actor.n_emb < 4:
            raise ValueError(
                f"model.actor.n_emb must be even and >= 4 for the sinusoidal timestep "
                f"embedding, got {actor.n_emb}"
            )
        if not actor.time_as_cond:
            raise ValueError(
                "model.actor.time_as_cond: False puts the timestep in the trunk, which the "
                "transformer only allows without observation conditioning; a policy always "
                "conditions on observations"
            )
        for name in ("p_drop_emb", "p_drop_attn"):
            if not 0.0 <= getattr(actor, name) < 1.0:
                raise ValueError(
                    f"model.actor.{name} must be in [0, 1), got {getattr(actor, name)}"
                )
        if actor.beta_schedule not in BETA_SCHEDULES:
            raise ValueError(
                f"model.actor.beta_schedule must be one of {BETA_SCHEDULES}, got "
                f"{actor.beta_schedule!r}"
            )
        if actor.prediction_type not in PREDICTION_TYPES:
            raise ValueError(
                f"model.actor.prediction_type must be one of {PREDICTION_TYPES}, got "
                f"{actor.prediction_type!r}"
            )
        if actor.inference_steps > actor.num_diffusion_iters:
            raise ValueError(
                f"model.actor.inference_steps ({actor.inference_steps}) cannot exceed "
                f"model.actor.num_diffusion_iters ({actor.num_diffusion_iters})"
            )
        if not 0 <= actor.warm_start_timestep < actor.num_diffusion_iters:
            raise ValueError(
                f"model.actor.warm_start_timestep must be in [0, num_diffusion_iters="
                f"{actor.num_diffusion_iters}), got {actor.warm_start_timestep}"
            )
        if actor.obs_clip <= 0:
            raise ValueError(f"model.actor.obs_clip must be > 0, got {actor.obs_clip}")
