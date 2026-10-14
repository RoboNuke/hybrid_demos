"""The `model` sections of the diffusion policy: ``model.architecture: diffusion_transformer``
or ``diffusion_unet``.

The diffusion policy (Chi et al., "Diffusion Policy", RSS 2023,
https://arxiv.org/abs/2303.04137) as robust-rearrangement runs it
(Ankile et al., https://github.com/ankile/robust-rearrangement): a noise-prediction
transformer plus the diffusion process and the action-chunking rules around it. The defaults
are that repo's ``config/actor/diffusion.yaml``, ``diffusion_model/transformer.yaml`` /
``diffusion_model/unet.yaml`` and ``config/base.yaml`` horizons. Every architecture is
registered in the entry scripts' ``setup()`` by :func:`register_all`; the diffusion process,
the horizons and the inference rules are shared (:class:`DiffusionActorCfg`), the network
fields are each architecture's own.

**Everything the policy and the trainer both read lives here** (horizons, the noise
schedule); the `bc` section holds only what the training loop alone uses, and reads these.

The network is a plain single-agent module for this project's own learner, not the
package's SAC / PPO pipeline (no ``VmapEnsemble``, no ``MODEL_BUILDERS`` entry). There is
no ``critic`` group. The action and observation widths come from the dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

__all__ = [
    "DiffusionActorCfg",
    "DiffusionTransformerActorCfg",
    "DiffusionTransformerModelCfg",
    "DiffusionUnetActorCfg",
    "DiffusionUnetModelCfg",
    "DIFFUSION_ARCHITECTURES",
    "register_all",
    "actor_cfg_from",
]

#: diffusers beta schedules the DDPM / DDIM schedulers accept
BETA_SCHEDULES = ("linear", "scaled_linear", "squaredcos_cap_v2")
#: what the network predicts: the added noise, the clean sample, or the v-parameterization
PREDICTION_TYPES = ("epsilon", "sample", "v_prediction")


@dataclass
class DiffusionActorCfg:
    """What every diffusion-policy architecture shares: horizons, the diffusion process,
    inference. An architecture's actor cfg adds its network's fields."""

    # ---- horizons (the dataset windows and the policy's queues both follow these)
    #: action steps predicted per denoising pass (robust-rearrangement's ``pred_horizon``)
    horizon: int = 32
    #: observation steps the prediction is conditioned on (``obs_horizon``)
    n_obs_steps: int = 1
    #: predicted actions executed before predicting again (``action_horizon``)
    action_horizon: int = 8

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
class DiffusionTransformerActorCfg(DiffusionActorCfg):
    """The noise-prediction transformer (field names follow DiffusionTransformer.__init__)."""

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


@dataclass
class DiffusionUnetActorCfg(DiffusionActorCfg):
    """The noise-prediction 1-D temporal U-Net (``my_project/models/unet.py``; field names
    follow ConditionalUnet1D.__init__). Defaults: robust-rearrangement's
    ``actor/diffusion_model/unet.yaml``, ResiP's state model (paper Table IV)."""

    #: channels per level; each level but the last halves the sequence
    down_dims: List[int] = field(default_factory=lambda: [256, 512, 1024])
    diffusion_step_embed_dim: int = 256
    kernel_size: int = 5
    #: GroupNorm groups (every down_dims entry must be divisible by it)
    n_groups: int = 8


def _validate_diffusion(actor: DiffusionActorCfg) -> None:
    """The checks every architecture shares."""
    for name in ("horizon", "n_obs_steps", "action_horizon", "num_diffusion_iters", "inference_steps"):
        if getattr(actor, name) < 1:
            raise ValueError(f"model.actor.{name} must be >= 1, got {getattr(actor, name)}")
    if actor.action_horizon > actor.horizon:
        raise ValueError(
            f"model.actor.action_horizon ({actor.action_horizon}) cannot exceed "
            f"model.actor.horizon ({actor.horizon}): it executes predicted actions"
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


@dataclass
class DiffusionTransformerModelCfg:
    """The `model` section when ``architecture`` is ``diffusion_transformer``."""

    architecture: str = "diffusion_transformer"
    actor: DiffusionTransformerActorCfg = field(default_factory=DiffusionTransformerActorCfg)

    def validate(self, cfg: Any) -> None:
        actor = self.actor
        _validate_diffusion(actor)
        for name in ("n_layer", "n_head", "n_emb"):
            if getattr(actor, name) < 1:
                raise ValueError(f"model.actor.{name} must be >= 1, got {getattr(actor, name)}")
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


@dataclass
class DiffusionUnetModelCfg:
    """The `model` section when ``architecture`` is ``diffusion_unet``."""

    architecture: str = "diffusion_unet"
    actor: DiffusionUnetActorCfg = field(default_factory=DiffusionUnetActorCfg)

    def validate(self, cfg: Any) -> None:
        actor = self.actor
        _validate_diffusion(actor)
        dims = list(actor.down_dims)
        if not dims or min(dims) < 1:
            raise ValueError(f"model.actor.down_dims must be a non-empty list of positive ints, got {dims}")
        if actor.n_groups < 1 or any(d % actor.n_groups for d in dims):
            raise ValueError(f"model.actor.down_dims {dims} must each be divisible by model.actor.n_groups ({actor.n_groups})")
        if actor.horizon % (2 ** (len(dims) - 1)):
            raise ValueError(
                f"model.actor.horizon ({actor.horizon}) must be divisible by 2 ** (len(down_dims) - 1) = "
                f"{2 ** (len(dims) - 1)}: each level but the last halves the sequence"
            )
        if actor.kernel_size < 1 or actor.kernel_size % 2 == 0:
            raise ValueError(f"model.actor.kernel_size must be odd and >= 1, got {actor.kernel_size}")
        if actor.diffusion_step_embed_dim % 2 or actor.diffusion_step_embed_dim < 4:
            raise ValueError(
                f"model.actor.diffusion_step_embed_dim must be even and >= 4, got {actor.diffusion_step_embed_dim}"
            )


#: every diffusion-policy architecture: name -> (model section, actor cfg). The BC trainer,
#: eval, the residual and the analysis scripts accept any of them; register each with
#: ``register_architecture(name, model_cfg)`` in the scripts' ``setup()`` (``register_all``).
DIFFUSION_ARCHITECTURES = {
    "diffusion_transformer": (DiffusionTransformerModelCfg, DiffusionTransformerActorCfg),
    "diffusion_unet": (DiffusionUnetModelCfg, DiffusionUnetActorCfg),
}


def register_all() -> None:
    """``register_architecture`` every diffusion architecture (the scripts' ``setup()``)."""
    from robonuke_rl_core.models.cfg import register_architecture

    for name, (model_cfg, _) in DIFFUSION_ARCHITECTURES.items():
        register_architecture(name, model_cfg)


def actor_cfg_from(model: Any) -> DiffusionActorCfg:
    """A run's ``model`` section (a resolved config node or the loaded section) -> its typed
    actor cfg, by ``model.architecture``."""
    from omegaconf import OmegaConf

    arch = model.architecture
    if arch not in DIFFUSION_ARCHITECTURES:
        raise ValueError(f"model.architecture {arch!r} is not a diffusion policy; known: {sorted(DIFFUSION_ARCHITECTURES)}")
    actor_cls = DIFFUSION_ARCHITECTURES[arch][1]
    if isinstance(model.actor, actor_cls):
        return model.actor
    return OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(actor_cls), model.actor))
