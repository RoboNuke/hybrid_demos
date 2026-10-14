"""The 1-D U-Net diffusion architecture: the network, its config section, and the policy
running on it exactly as it runs on the transformer."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from my_project.models.cfg import (
    DIFFUSION_ARCHITECTURES,
    DiffusionTransformerActorCfg,
    DiffusionUnetActorCfg,
    DiffusionUnetModelCfg,
    actor_cfg_from,
)
from my_project.models.diffusion_policy import ChunkedBase, DiffusionPolicy, build_noise_net
from my_project.models.transformer import DiffusionTransformer
from my_project.models.unet import ConditionalUnet1D

OBS_DIM, ACT_DIM, B = 7, 4, 3


def small(**overrides) -> DiffusionUnetActorCfg:
    cfg = DiffusionUnetActorCfg(
        horizon=8, n_obs_steps=2, action_horizon=3, down_dims=[16, 32], diffusion_step_embed_dim=16,
        n_groups=4, inference_steps=4, num_diffusion_iters=20, warm_start=False, warm_start_timestep=10,
    )
    return dataclasses.replace(cfg, **overrides)


def test_the_unet_maps_a_sequence_to_one_of_the_same_shape_and_uses_the_conditioning():
    torch.manual_seed(0)
    net = ConditionalUnet1D(ACT_DIM, global_cond_dim=2 * OBS_DIM, diffusion_step_embed_dim=16, down_dims=[16, 32], n_groups=4)
    x = torch.randn(B, 8, ACT_DIM)
    cond = torch.randn(B, 2, OBS_DIM)  # flattened inside
    out = net(x, torch.tensor([3, 5, 7]), global_cond=cond)
    assert out.shape == x.shape
    assert not torch.allclose(out, net(x, torch.tensor([3, 5, 7]), global_cond=cond + 1.0))  # FiLM on the obs
    assert not torch.allclose(out, net(x, 9, global_cond=cond))  # and on the timestep (an int works)
    out.sum().backward()
    assert all(p.grad is not None for p in net.parameters())


def test_resips_widths_give_resips_parameter_count():
    """Paper Table IV: U-Net [256, 512, 1024], 66M parameters (their 10-d action, ~50-d obs)."""
    net = ConditionalUnet1D(10, global_cond_dim=61)
    count = sum(p.numel() for p in net.parameters())
    assert 60e6 < count < 72e6


def test_the_policy_builds_the_network_its_actor_cfg_names():
    assert isinstance(build_noise_net(small(), OBS_DIM, ACT_DIM), ConditionalUnet1D)
    tcfg = DiffusionTransformerActorCfg(horizon=8, n_layer=1, n_head=2, n_emb=16)
    assert isinstance(build_noise_net(tcfg, OBS_DIM, ACT_DIM), DiffusionTransformer)
    with pytest.raises(TypeError):
        build_noise_net(object(), OBS_DIM, ACT_DIM)


def test_the_policy_trains_and_acts_on_the_unet():
    torch.manual_seed(0)
    policy = DiffusionPolicy(small(), OBS_DIM, ACT_DIM)
    policy.normalizer.fit("obs", torch.randn(50, OBS_DIM))
    policy.normalizer.fit("action", torch.randn(50, ACT_DIM))
    loss, parts = policy.train().compute_loss(torch.randn(B, 2, OBS_DIM), torch.randn(B, 8, ACT_DIM))
    loss.backward()
    assert loss.item() > 0 and "bc_loss" in parts
    policy.eval()
    a = policy.act(torch.randn(B, OBS_DIM))
    assert a.shape == (B, ACT_DIM)
    chunk = policy.predict_chunk(torch.randn(B, 2, OBS_DIM))
    assert chunk.shape == (B, 8, ACT_DIM)
    base = ChunkedBase(policy, 2)
    assert base.actions(torch.randn(2, OBS_DIM), torch.tensor([True, True])).shape == (2, ACT_DIM)


def test_the_section_validates_its_own_fields():
    DiffusionUnetModelCfg().validate(None)
    for actor, match in (
        (small(down_dims=[]), "down_dims"),
        (small(down_dims=[16, 30]), "divisible"),
        (small(horizon=6, down_dims=[16, 32, 64], action_horizon=3), "horizon"),
        (small(kernel_size=4), "kernel_size"),
        (small(diffusion_step_embed_dim=7), "diffusion_step_embed_dim"),
        (small(action_horizon=9), "action_horizon"),  # the shared checks run too
    ):
        with pytest.raises(ValueError, match=match):
            DiffusionUnetModelCfg(actor=actor).validate(None)


def test_both_architectures_are_known_and_a_run_section_maps_to_its_actor_cfg():
    from omegaconf import OmegaConf

    assert set(DIFFUSION_ARCHITECTURES) == {"diffusion_transformer", "diffusion_unet"}
    node = OmegaConf.create({"architecture": "diffusion_unet", "actor": {"down_dims": [8, 16], "horizon": 16}})
    actor = actor_cfg_from(node)
    assert isinstance(actor, DiffusionUnetActorCfg) and actor.down_dims == [8, 16] and actor.kernel_size == 5
    with pytest.raises(ValueError, match="not a diffusion policy"):
        actor_cfg_from(OmegaConf.create({"architecture": "simba", "actor": {}}))
