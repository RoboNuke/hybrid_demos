"""DiffusionTransformer: a plain single-agent module for the project's own learner."""

from __future__ import annotations

import warnings

import pytest
import torch

from my_project.models.transformer import DiffusionTransformer

ACT_DIM, OBS_DIM = 7, 11
HORIZON, OBS_STEPS = 8, 2
ROWS = 5

#: (causal_attn, n_cond_layers): MLP vs transformer condition encoder, with and without masks
VARIANTS = [(True, 0), (False, 0), (True, 2), (False, 2)]
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)]


def build(causal_attn: bool = True, n_cond_layers: int = 0, p_drop: float = 0.3, device="cpu"):
    return DiffusionTransformer(
        input_dim=ACT_DIM,
        output_dim=ACT_DIM,
        horizon=HORIZON,
        n_obs_steps=OBS_STEPS,
        cond_dim=OBS_DIM,
        n_layer=2,
        n_head=4,
        n_emb=32,
        p_drop_emb=0.0,
        p_drop_attn=p_drop,
        causal_attn=causal_attn,
        n_cond_layers=n_cond_layers,
    ).to(device)


def inputs(device="cpu"):
    sample = torch.randn(ROWS, HORIZON, ACT_DIM, device=device)
    timestep = torch.randint(0, 100, (ROWS,), device=device)
    cond = torch.randn(ROWS, OBS_STEPS, OBS_DIM, device=device)
    return sample, timestep, cond


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("causal_attn,n_cond_layers", VARIANTS)
def test_forward_and_backward_run_cleanly(device, causal_attn, n_cond_layers):
    """Training mode with dropout on, on the default attention kernels: no warnings."""
    torch.manual_seed(0)
    model = build(causal_attn, n_cond_layers, device=device).train()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = model(*inputs(device))
        out.square().mean().backward()
    assert out.shape == (ROWS, HORIZON, ACT_DIM)
    assert not [str(w.message) for w in caught]
    for name, param in model.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name


def test_dropout_is_live_in_train_mode_and_off_in_eval():
    torch.manual_seed(0)
    model = build(p_drop=0.3)
    x = inputs()
    model.train()
    assert not torch.equal(model(*x), model(*x))
    model.eval()
    assert torch.equal(model(*x), model(*x))


@pytest.mark.parametrize("n_cond_layers", [0, 2])
def test_causal_attention_does_not_look_ahead(n_cond_layers):
    torch.manual_seed(0)
    model = build(True, n_cond_layers).eval()
    sample, timestep, cond = inputs()
    changed = sample.clone()
    changed[:, 5:] += 10.0
    a, b = model(sample, timestep, cond), model(changed, timestep, cond)
    assert torch.allclose(a[:, :5], b[:, :5], atol=1e-5)
    assert not torch.allclose(a[:, 5:], b[:, 5:])


def test_scalar_and_batched_timesteps_agree():
    torch.manual_seed(0)
    model = build().eval()
    sample, _, cond = inputs()
    batched = model(sample, torch.full((ROWS,), 17), cond)
    assert torch.allclose(model(sample, 17, cond), batched)
    assert torch.allclose(model(sample, torch.tensor(17), cond), batched)


def test_encoder_only_variant_runs_without_conditioning():
    """cond_dim=0 and time_as_cond=False: the BERT-style trunk with the timestep as a token."""
    torch.manual_seed(0)
    model = DiffusionTransformer(
        ACT_DIM, ACT_DIM, HORIZON, n_layer=2, n_head=4, n_emb=32, causal_attn=True,
        time_as_cond=False,
    )
    sample, timestep, _ = inputs()
    assert model.encoder_only
    assert model(sample, timestep).shape == (ROWS, HORIZON, ACT_DIM)


def test_obs_conditioning_needs_time_as_cond():
    with pytest.raises(ValueError, match="time_as_cond"):
        DiffusionTransformer(ACT_DIM, ACT_DIM, HORIZON, cond_dim=OBS_DIM, time_as_cond=False)


@pytest.mark.parametrize("causal_attn,n_cond_layers", VARIANTS)
def test_optim_groups_split_every_parameter_once(causal_attn, n_cond_layers):
    model = build(causal_attn, n_cond_layers)
    decay, no_decay = model.get_optim_groups(weight_decay=0.1)
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    ids = [id(p) for p in decay["params"] + no_decay["params"]]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    decayed = {id(p) for p in decay["params"]}
    for name, param in model.named_parameters():
        if name.endswith("bias") or name in ("pos_emb", "cond_pos_emb") or ".norm" in name:
            assert id(param) not in decayed, name


def test_configure_optimizers_takes_a_step():
    torch.manual_seed(0)
    model = build()
    optimizer = model.configure_optimizers(learning_rate=1e-3)
    before = model.head.weight.detach().clone()
    model(*inputs()).square().mean().backward()
    optimizer.step()
    assert not torch.equal(model.head.weight, before)
    assert model.device == torch.device("cpu") and model.dtype == torch.float32
