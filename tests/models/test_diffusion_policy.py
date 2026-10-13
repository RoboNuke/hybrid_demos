"""DiffusionPolicy: the reference's loss and warm-started inference, chunking, reset, clip."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from my_project.models.cfg import DiffusionTransformerActorCfg
from my_project.models.diffusion_policy import DiffusionPolicy, MinMaxNormalizer

OBS_DIM, ACT_DIM, B = 5, 3, 4


def make(**overrides) -> DiffusionPolicy:
    cfg = DiffusionTransformerActorCfg(
        horizon=8, n_obs_steps=2, action_horizon=3, n_layer=2, n_head=2, n_emb=16,
        inference_steps=4, num_diffusion_iters=20, warm_start_timestep=10,
    )
    torch.manual_seed(0)
    policy = DiffusionPolicy(dataclasses.replace(cfg, **overrides), OBS_DIM, ACT_DIM)
    policy.normalizer.fit("obs", torch.randn(50, OBS_DIM) * 3)
    policy.normalizer.fit("action", torch.randn(50, ACT_DIM))
    return policy.eval()


# ------------------------------------------------------------------ normalizer
def test_normalizer_round_trips_and_maps_constant_columns_to_zero():
    norm = MinMaxNormalizer(2, 1)
    data = torch.tensor([[0.0, 5.0], [2.0, 5.0], [4.0, 5.0]])
    norm.fit("obs", data)
    out = norm.normalize(data, "obs")
    assert torch.allclose(out[:, 0], torch.tensor([-1.0, 0.0, 1.0]))
    assert torch.equal(out[:, 1], torch.zeros(3))
    assert torch.allclose(norm.denormalize(out, "obs"), data)
    assert {"normalizer.obs_min", "normalizer.action_max"} <= set(make().state_dict())


# ------------------------------------------------------------------ the reference, ported
def test_loss_is_the_reference_compute_loss():
    policy = make().train()
    nobs, naction = torch.randn(B, 2, OBS_DIM), torch.randn(B, 8, ACT_DIM)

    torch.manual_seed(7)
    ours, parts = policy.compute_loss(nobs, naction)

    torch.manual_seed(7)
    noise = torch.randn(naction.shape)
    timesteps = torch.randint(0, policy.train_noise_scheduler.config.num_train_timesteps, (B,)).long()
    noisy = policy.train_noise_scheduler.add_noise(naction, noise, timesteps)
    torch.manual_seed(7)  # dropout draws after the reference's two draws above
    torch.randn(naction.shape), torch.randint(0, 20, (B,))
    pred = policy.model(noisy, timesteps, global_cond=nobs.float())
    ref = torch.nn.MSELoss(reduction="none")(pred, noise).mean(dim=[1, 2]).unsqueeze(1).mean()
    assert torch.allclose(ours, ref) and torch.equal(parts["bc_loss"], ours.detach())


def reference_normalized_action(policy, nobs, prev_naction, warmstart_timestep):
    """robust-rearrangement's DiffusionPolicy._normalized_action, minus the policy object."""
    B = nobs.shape[0]
    noise = torch.randn((B, policy.horizon, policy.action_dim))
    policy.inference_noise_scheduler.set_timesteps(policy.cfg.inference_steps)
    naction = policy.inference_noise_scheduler.add_noise(
        prev_naction, noise, torch.full((B,), warmstart_timestep).long()
    )
    for k in policy.inference_noise_scheduler.timesteps:
        noise_pred = policy.model(sample=naction, timestep=k, global_cond=nobs)
        naction = policy.inference_noise_scheduler.step(
            model_output=noise_pred, timestep=k, sample=naction, eta=0.0
        ).prev_sample
    prev_naction[:, : policy.horizon - policy.action_horizon, :] = naction[:, policy.action_horizon :, :]
    return naction


def test_warm_started_acting_is_the_reference_inference_two_chunks_running():
    policy = make()
    obs = [torch.randn(B, OBS_DIM) for _ in range(6)]

    torch.manual_seed(11)
    ours = torch.stack([policy.act(o) for o in obs])  # 6 steps = two chunks of 3

    torch.manual_seed(11)
    prev = torch.zeros(B, 8, ACT_DIM)
    expected = []
    for chunk in range(2):
        history = torch.stack([obs[max(3 * chunk - 1, 0)], obs[3 * chunk]], dim=1)
        nobs = policy.normalizer.normalize(history, "obs").clamp(-3, 3)
        naction = reference_normalized_action(policy, nobs, prev, 10)
        expected.append(policy.normalizer.denormalize(naction, "action")[:, :3].transpose(0, 1))
    assert torch.allclose(ours, torch.cat(expected), atol=1e-5)


# ------------------------------------------------------------------ chunking, reset, switches
class Spy:
    def __init__(self, policy):
        self.calls = []
        self._inner = policy._denoise
        policy._denoise = self

    def __call__(self, nobs, start):
        self.calls.append((nobs.clone(), None if start is None else start.clone()))
        return self._inner(nobs, start)


def test_a_new_chunk_every_action_horizon_steps_and_reset_forgets_everything():
    policy = make()
    spy = Spy(policy)
    for _ in range(7):
        policy.act(torch.randn(B, OBS_DIM))
    assert len(spy.calls) == 3  # steps 0, 3, 6
    assert spy.calls[1][1].abs().sum() > 0  # the second chunk warm-starts from the first's tail
    policy.reset()
    policy.act(torch.randn(B, OBS_DIM))
    assert len(spy.calls) == 4 and torch.equal(spy.calls[3][1], torch.zeros(B, 8, ACT_DIM))


def test_warm_start_off_starts_every_chunk_from_noise():
    policy = make(warm_start=False)
    spy = Spy(policy)
    for _ in range(4):
        policy.act(torch.randn(B, OBS_DIM))
    assert [start for _, start in spy.calls] == [None, None]


@pytest.mark.parametrize("clip", [3.0, 0.5])
def test_observations_are_clipped_when_acting_only(clip):
    policy = make(obs_clip=clip)
    spy = Spy(policy)
    policy.act(torch.full((B, OBS_DIM), 1e3))
    assert spy.calls[0][0].abs().max() == clip
    loss, _ = policy.compute_loss(torch.full((B, 2, OBS_DIM), 50.0), torch.zeros(B, 8, ACT_DIM))
    assert torch.isfinite(loss)  # training sees the unclipped value (no clamp to compare, but it runs)


def test_predict_chunk_never_touches_the_acting_state():
    policy = make()
    spy = Spy(policy)
    chunk = policy.predict_chunk(torch.randn(B, 2, OBS_DIM))
    assert chunk.shape == (B, 8, ACT_DIM)
    assert spy.calls[0][1] is None and policy._prev_naction is None and not policy._actions


def test_a_wrong_observation_width_raises():
    with pytest.raises(ValueError, match="trained on 5"):
        make().act(torch.zeros(B, OBS_DIM + 1))


@pytest.mark.gpu
def test_it_trains_and_acts_on_cuda():
    policy = make().to("cuda").train()
    loss, _ = policy.compute_loss(torch.randn(B, 2, OBS_DIM, device="cuda"), torch.randn(B, 8, ACT_DIM, device="cuda"))
    loss.backward()
    policy.eval()
    assert policy.act(torch.randn(B, OBS_DIM, device="cuda")).device.type == "cuda"


# ------------------------------------------------------------------ the per-env runner
from my_project.models.diffusion_policy import ChunkedBase


def test_chunked_base_in_lockstep_is_act():
    """All envs fresh together and never reset: exactly DiffusionPolicy.act's sequence."""
    obs = [torch.randn(B, OBS_DIM) for _ in range(7)]
    policy = make()
    torch.manual_seed(5)
    expected = torch.stack([policy.act(o) for o in obs])
    runner = ChunkedBase(make(), B)
    torch.manual_seed(5)
    fresh = torch.ones(B, dtype=torch.bool)
    got = []
    for i, o in enumerate(obs):
        got.append(runner.actions(o, fresh if i == 0 else torch.zeros(B, dtype=torch.bool)))
    assert torch.allclose(torch.stack(got), expected, atol=1e-5)


def test_an_env_that_resets_replans_alone():
    runner = ChunkedBase(make(), B)
    calls = []
    inner = runner.policy._denoise
    runner.policy._denoise = lambda nobs, start: calls.append(nobs.shape[0]) or inner(nobs, start)
    none = torch.zeros(B, dtype=torch.bool)
    runner.actions(torch.randn(B, OBS_DIM), torch.ones(B, dtype=torch.bool))  # all plan: B rows
    runner.actions(torch.randn(B, OBS_DIM), none)
    one = none.clone()
    one[1] = True
    runner.actions(torch.randn(B, OBS_DIM), one)  # env 1 restarts: it alone replans
    assert calls == [B, 1]
    assert runner.index.tolist() == [3, 1, 3, 3]
    for _ in range(3):  # envs 0, 2, 3 reach action_horizon=3 first and replan together
        runner.actions(torch.randn(B, OBS_DIM), none)
    assert calls == [B, 1, 3, 1]


def test_chunked_base_is_frozen_and_checks_widths():
    runner = ChunkedBase(make().train(), B)
    assert not runner.policy.training and not any(p.requires_grad for p in runner.policy.parameters())
    with pytest.raises(ValueError, match="expected obs"):
        runner.actions(torch.zeros(B, OBS_DIM + 1), torch.ones(B, dtype=torch.bool))
