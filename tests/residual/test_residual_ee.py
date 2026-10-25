"""residual.action_frame: ee (A) and residual.accumulate (B): the EE-frame correction on an absolute base."""

from __future__ import annotations

import math
from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch

from my_project.models import action_frame as af
from my_project.residual_ee import EEResidual

N = 8


def poses(n, seed=0, max_deg=180.0):
    g = torch.Generator().manual_seed(seed)
    axis = torch.nn.functional.normalize(torch.randn(n, 3, generator=g), dim=-1)
    R = af.axis_angle_to_mat(axis * torch.rand(n, 1, generator=g) * math.radians(max_deg))
    return torch.rand(n, 3, generator=g), R


def base_actions(n, seed=1):
    p, R = poses(n, seed)
    return torch.cat([p, af.mat_to_rot6d(R), torch.sign(torch.randn(n, 1))], dim=-1)


def angle_deg(R1, R2):
    return torch.rad2deg(torch.arccos((((R1.transpose(-1, -2) @ R2).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)))


@pytest.mark.parametrize("accumulate", [False, True])
def test_zero_correction_is_the_base_and_invert_undoes_step(accumulate):
    ee = EEResidual(0.01, 5.0, accumulate, horizon=8)
    base = base_actions(N)
    _, R_ee = poses(N, seed=2)
    o, O = ee.identity(N, "cpu")
    sent, o1, O1 = ee.step(base, R_ee, o, O, torch.zeros(N, 6))
    assert torch.allclose(sent[:, 0:3], base[:, 0:3]) and torch.equal(sent[:, 9], base[:, 9])
    assert torch.allclose(angle_deg(af.rot6d_to_mat(sent[:, 3:9]), af.rot6d_to_mat(base[:, 3:9])), torch.zeros(N), atol=1e-2)
    r = torch.rand(N, 6) * 2 - 1
    o_prev, O_prev = (o1 + 0.003, O1) if accumulate else (o1, O1)  # B: from some earlier offset
    sent, o2, O2 = ee.step(base, R_ee, o_prev, O_prev, r)
    back, o3, O3 = ee.invert(base, sent, R_ee, o_prev, O_prev)
    assert torch.allclose(back, r, atol=1e-4)
    assert torch.allclose(o3, o2, atol=1e-6) and torch.allclose(O3, O2, atol=1e-5)


def test_a_step_moves_along_the_ees_axes_by_the_scale_and_turns_by_degrees():
    ee = EEResidual(0.01, 5.0, False, horizon=8)
    base = base_actions(N)
    _, R_ee = poses(N, seed=3)
    r = torch.zeros(N, 6)
    r[:, 0], r[:, 5] = 1.0, 1.0  # +1 along the EE's x, +1 about the target's z
    sent, _, _ = ee.step(base, R_ee, *ee.identity(N, "cpu"), r)
    assert torch.allclose(sent[:, 0:3] - base[:, 0:3], 0.01 * R_ee[:, :, 0], atol=1e-6)
    assert torch.allclose(angle_deg(af.rot6d_to_mat(base[:, 3:9]), af.rot6d_to_mat(sent[:, 3:9])), torch.full((N,), 5.0), atol=1e-3)


def test_accumulate_adds_the_steps_up_and_the_offset_observation_round_trips():
    ee = EEResidual(0.01, 5.0, True, horizon=8)
    base = base_actions(N)
    _, R_ee = poses(N, seed=4)
    o, O = ee.identity(N, "cpu")
    r = torch.zeros(N, 6)
    r[:, 1] = 0.5
    for _ in range(3):  # three steps of half a unit along the EE's y
        sent, o, O = ee.step(base, R_ee, o, O, r)
    assert torch.allclose(sent[:, 0:3] - base[:, 0:3], 0.015 * R_ee[:, :, 1], atol=1e-6)
    x = ee.offset_obs(R_ee, o, O)
    assert torch.allclose(x[:, 1], torch.full((N,), 0.015 / 0.08), atol=1e-6)  # over a whole chunk's reach
    o2, O2 = ee.from_offset_obs(R_ee, x)
    assert torch.allclose(o2, o, atol=1e-6) and torch.allclose(O2, O, atol=1e-5)


# ------------------------------------------------------------------ the env wrapper
class FakePoseEnv(gym.Env):
    """A manager-based FBAbs env in miniature (obs 61, action 10) whose observation holds a real EE pose."""

    def __init__(self):
        self.single_observation_space = gym.spaces.Dict({"policy": gym.spaces.Box(-1, 1, (61,))})  # no critic group, as the real env
        self.single_action_space = gym.spaces.Box(-1, 1, (10,))
        self.observation_space, self.action_space = self.single_observation_space, self.single_action_space
        self.num_envs, self.device = 2, "cpu"
        self.episode_length_buf = torch.tensor([0, 0])
        p, R = poses(2, seed=5, max_deg=60.0)
        self.obs = torch.cat([p, af.mat_to_rot6d(R), torch.zeros(2, 52)], dim=-1)
        self.R = R
        self.observation_manager = SimpleNamespace(compute=lambda update_history=False: {"policy": self.obs.clone()})
        self.cfg = SimpleNamespace()
        self.sent = None

    def reset(self, seed=None, options=None):
        return self.observation_manager.compute(update_history=True), {}

    def step(self, action):
        self.sent = action
        self.episode_length_buf = self.episode_length_buf + 1
        return self.observation_manager.compute(update_history=True), torch.zeros(2), torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool), {}


def make_wrapper(monkeypatch, accumulate):
    from my_project import residual as residual_mod
    from my_project.models.cfg import DiffusionUnetActorCfg
    from my_project.models.diffusion_policy import DiffusionPolicy

    torch.manual_seed(0)
    actor = DiffusionUnetActorCfg(horizon=8, n_obs_steps=1, action_horizon=4, down_dims=[16, 32], diffusion_step_embed_dim=16,
                                  n_groups=4, inference_steps=2, num_diffusion_iters=10, warm_start=False)
    policy = DiffusionPolicy(actor, 61, 10)
    policy.normalizer.fit("obs", torch.cat([torch.randn(20, 61), FakePoseEnv().obs]))
    policy.normalizer.fit("action", torch.randn(20, 10))
    monkeypatch.setattr(residual_mod, "load_base_policy", lambda *a, **k: policy.eval())
    from my_project.cfg import ResidualCfg

    rc = ResidualCfg(base_run="r", base_checkpoint="c", action_scale=None, action_scale_fraction=0.1, action_frame="ee",
                     ee_pos_scale=0.01, ee_rot_scale_deg=5.0, accumulate=accumulate, critic_action="full", enabled=True)
    cfg = SimpleNamespace(task_name="Isaac-Furniture-OneLeg-FBAbs-ResiP-v0", residual=rc, hybrid=SimpleNamespace(enabled=False))
    return residual_mod, residual_mod.ResidualPolicyWrapper(FakePoseEnv(), cfg), cfg


@pytest.mark.parametrize("accumulate", [False, True])
def test_the_wrapper_takes_a_6_wide_ee_correction_and_the_critic_rebuilds_what_was_sent(monkeypatch, accumulate):
    residual_mod, env, cfg = make_wrapper(monkeypatch, accumulate)
    extra = 10 + (6 if accumulate else 0)
    assert env.unwrapped.single_action_space.shape == (6,)  # what skrl hands the agent
    obs, _ = env.reset()
    # the critic state the wrapper creates: the normalized base observation, the base action, the offset
    assert obs["policy"].shape == (2, 61 + extra) and obs["critic"].shape == (2, 61 + extra)
    transform = residual_mod.residual_critic_action(cfg, env)
    r = torch.tensor([[1.0, 0, 0, 0, 0, 0.5], [0.0, -0.5, 0.2, 0.3, 0, 0]])
    base = env.base_action.clone()
    critic_action = transform(obs["critic"], r)  # built from the critic state before the step
    R_ee = env.R_ee.clone()
    _, _, _, _, info = env.step(r)
    assert env.env.sent.shape == (2, 10) and critic_action.shape == r.shape  # the learner's rule: r's own width
    # the executed correction: base + that offset is exactly what the env got
    o, O = env.ee.from_offset_obs(R_ee, critic_action)
    assert torch.allclose(base[:, 0:3] + o, env.env.sent[:, 0:3], atol=1e-5)
    R_sent = af.rot6d_to_mat(base[:, 3:9]) @ O
    assert torch.allclose(R_sent, af.rot6d_to_mat(env.env.sent[:, 3:9]), atol=1e-4)
    # +1 along the EE's x is 1 cm along it, from the base's target
    assert torch.allclose(env.env.sent[0, 0:3] - base[0, 0:3], 0.01 * env.env.R[0, :, 0], atol=1e-5)
    assert torch.allclose(info["metrics_to_log"]["residual/position_mm"][0], torch.tensor(10.0), atol=1e-3)


def test_the_accumulated_offset_resets_when_the_base_starts_a_new_chunk(monkeypatch):
    _, env, _ = make_wrapper(monkeypatch, True)
    obs, _ = env.reset()
    r = torch.zeros(2, 6)
    r[:, 0] = 1.0
    sizes = []
    for _ in range(8):  # two chunks of action_horizon 4
        base = env.base_action.clone()
        env.step(r)
        sizes.append(round(float((env.env.sent[0, 0:3] - base[0, 0:3]).norm()) * 1000, 3))
    assert sizes == [10.0, 20.0, 30.0, 40.0, 10.0, 20.0, 30.0, 40.0]


def test_the_ee_frame_is_validated():
    from my_project.cfg import ResidualCfg

    with pytest.raises(ValueError, match="action_frame"):
        ResidualCfg(action_frame="world").validate(None)
    with pytest.raises(ValueError, match="accumulate"):
        ResidualCfg(accumulate=True).validate(None)
    cfg = SimpleNamespace(task_name="Isaac-Forge-PegInsert-Direct-v0", hybrid=SimpleNamespace(enabled=False))
    with pytest.raises(ValueError, match="FBAbs"):
        ResidualCfg(enabled=True, base_run="r", action_frame="ee").validate(cfg)


def test_the_rotation_maps_agree_with_the_bc_ones_under_pi():
    from my_project.residual_ee import exp_so3, log_so3

    g = torch.Generator().manual_seed(9)
    axis = torch.nn.functional.normalize(torch.randn(64, 3, generator=g, dtype=torch.float64), dim=-1)
    v = axis * torch.cat([torch.zeros(1), torch.tensor([1e-7, 1e-4]), torch.rand(61, generator=g) * 3.0]).double()[:, None]
    assert torch.allclose(exp_so3(v), af.axis_angle_to_mat(v), atol=1e-10)
    assert torch.allclose(log_so3(exp_so3(v)), v, atol=1e-6)


@pytest.mark.parametrize("accumulate", [False, True])
def test_gradients_through_the_critic_action_stay_finite_even_at_the_identity(monkeypatch, accumulate):
    """The NaN that killed the first A+B run: the actor's action goes through the critics' executed
    correction; a square root at zero (identity rotation, zero offset, zero action) must not poison it."""
    residual_mod, env, cfg = make_wrapper(monkeypatch, accumulate)
    obs, _ = env.reset()  # a fresh chunk: the offset is exactly zero
    transform = residual_mod.residual_critic_action(cfg, env)
    for r0 in (torch.zeros(2, 6), torch.rand(2, 6) * 2 - 1, torch.tensor([[0.0, 0, 0, 1e-9, 0, 0], [1.0, -1, 1, -1, 1, -1]])):
        r = r0.clone().requires_grad_(True)
        out = transform(obs["critic"], r)
        out.pow(2).sum().backward()
        assert torch.isfinite(out).all() and torch.isfinite(r.grad).all(), r0
    ee = env.ee  # the map itself, from the identity offset
    base = env.base_action
    o, O = ee.identity(2, "cpu")
    r = torch.zeros(2, 6, requires_grad=True)
    _, o2, O2 = ee.step(base, env.R_ee, o, O, r)
    ee.offset_obs(env.R_ee, o2, O2).sum().backward()
    assert torch.isfinite(r.grad).all()
