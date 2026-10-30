"""``residual.steps_per_chunk`` (the chunk-level residual), the base overrides and the observation
flags (``normalize_observation`` / ``observe_base_action``), on CPU fakes: a scripted base whose chunk
values say which chunk and step each action came from, and a manager-style env that records what it
was sent and ends an episode when told to."""

from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch

from my_project import residual as residual_mod
from my_project.cfg import ResidualCfg
from my_project.envhooks import DirectHooks
from my_project.models.diffusion_policy import MinMaxNormalizer
from my_project.residual import ResidualPolicyWrapper, override_base, residual_observation

OBS, ACT, N = 5, 8, 2
TASK = "Isaac-Furniture-OneLeg-FB-v0"  # action layout [pos 3, quat 4, gripper]: 8 dims, no furniture_lab needed


class IdentityNormalizer:
    action_min = torch.full((ACT,), -1.0)
    action_max = torch.ones(ACT)

    def normalize(self, x, key, widen=False):
        return x

    def denormalize(self, x, key):
        return x


class ScriptedPolicy:
    """Chunk ``c`` (the c-th denoising call) holds ``100 c + i`` in dim 0 of its step ``i``, 0 elsewhere."""

    obs_dim, action_dim, net_action_dim, n_obs_steps = OBS, ACT, ACT, 1
    device = torch.device("cpu")
    normalizer = IdentityNormalizer()

    def __init__(self, horizon=4, action_horizon=4):
        self.horizon, self.action_horizon = horizon, action_horizon
        self.cfg = SimpleNamespace(obs_clip=10.0, warm_start=False, inference_steps=2, num_diffusion_iters=10)
        self.calls = 0

    def eval(self):
        return self

    def parameters(self):
        return []

    def reframe(self, naction, obs_old, obs_new):
        return naction

    def to_env(self, chunk, obs):
        return chunk

    def _denoise(self, nobs, start):
        out = torch.zeros(nobs.shape[0], self.horizon, ACT)
        out[:, :, 0] = torch.arange(self.horizon, dtype=torch.float32) + 100.0 * self.calls
        self.calls += 1
        return out


class ChunkEnv(gym.Env):
    """A manager-based env in miniature: obs ``[t, 1, 2, 3, 4]`` per env, reward ``t`` after its ``t``-th
    sim step, metric ``m = t``; ``end_at[env] = t`` ends that env's episode at its ``t``-th step
    (terminated) and resets it, as Isaac Lab does."""

    def __init__(self, end_at=None, critic=True, moving=None):
        spaces = {"policy": gym.spaces.Box(-1, 1, (OBS,))}
        if critic:
            spaces["critic"] = gym.spaces.Box(-1, 1, (3,))
        self.single_observation_space = gym.spaces.Dict(spaces)
        self.single_action_space = gym.spaces.Box(-1, 1, (ACT,))
        self.observation_space, self.action_space = self.single_observation_space, self.single_action_space
        self.num_envs, self.device = N, "cpu"
        self.episode_length_buf = torch.zeros(N, dtype=torch.long)
        self.end_at = end_at or {}
        self.critic = critic
        #: with ``moving`` (a set of (env, global step)), an eval_state: phase/id = the step's t before any reset, and
        #: phase/still = 1 except at the listed steps
        self.moving = moving
        self.observation_manager = SimpleNamespace(compute=self._compute)
        self.cfg = SimpleNamespace()
        self.sent, self.steps = [], 0

    def _compute(self, update_history=False):
        t = self.episode_length_buf.to(torch.float32)
        obs = {"policy": torch.cat((t[:, None], torch.tensor([[1.0, 2.0, 3.0, 4.0]]).expand(N, -1)), dim=-1)}
        if self.critic:
            obs["critic"] = torch.full((N, 3), float(self.steps))
        return obs

    def reset(self, seed=None, options=None):
        self.episode_length_buf.zero_()
        return self.observation_manager.compute(update_history=True), {}

    def step(self, action):
        self.sent.append(action.clone())
        self.steps += 1
        self.episode_length_buf += 1
        t = self.episode_length_buf.clone()
        terminated = torch.tensor([self.end_at.get(i) == int(t[i]) for i in range(N)])
        self.episode_length_buf[terminated] = 0  # Isaac Lab's auto-reset
        reward = t.to(torch.float32)
        infos = {
            "metrics_to_log": {"m": t.to(torch.float32)},
            "episode_metrics_to_log": {"e": torch.full((N,), float(self.steps))},
        }
        if self.moving is not None:
            infos["eval_state"] = {
                "phase/id": t.to(torch.float32),
                "phase/still": torch.tensor([0.0 if (i, self.steps) in self.moving else 1.0 for i in range(N)]),
            }
        return self.observation_manager.compute(update_history=True), reward, terminated, torch.zeros(N, dtype=torch.bool), infos


def make(monkeypatch, env, **fields):
    policy = ScriptedPolicy(horizon=fields.pop("horizon", 4), action_horizon=fields.pop("action_horizon", 4))
    monkeypatch.setattr(residual_mod, "load_base_policy", lambda *a, **k: policy)
    rc = ResidualCfg(enabled=True, base_run="r", base_checkpoint="c", action_scale=[0.5] * ACT, **fields)
    cfg = SimpleNamespace(task_name=TASK, residual=rc, hybrid=SimpleNamespace(enabled=False, extra_dims=0))
    return ResidualPolicyWrapper(env, cfg), policy


def residual(values):
    """A (N, k x A) agent action with ``values[env][j]`` in dim 0 of sub-step j, zeros elsewhere."""
    k = len(values[0])
    out = torch.zeros(N, k, ACT)
    for i, row in enumerate(values):
        out[i, :, 0] = torch.tensor(row, dtype=torch.float32)
    return out.reshape(N, k * ACT)


# ---------------------------------------------------------------------- the chunk step
def test_the_agent_acts_every_k_steps_with_a_k_by_a_action_on_the_bases_own_actions(monkeypatch):
    env, policy = make(monkeypatch, ChunkEnv(), steps_per_chunk=2)
    assert env.unwrapped.single_action_space.shape == (2 * ACT,) and env.action_space.shape == (N, 2 * ACT)
    obs, _ = env.reset()
    assert obs["policy"].shape == (N, OBS + 2 * ACT) and obs["critic"].shape == (N, 3 + 2 * ACT)
    # the block: chunk 0's steps 0 and 1, this step's base action and the next (dim 0 of each row)
    assert policy.calls == 1 and obs["policy"][:, [OBS, OBS + ACT]].tolist() == [[0.0, 1.0], [0.0, 1.0]]
    obs, reward, terminated, truncated, infos = env.step(residual([[1.0, -1.0], [0.2, 0.4]]))
    sent = env.unwrapped.sent
    assert len(sent) == 2  # two sim steps for one agent step
    # sub-step j: the base's own j-th action plus scale 0.5 x the j-th row
    assert sent[0][:, 0].tolist() == pytest.approx([0.0 + 0.5, 0.0 + 0.1]) and sent[1][:, 0].tolist() == pytest.approx([1.0 - 0.5, 1.0 + 0.2])
    assert torch.all(sent[0][:, 1:] == 0) and torch.all(sent[1][:, 1:] == 0)
    assert reward.tolist() == [1.0 + 2.0, 1.0 + 2.0]  # the two sub-steps' rewards summed
    assert not terminated.any() and not truncated.any()
    assert env.counted.tolist() == [2, 2]
    # metrics are per-sub-step means: m = (1 + 2) / 2; position_mm = mean of |0.5 r_j| x 1000
    assert infos["metrics_to_log"]["m"].tolist() == [1.5, 1.5]
    assert infos["metrics_to_log"]["residual/position_mm"].tolist() == pytest.approx([500.0, 150.0])
    # the next agent step sees the base's 3rd and 4th actions, and the chunk boundary never splits a re-plan
    assert obs["policy"][:, [OBS, OBS + ACT]].tolist() == [[2.0, 3.0], [2.0, 3.0]]
    env.step(residual([[0.0, 0.0], [0.0, 0.0]]))
    assert env.unwrapped.sent[-1][:, 0].tolist() == [3.0, 3.0] and policy.calls == 2  # chunk 1 planned for the step after
    assert env.base_action[:, 0].tolist() == [100.0, 100.0]


def test_an_env_whose_episode_ends_mid_chunk_runs_the_rest_on_the_base_and_lines_up_again(monkeypatch):
    """Env 0 ends at its 3rd sim step, sub-step 0 of the second chunk: only that sub-step counts for
    it (reward, done, metrics), its last sub-step is the base alone, and its base re-plans at the chunk
    boundary so the new episode's chunks line up with the agent's."""
    env, policy = make(monkeypatch, ChunkEnv(end_at={0: 3}), steps_per_chunk=2)
    env.reset()
    env.step(residual([[1.0, 1.0], [1.0, 1.0]]))
    obs, reward, terminated, truncated, infos = env.step(residual([[1.0, 1.0], [1.0, 1.0]]))
    sent = env.unwrapped.sent
    assert terminated.tolist() == [True, False] and env.counted.tolist() == [1, 2]
    assert reward.tolist() == [3.0, 3.0 + 4.0]  # env 0: sub-step 0 only (its new episode's step is not its)
    # env 0 reset at sub-step 0 (fresh: chunk 1 planned there, for it alone), so sub-step 1 sent it that
    # chunk's first action with no residual; env 1 got chunk 0's 4th action plus the residual
    assert sent[-1][0, 0].item() == 100.0 and sent[-1][1, 0].item() == 3.0 + 0.5
    # metrics: env 0's are its counted sub-step's alone; e is from the sub-step it closed on
    assert infos["metrics_to_log"]["m"].tolist() == [3.0, 3.5]
    assert infos["episode_metrics_to_log"]["e"].tolist() == [3.0, 4.0]
    # both bases start a chunk at the boundary: env 1 used its 4, env 0 was made to re-plan; one call
    # plans both (chunk 2)
    assert policy.calls == 3 and env.base.index.tolist() == [1, 1]
    assert obs["policy"][:, [OBS, OBS + ACT]].tolist() == [[200.0, 201.0], [200.0, 201.0]]
    assert obs["policy"][0, 0].item() == 1.0  # env 0's observation: its new episode, one step in


def test_a_chunk_passes_on_the_closing_sub_steps_state_and_is_still_only_if_every_own_sub_step_was(monkeypatch):
    """An env whose episode ends mid-chunk keeps the eval_state of the sub-step it closed on (its phase label, its
    stall flags), not its next episode's; phase/still is 1 only if every sub-step of the chunk's own episode was."""
    # env 0 ends at its 3rd step (sub-step 0 of the second chunk) and its next episode's first step moves; env 1
    # moves at global step 3 (sub-step 0 of the second chunk)
    env, _ = make(monkeypatch, ChunkEnv(end_at={0: 3}, moving={(0, 4), (1, 3)}), steps_per_chunk=2)
    env.reset()
    _, _, _, _, infos = env.step(residual([[0.0, 0.0], [0.0, 0.0]]))
    assert infos["eval_state"]["phase/still"].tolist() == [1.0, 1.0]
    _, _, terminated, _, infos = env.step(residual([[0.0, 0.0], [0.0, 0.0]]))
    assert terminated.tolist() == [True, False]
    assert infos["eval_state"]["phase/id"].tolist() == [3.0, 4.0]  # env 0: its closing step's, not the new episode's 1
    assert infos["eval_state"]["phase/still"].tolist() == [1.0, 0.0]  # env 0's moving step is its next episode's


def test_the_action_width_is_checked_and_k_1_is_the_plain_step(monkeypatch):
    env, _ = make(monkeypatch, ChunkEnv(), steps_per_chunk=2)
    env.reset()
    with pytest.raises(ValueError, match=r"\(2, 2 x 8\)"):
        env.step(torch.zeros(N, ACT))
    env, _ = make(monkeypatch, ChunkEnv())
    env.reset()
    assert env.unwrapped.single_action_space.shape == (ACT,)
    _, reward, _, _, infos = env.step(torch.zeros(N, ACT))
    assert len(env.unwrapped.sent) == 1 and reward.tolist() == [1.0, 1.0] and env.counted.tolist() == [1, 1]
    assert infos["metrics_to_log"]["m"].tolist() == [1.0, 1.0]


# ---------------------------------------------------------------------- the base overrides
def test_the_base_overrides_apply_and_the_executed_horizon_must_be_a_multiple_of_k():
    policy = ScriptedPolicy(horizon=8, action_horizon=8)
    override_base(policy, ResidualCfg(base_action_horizon=4, base_inference_steps=4, steps_per_chunk=4))
    assert policy.action_horizon == 4 and policy.cfg.inference_steps == 4
    override_base(policy, ResidualCfg())  # null keeps what the base has
    assert policy.action_horizon == 4 and policy.cfg.inference_steps == 4
    with pytest.raises(ValueError, match="exceeds .* horizon"):
        override_base(ScriptedPolicy(horizon=8), ResidualCfg(base_action_horizon=9))
    with pytest.raises(ValueError, match="num_diffusion_iters"):
        override_base(ScriptedPolicy(), ResidualCfg(base_inference_steps=11))
    with pytest.raises(ValueError, match="must divide"):
        override_base(ScriptedPolicy(horizon=8, action_horizon=6), ResidualCfg(steps_per_chunk=4))
    override_base(ScriptedPolicy(horizon=8, action_horizon=6), ResidualCfg(steps_per_chunk=4, base_action_horizon=8))


def test_the_wrapper_runs_the_base_as_overridden(monkeypatch):
    env, policy = make(monkeypatch, ChunkEnv(), horizon=8, action_horizon=8, base_action_horizon=2, base_inference_steps=3, steps_per_chunk=2)
    assert policy.action_horizon == 2 and policy.cfg.inference_steps == 3
    env.reset()
    env.step(residual([[0.0, 0.0], [0.0, 0.0]]))
    assert env.unwrapped.sent[-1][:, 0].tolist() == [1.0, 1.0] and env.base_action[:, 0].tolist() == [100.0, 100.0]


def test_the_chunk_observes_the_bases_next_k_actions_and_full_gives_the_critics_what_the_env_got(monkeypatch):
    """At the boundary the base-action block is the next k base actions (normalized, step-major); the
    full critic action rebuilt from that observation and r is exactly the normalized actions the env
    received over the k sub-steps."""
    policy = ScriptedPolicy()
    policy.normalizer = a_normalizer()
    monkeypatch.setattr(residual_mod, "load_base_policy", lambda *a, **k: policy)
    rc = ResidualCfg(enabled=True, base_run="r", base_checkpoint="c", action_scale=[0.5] * ACT, steps_per_chunk=2, critic_action="full")
    cfg = SimpleNamespace(task_name=TASK, residual=rc, hybrid=SimpleNamespace(enabled=False, extra_dims=0))
    env = ResidualPolicyWrapper(ChunkEnv(), cfg)
    assert env.base_action_slice == slice(OBS, OBS + 2 * ACT)
    normalizer = policy.normalizer
    transform = residual_mod.residual_critic_action(cfg, env)
    obs, _ = env.reset()
    assert obs["policy"].shape == (N, OBS + 2 * ACT) and obs["critic"].shape == (N, 3 + 2 * ACT)
    assert torch.equal(obs["critic"][:, 3:], obs["policy"][:, OBS:])  # the env's critic state, then the base chunk
    for chunk in range(3):  # across a base re-plan (action_horizon 4, k 2: chunk 2 is a new base chunk)
        block = obs["policy"][:, OBS:].reshape(N, 2, ACT)
        assert torch.allclose(normalizer.denormalize(block[:, 0], "action"), env.base_action)  # row 0: this step's base action
        r = torch.rand(N, 2 * ACT) * 2 - 1
        critic_action = transform(obs["critic"], r)  # the learner hands the transform the critic state
        assert critic_action.shape == r.shape
        obs, _, _, _, _ = env.step(r)
        sent = env.unwrapped.sent[-2:]
        for j in range(2):
            got = critic_action[:, j * ACT : (j + 1) * ACT]
            assert torch.allclose(got, normalizer.normalize(sent[j], "action"), atol=1e-5), (chunk, j)
            assert torch.allclose(normalizer.denormalize(block[:, j], "action") + 0.5 * r[:, j * ACT : (j + 1) * ACT], sent[j], atol=1e-5)
    # the observation holds the next two base actions, normalized: chunk 0 (0..3) ran in agent steps 1-2, chunk 1
    # (planned at sim step 4) gave 100, 101 to agent step 3, so its 102, 103 come next (the base denormalizes them
    # for the env; normalized again they read as the network planned them)
    assert torch.allclose(obs["policy"][:, OBS:].reshape(N, 2, ACT)[:, :, 0], torch.tensor([[102.0, 103.0]] * N), atol=1e-3)


def test_the_bases_block_is_the_k_aligned_group_of_the_action_just_handed_out():
    from my_project.models.diffusion_policy import ChunkedBase

    base = ChunkedBase(ScriptedPolicy(horizon=8, action_horizon=8), 2)
    obs = torch.zeros(2, OBS)
    base.actions(obs, torch.tensor([True, True]))  # plans; hands out step 0
    assert base.block(4)[:, :, 0].tolist() == [[0.0, 1.0, 2.0, 3.0]] * 2
    base.actions(obs, torch.tensor([False, False]))  # step 1: still group 0
    assert base.block(4)[:, :, 0].tolist() == [[0.0, 1.0, 2.0, 3.0]] * 2
    for _ in range(3):
        base.actions(obs, torch.tensor([False, False]))  # steps 2, 3, 4: group 1 from step 4
    assert base.block(4)[:, :, 0].tolist() == [[4.0, 5.0, 6.0, 7.0]] * 2
    with pytest.raises(ValueError, match="not a multiple of 3"):
        base.block(3)


def test_the_chunk_config_is_validated():
    with pytest.raises(ValueError, match="steps_per_chunk"):
        ResidualCfg(steps_per_chunk=0).validate(None)
    with pytest.raises(ValueError, match="base_action_horizon"):
        ResidualCfg(base_action_horizon=0).validate(None)
    with pytest.raises(ValueError, match="multiple of"):
        ResidualCfg(steps_per_chunk=4, base_action_horizon=6).validate(None)
    ResidualCfg(steps_per_chunk=2, critic_action="full").validate(None)
    with pytest.raises(NotImplementedError, match="action_frame: ee"):
        ResidualCfg(steps_per_chunk=2, critic_action="full", action_frame="ee").validate(None)
    # the critics read the base action from their own state: full is fine whatever the agent observes
    ResidualCfg(critic_action="full", observe_base_action=False).validate(None)
    ResidualCfg(steps_per_chunk=4, base_action_horizon=4, base_inference_steps=4, observe_base_action=False, normalize_observation=False).validate(None)
    hybrid = SimpleNamespace(hybrid=SimpleNamespace(enabled=True))
    with pytest.raises(NotImplementedError, match="hybrid"):
        ResidualCfg(enabled=True, base_run="r", steps_per_chunk=2).validate(hybrid)


def test_critic_action_full_works_whatever_the_agent_observes_and_is_refused_on_a_chunked_ee_one(monkeypatch):
    """The critic's input does not depend on the agent's observation: a wrapper whose agent sees neither the
    base action nor a normalized observation still gives the critics the executed chunk."""
    policy = ScriptedPolicy()
    policy.normalizer = a_normalizer()
    monkeypatch.setattr(residual_mod, "load_base_policy", lambda *a, **k: policy)
    rc = ResidualCfg(enabled=True, base_run="r", base_checkpoint="c", action_scale=[0.5] * ACT, steps_per_chunk=2,
                     critic_action="full", observe_base_action=False, normalize_observation=False)
    cfg = SimpleNamespace(task_name=TASK, residual=rc, hybrid=SimpleNamespace(enabled=False, extra_dims=0))
    env = ResidualPolicyWrapper(ChunkEnv(), cfg)
    transform = residual_mod.residual_critic_action(cfg, env)
    obs, _ = env.reset()
    assert obs["policy"].shape == (N, OBS) and obs["critic"].shape == (N, 3 + 2 * ACT)
    r = torch.rand(N, 2 * ACT) * 2 - 1
    critic_action = transform(obs["critic"], r)
    env.step(r)
    for j, sent in enumerate(env.unwrapped.sent[-2:]):
        assert torch.allclose(critic_action[:, j * ACT : (j + 1) * ACT], policy.normalizer.normalize(sent, "action"), atol=1e-5)
    stub = ResidualPolicyWrapper.__new__(ResidualPolicyWrapper)
    stub.ee, stub.steps_per_chunk = object(), 2
    with pytest.raises(NotImplementedError, match="action_frame: ee and residual.steps_per_chunk 2"):
        residual_mod.residual_critic_action(SimpleNamespace(residual=SimpleNamespace(enabled=True, critic_action="full")), stub)


def test_direct_hooks_announce_the_agents_action_width_and_keep_forges_buffer():
    class FakeDirect:
        def __init__(self):
            self.cfg = SimpleNamespace(observation_space=5, state_space=7, action_space=4)
            self.num_envs, self.device = 2, "cpu"
            self.actions = torch.ones(2, 7)

        def _configure_gym_env_spaces(self):
            self.actions = torch.zeros(2, self.cfg.action_space)
            self.action_space = gym.spaces.Box(-1, 1, (self.num_envs, self.cfg.action_space))

    env = FakeDirect()
    DirectHooks(env).set_agent_action_width(16)
    assert env.cfg.action_space == 16 and env.action_space.shape == (2, 16)
    assert torch.equal(env.actions, torch.ones(2, 7))


# ---------------------------------------------------------------------- the observation flags
def a_normalizer(seed=0):
    torch.manual_seed(seed)
    normalizer = MinMaxNormalizer(OBS, ACT)
    normalizer.fit("obs", torch.randn(64, OBS) * 3)
    normalizer.fit("action", torch.randn(64, ACT) * 2)
    return normalizer


def test_residual_observation_flags():
    normalizer = a_normalizer()
    obs, base = torch.randn(4, OBS), torch.randn(4, ACT)
    full, extra = residual_observation(normalizer, obs, base)
    assert torch.allclose(full[:, :OBS], normalizer.normalize(obs, "obs", widen=True)) and torch.equal(extra, full[:, OBS:])
    raw, extra = residual_observation(normalizer, obs, base, normalize_obs=False)
    assert torch.equal(raw[:, :OBS], obs) and torch.equal(extra, normalizer.normalize(base, "action"))
    alone, extra = residual_observation(normalizer, obs, base, observe_base_action=False)
    assert torch.equal(alone, full[:, :OBS]) and extra.shape == (4, 0)
    plain, extra = residual_observation(normalizer, obs, base, normalize_obs=False, observe_base_action=False)
    assert torch.equal(plain, obs) and extra.shape == (4, 0)


def test_the_wrapper_can_leave_the_base_action_out_and_the_observation_raw(monkeypatch):
    env, _ = make(monkeypatch, ChunkEnv(), observe_base_action=False)
    obs, _ = env.reset()
    # nothing appended to the agent's observation; the critic state still gets the base action
    assert obs["policy"].shape == (N, OBS) and obs["critic"].shape == (N, 3 + ACT)
    assert env.base_action_slice is None and env.observation_space["policy"].shape == (N, OBS)
    _, _, _, _, infos = env.step(torch.ones(N, ACT))
    assert env.unwrapped.sent[-1][:, 0].tolist() == [0.5, 0.5]  # the base action is still executed
    # the identity normalizer makes normalize_observation moot here: check the flag through the real one
    policy = ScriptedPolicy()
    policy.normalizer = a_normalizer()
    monkeypatch.setattr(residual_mod, "load_base_policy", lambda *a, **k: policy)
    for normalize in (True, False):
        rc = ResidualCfg(enabled=True, base_run="r", base_checkpoint="c", action_scale=[0.5] * ACT, normalize_observation=normalize)
        cfg = SimpleNamespace(task_name=TASK, residual=rc, hybrid=SimpleNamespace(enabled=False, extra_dims=0))
        wrapped = ResidualPolicyWrapper(ChunkEnv(), cfg)
        obs, _ = wrapped.reset()
        raw = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]]).expand(N, -1)
        expected = policy.normalizer.normalize(raw, "obs", widen=True) if normalize else raw
        assert torch.allclose(obs["policy"][:, :OBS], expected)
        assert torch.allclose(obs["policy"][:, OBS:], policy.normalizer.normalize(wrapped.base_action, "action"))


def test_the_ee_critic_action_reads_the_critic_state_whatever_the_agent_observes(monkeypatch):
    from tests.residual.test_residual_ee import FakePoseEnv
    from my_project.models.cfg import DiffusionUnetActorCfg
    from my_project.models.diffusion_policy import DiffusionPolicy

    torch.manual_seed(0)
    actor = DiffusionUnetActorCfg(horizon=8, n_obs_steps=1, action_horizon=4, down_dims=[16, 32], diffusion_step_embed_dim=16,
                                  n_groups=4, inference_steps=2, num_diffusion_iters=10, warm_start=False)
    policy = DiffusionPolicy(actor, 61, 10)
    policy.normalizer.fit("obs", torch.cat([torch.randn(20, 61), FakePoseEnv().obs]))
    policy.normalizer.fit("action", torch.randn(20, 10))
    monkeypatch.setattr(residual_mod, "load_base_policy", lambda *a, **k: policy.eval())
    rc = ResidualCfg(enabled=True, base_run="r", base_checkpoint="c", action_frame="ee", critic_action="full", normalize_observation=False)
    cfg = SimpleNamespace(task_name="Isaac-Furniture-OneLeg-FBAbs-ResiP-v0", residual=rc, hybrid=SimpleNamespace(enabled=False))
    env = ResidualPolicyWrapper(FakePoseEnv(), cfg)
    obs, _ = env.reset()
    transform = residual_mod.residual_critic_action(cfg, env)
    r = torch.tensor([[1.0, 0, 0, 0, 0, 0.5], [0.0, -0.5, 0.2, 0.3, 0, 0]])
    critic_action = transform(obs["critic"], r)
    base, R_ee = env.base_action.clone(), env.R_ee.clone()
    env.step(r)
    o, _ = env.ee.from_offset_obs(R_ee, critic_action)
    assert torch.allclose(base[:, 0:3] + o, env.env.sent[:, 0:3], atol=1e-5)


class PhaseEnv(ChunkEnv):
    """ChunkEnv publishing a task phase per env in eval_state (``phases[env]`` after every step)."""

    def __init__(self, phases, **kwargs):
        super().__init__(**kwargs)
        self.phases = phases

    def step(self, action):
        out = super().step(action)
        out[4]["eval_state"] = {"phase/id": torch.tensor(self.phases, dtype=torch.float32)}
        return out


def test_active_from_phase_runs_the_base_alone_before_that_phase_and_reports_it(monkeypatch):
    # env 0 stays in grasp (1), env 1 is in insert (2) after the first step
    env, _ = make(monkeypatch, PhaseEnv([1, 2]), active_from_phase="insert")
    env.reset()
    _, _, _, _, infos = env.step(torch.ones(N, ACT))  # a fresh episode starts in corner: nobody acts
    assert infos["residual_active"].tolist() == [0.0, 0.0]
    assert env.unwrapped.sent[-1][:, 0].tolist() == pytest.approx([0.0, 0.0])  # the base alone
    _, _, _, _, infos = env.step(torch.ones(N, ACT))  # now env 1 is past the grasp
    assert infos["residual_active"].tolist() == [0.0, 1.0]
    assert env.unwrapped.sent[-1][:, 0].tolist() == pytest.approx([1.0, 1.0 + 0.5])


def test_active_from_phase_needs_the_phase_channel(monkeypatch):
    env, _ = make(monkeypatch, ChunkEnv(), active_from_phase="insert")
    env.reset()
    with pytest.raises(RuntimeError, match="track_phases"):
        env.step(torch.ones(N, ACT))


def test_active_from_phase_is_validated():
    with pytest.raises(ValueError, match="active_from_phase"):
        ResidualCfg(active_from_phase="lift").validate(None)


def test_the_episode_count_check_matches_the_envs_own_clock(monkeypatch):
    """episode/agent_steps counts the wrapper's agent steps per episode; with the env's own clock
    (episode/isaac_steps) episode/count_offset = isaac - k x agent steps, within +-(k - 1) for a mid-chunk end."""

    class ClockEnv(ChunkEnv):
        def step(self, action):
            before = self.episode_length_buf.clone() + 1
            out = list(super().step(action))
            out[4]["episode_metrics_to_log"]["episode/isaac_steps"] = before.to(torch.float32)
            return tuple(out)

    env, _ = make(monkeypatch, ClockEnv(end_at={0: 3}), steps_per_chunk=2)
    env.reset()
    env.step(residual([[0.0, 0.0], [0.0, 0.0]]))
    _, _, terminated, _, infos = env.step(residual([[0.0, 0.0], [0.0, 0.0]]))
    assert terminated.tolist() == [True, False]
    metrics = infos["episode_metrics_to_log"]
    # env 0 ended at its 3rd sim step (sub-step 0 of the 2nd chunk): 2 agent steps, Isaac's 3 -> offset 3 - 4 = -1
    assert metrics["episode/agent_steps"][0].item() == 2.0 and metrics["episode/count_offset"][0].item() == -1.0
    assert env.agent_steps.tolist() == [0, 2]  # env 0 counts its next episode from 0
