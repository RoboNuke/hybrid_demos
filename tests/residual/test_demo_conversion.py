"""The offline demo conversion's bookkeeping, on CPU: alignment of observations, next
observations, critic states and labels with the recorded frames, the base's chunk clock, the
history, and the clip statistics. Isaac's reward code is stubbed (``--check_in_sim`` is the
reward's test against the real env); its inputs are checked for their frame alignment.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from my_project import residual_demos
from my_project import wrappers


class FakeNormalizer:
    """Identity, so the expected values stay readable."""

    action_min = torch.full((2,), -1.0)
    action_max = torch.full((2,), 1.0)

    def normalize(self, x, key):
        return x

    def denormalize(self, x, key):
        return x


class FakePolicy:
    """A base whose chunk is ``[k, k+1, ...]`` x (1, 0) for the k-th chunk it plans per demo."""

    obs_dim, action_dim, horizon, action_horizon, n_obs_steps = 3, 2, 4, 2, 1
    device = torch.device("cpu")
    cfg = SimpleNamespace(obs_clip=10.0, warm_start=False)
    normalizer = FakeNormalizer()

    def __init__(self):
        self.calls = 0

    def eval(self):
        return self

    def parameters(self):
        return []

    def _denoise(self, nobs, start):
        k = float(self.calls)
        self.calls += 1
        steps = torch.arange(self.horizon, dtype=torch.float32) + 100 * k
        return torch.stack((steps, torch.zeros(self.horizon)), dim=-1).expand(nobs.shape[0], -1, -1).clone()


def frames(length: int, offset: float) -> dict:
    t = torch.arange(length, dtype=torch.float32)
    return {
        "state": torch.stack((t + offset, 10 + t, 20 + t), dim=-1),
        "critic": torch.stack((t + offset, -t), dim=-1),
        "action": torch.stack((t * 0 + 0.5, torch.full_like(t, 3.0)), dim=-1),
        "applied": t[:, None].expand(-1, 7) + offset,
        "prev_applied": torch.full((1, 7), -1.0),
        "wrench": torch.zeros(length, 6),
        "force_threshold": torch.ones(length, 1),
        "noisy_hole_pos": torch.zeros(length, 3),
        "fingertip_pos": torch.zeros(length, 3),
        "fingertip_quat": torch.zeros(length, 4),
        "peg_pos": t[:, None].expand(-1, 3) + offset,
        "peg_quat": torch.zeros(length, 4),
        "transitions": length - 1,
    }


def run(monkeypatch, history: bool, beyond: str = "store", hybrid=None):
    seen = []

    def fake_rewards(env_cfg, **kw):
        seen.append(kw)
        return {"total": kw["held_pos"][:, 0], "curr_success": kw["held_pos"][:, 0] > 2}

    monkeypatch.setattr(residual_demos, "forge_rewards", fake_rewards)
    monkeypatch.setattr(residual_demos, "_state_slices", lambda env_cfg: {"fixed_pos": slice(0, 1), "fixed_quat": slice(1, 2)})
    monkeypatch.setattr(wrappers, "obs_term_slices", lambda unwrapped, task_name=None: {"a": slice(0, 1), "b": slice(1, 3)})
    cfg = SimpleNamespace(
        task_name="Isaac-Forge-Fake",
        task_cfg=SimpleNamespace(),
        residual=SimpleNamespace(action_scale=[0.5, 1.0], action_scale_fraction=0.1, demo_actions_beyond_scale=beyond),
        history=SimpleNamespace(enabled=history, terms=["a"], length=2),
        hybrid=hybrid or SimpleNamespace(enabled=False),
    )
    episodes = [{"episode_index": 0}, {"episode_index": 1}]
    data = [frames(5, 0.0), frames(3, 1000.0)]
    out, stats = residual_demos.convert(cfg, FakePolicy(), episodes, data, torch.device("cpu"))
    return out, stats, seen


def test_transitions_line_up_with_the_frames_and_the_base_runs_on_its_own_clock(monkeypatch):
    out, stats, seen = run(monkeypatch, history=False)
    # T - 1 transitions per demo
    assert stats["transitions"] == 4 + 2 and out["observations"].shape == (6, 3 + 2)
    obs, nxt = out["observations"], out["next_observations"]
    # demo 0 first, then demo 1; the next observation is the following frame's
    assert obs[:4, 0].tolist() == [0.0, 1.0, 2.0, 3.0] and nxt[:4, 0].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert obs[4:, 0].tolist() == [1000.0, 1001.0] and nxt[4:, 0].tolist() == [1001.0, 1002.0]
    # base action: both demos plan together at t = 0, 2, 4 (chunk k = 0, 1, 2), 2 steps each
    base_x = obs[:4, 3].tolist()
    assert base_x == [0.0, 1.0, 100.0, 101.0]
    # the critic state is the recorded one with the base action appended
    assert out["states"][:4, 0].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert torch.equal(out["states"][:, 2:], obs[:, 3:5])
    # actions: (demo - base) / scale, exactly; x: (0.5 - base) / 0.5, y: (3 - 0) / 1 = 3, never clipped
    assert out["actions"][:4, 0].tolist() == [1.0, -1.0, -199.0, -201.0]
    assert torch.all(out["actions"][:, 1] == 3.0)
    assert stats["beyond_scale_fraction"] == 1.0 and stats["beyond_scale_fraction_per_dim"][1] == 1.0
    assert not out["terminated"].any() and not out["truncated"].any()
    # the reward of step t reads the state after it (frame t + 1) and the action of step t
    assert seen[0]["held_pos"][:, 0].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert seen[0]["actions"][:, 0].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert seen[0]["prev_actions"][:, 0].tolist() == [-1.0, 0.0, 1.0, 2.0]
    assert out["rewards"][:4, 0].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert stats["success_rate"] == 1.0


def test_with_history_the_observation_and_state_carry_the_same_past(monkeypatch):
    out, _, _ = run(monkeypatch, history=True)
    obs = out["observations"]
    assert obs.shape[1] == 3 + 2 + 2  # base obs, base action, 2 past values of term "a"
    assert obs[:4, 5:].tolist() == [[0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 2.0]]
    assert torch.equal(out["states"][:, -2:], obs[:, -2:])
    assert out["next_observations"][3, 5:].tolist() == [2.0, 3.0]


def test_drop_keeps_only_transitions_the_policy_could_produce(monkeypatch):
    out, stats, _ = run(monkeypatch, history=False, beyond="drop")
    # every frame here needs y = 3 > 1, so every transition is dropped
    assert stats["transitions"] == 0 and stats["demo_steps"] == 6
    assert out["actions"].shape == (0, 2) and out["observations"].shape[0] == 0


def test_with_hybrid_the_demos_choose_position_and_no_force(monkeypatch):
    plain, _, _ = run(monkeypatch, history=False)
    out, stats, _ = run(monkeypatch, history=False, hybrid=SimpleNamespace(enabled=True, axes_for=lambda cfg: 3))
    pose = plain["actions"].shape[1]
    assert out["actions"].shape[1] == pose + 6
    assert torch.equal(out["actions"][:, :pose], plain["actions"])
    assert bool((out["actions"][:, pose : pose + 3] == -1.0).all())
    assert bool((out["actions"][:, pose + 3 :] == 0.0).all())
    # the stats describe the pose correction only
    assert len(stats["max_abs_action"]) == pose
