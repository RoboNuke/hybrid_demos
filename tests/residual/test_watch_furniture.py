"""The residual watcher on a FurnitureBench env: each env's first episode counts."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from my_project import residual_eval


class FakeTerms:
    active_terms = ["time_out", "success", "part_out_of_limits"]

    def __init__(self):
        self.terms = {k: torch.zeros(3, dtype=torch.bool) for k in self.active_terms}

    def get_term(self, name):
        return self.terms[name]


class FakeEnv:
    """3 envs, 6-step episodes: env 0 succeeds at step 2 (and again in its second episode),
    env 1 goes out of bounds at step 1, env 2 times out."""

    def __init__(self):
        self.termination_manager = FakeTerms()
        self.num_envs, self.device, self.max_episode_length = 3, "cpu", 6
        self.t = 0
        self.unwrapped = self

    def step(self, actions):
        self.t += 1
        terms = self.termination_manager.terms
        for k in terms:
            terms[k] = torch.zeros(3, dtype=torch.bool)
        terms["success"][0] = self.t in (2, 5)
        terms["part_out_of_limits"][1] = self.t == 1
        terms["time_out"][2] = self.t == 6
        terminated = (terms["success"] | terms["part_out_of_limits"]).reshape(3, 1)
        return torch.zeros(3, 4), None, terminated, terms["time_out"].reshape(3, 1), {}


def test_furniture_rounds_count_each_envs_first_episode(monkeypatch):
    import robonuke_rl_core.envs.build as build
    import robonuke_rl_core.evaluation as evaluation

    monkeypatch.setattr(build, "reseed_env", lambda env, seed: None)
    monkeypatch.setattr(evaluation, "force_env_reset", lambda env: (torch.zeros(3, 4), None))
    monkeypatch.setattr(evaluation, "env_states", lambda env: None)
    out = residual_eval.evaluate(FakeEnv(), lambda o, s: torch.zeros(3, 2), seed=0)
    assert abs(out["success_rate"] - 1 / 3) < 1e-6 and abs(out["out_of_bounds_rate"] - 1 / 3) < 1e-6
    assert out["lengths"] == [2, 1, 6] and out["episodes"] == 3
    assert "force_mean" not in out and "success" in residual_eval._summary(out)


class IsaacTerms(FakeTerms):
    """Isaac Lab's TerminationManager buffer: a row's terms are rewritten only when one of them
    fires, and a reset never clears them — the last ended episode's terms stay."""

    def fire(self, name, rows):
        for k in self.terms:
            self.terms[k][rows] = False
        self.terms[name][rows] = True


class TwoRoundEnv(FakeEnv):
    """Round 1: env 0 succeeds at step 2. Round 2 (same env object, after a forced reset): env 0
    only times out, yet the buffer still says success until a term fires again."""

    def __init__(self):
        super().__init__()
        self.termination_manager = IsaacTerms()
        self.round = 1

    def step(self, actions):
        self.t += 1
        ended = torch.zeros(3, dtype=torch.bool)
        if self.round == 1 and self.t == 2:
            self.termination_manager.fire("success", [0])
            ended[0] = True
        # env 0 is in its second episode in round 1 (it resets on its own), so only 1 and 2 end
        rows = [0, 1, 2] if self.round == 2 else [1, 2]
        timeout = torch.zeros(3, dtype=torch.bool)
        if self.t == 6:
            self.termination_manager.fire("time_out", rows)
            timeout[rows] = True
        return torch.zeros(3, 4), None, ended.reshape(3, 1), timeout.reshape(3, 1), {}


def test_a_round_does_not_count_the_previous_rounds_successes(monkeypatch):
    import robonuke_rl_core.envs.build as build
    import robonuke_rl_core.evaluation as evaluation

    env = TwoRoundEnv()

    def reset(e):
        e.t = 0
        return torch.zeros(3, 4), None

    monkeypatch.setattr(build, "reseed_env", lambda env, seed: None)
    monkeypatch.setattr(evaluation, "force_env_reset", reset)
    monkeypatch.setattr(evaluation, "env_states", lambda env: None)
    first = residual_eval.evaluate(env, lambda o, s: torch.zeros(3, 2), seed=0)
    assert abs(first["success_rate"] - 1 / 3) < 1e-6
    env.round = 2
    assert bool(env.termination_manager.get_term("success")[0])  # stale from round 1
    second = residual_eval.evaluate(env, lambda o, s: torch.zeros(3, 2), seed=0)
    assert second["success_rate"] == 0.0 and second["lengths"] == [6, 6, 6]
