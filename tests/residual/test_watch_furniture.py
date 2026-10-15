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
