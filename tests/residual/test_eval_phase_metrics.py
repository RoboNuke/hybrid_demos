"""scripts/watch_residual_eval.py's FurnitureBench round (my_project/residual_eval.evaluate_furniture): the per-phase
success among the episodes that reached each phase, and the mean correction per phase."""

from __future__ import annotations

from types import SimpleNamespace

import torch

import my_project.residual_eval as residual_eval

NAN = float("nan")


class PhaseEnv:
    """Two envs, three agent steps each. Env 0 inserts and fails to screw (ends at step 2); env 1 never gets past
    grasp (ends at step 3). The phase label at each step's end and the correction each step applied are scripted."""

    def __init__(self):
        self.num_envs, self.device, self.max_episode_length = 2, "cpu", 3
        self.unwrapped = self
        self.t = 0
        terms = {"success": torch.zeros(2, dtype=torch.bool), "part_out_of_limits": torch.zeros(2, dtype=torch.bool)}
        self.termination_manager = SimpleNamespace(get_term=lambda name: terms[name], active_terms=list(terms))
        self.labels = [[2, 1], [3, 1], [5, 1]]
        self.mm = [[1.0, 0.0], [3.0, 0.0], [5.0, 0.0]]
        self.ends = [[False, False], [True, False], [False, True]]

    def step(self, action):
        t, self.t = self.t, self.t + 1
        terminated = torch.tensor(self.ends[t])
        infos = {
            "metrics_to_log": {"residual/position_mm": torch.tensor(self.mm[t]), "residual/rot_deg": torch.tensor(self.mm[t]) / 10},
            "episode_metrics_to_log": {
                "phase/passed_corner": torch.tensor([1.0, 1.0]),
                "phase/passed_grasp": torch.tensor([1.0, 0.0]),
                "phase/passed_insert": torch.tensor([1.0, NAN]),
                "phase/passed_success": torch.tensor([0.0, NAN]),
            },
            "eval_state": {"phase/id": torch.tensor(self.labels[t], dtype=torch.float32)},
        }
        return torch.zeros(2, 1), torch.zeros(2), terminated, torch.zeros(2, dtype=torch.bool), infos


def test_the_round_reports_each_phases_success_among_those_that_reached_it_and_the_correction_per_phase(monkeypatch):
    import robonuke_rl_core.envs.build as build
    import robonuke_rl_core.evaluation as evaluation

    env = PhaseEnv()
    monkeypatch.setattr(build, "reseed_env", lambda env, seed: None)
    monkeypatch.setattr(evaluation, "force_env_reset", lambda env: (torch.zeros(2, 1), None))
    monkeypatch.setattr(evaluation, "env_states", lambda env: None)
    monkeypatch.setattr(residual_eval, "_residual_or_none", lambda env: None)
    out = residual_eval.evaluate_furniture(env, lambda o, s: torch.zeros(2, 1), seed=0)
    assert out["phase_success/corner"] == 1.0 and out["phase_success/grasp"] == 0.5
    assert out["phase_success/insert"] == 1.0 and out["phase_reached/insert"] == 1  # env 1 never reached insert
    assert out["phase_success/screw"] == 0.0 and out["phase_reached/screw"] == 1
    # the phase each step started in: env 0 corner (0), insert (label 2 after step 0), screw (label 3 after step 1)
    assert out["residual_mm/corner"] == 0.5  # env 0's 1.0 and env 1's 0.0 at step 0
    assert out["residual_mm/insert"] == 3.0 and out["residual_mm/grasp"] == 0.0
    assert "residual_mm/screw" not in out  # env 0 ended at step 1, before acting in screw


def test_the_round_hands_a_routed_policy_each_steps_infos(monkeypatch):
    """A routed policy acts from the phase labels the env publishes (observe); without them it is the base alone."""
    import robonuke_rl_core.envs.build as build
    import robonuke_rl_core.evaluation as evaluation

    env = PhaseEnv()
    monkeypatch.setattr(build, "reseed_env", lambda env, seed: None)
    monkeypatch.setattr(evaluation, "force_env_reset", lambda env: (torch.zeros(2, 1), None))
    monkeypatch.setattr(evaluation, "env_states", lambda env: None)
    monkeypatch.setattr(residual_eval, "_residual_or_none", lambda env: None)

    class Routed:
        def __init__(self):
            self.seen = []

        def __call__(self, observations, states):
            return torch.zeros(2, 1)

        def observe(self, infos):
            self.seen.append(infos["eval_state"]["phase/id"].tolist())

    policy = Routed()
    residual_eval.evaluate_furniture(env, policy, seed=0)
    assert policy.seen == [[2.0, 1.0], [3.0, 1.0], [5.0, 1.0]]


def test_skip_base_leaves_out_the_base_alone_round(monkeypatch, tmp_path):
    from types import SimpleNamespace

    calls = []
    monkeypatch.setattr(residual_eval, "evaluate", lambda env, policy, seed, label: calls.append(label) or {"success_rate": 0.0})
    env = SimpleNamespace(action_space=SimpleNamespace(shape=(1, 28)))
    cfg = SimpleNamespace(trainer=SimpleNamespace(total_timesteps=10))
    recorded = []
    record = lambda step, label, result: recorded.append(step)  # noqa: E731
    for skip, expected in ((True, []), (False, [0])):
        recorded.clear()
        args = SimpleNamespace(steps=[], once=False, poll_seconds=0.0, stochastic=False, skip_base=skip)
        residual_eval._evaluate_all(env, cfg, tmp_path, set(), record, 0, args, None)
        assert recorded == expected
