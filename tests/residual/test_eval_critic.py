"""my_project/eval_critic.py: the eval rows each learner's critics are scored on, and the result keys."""

from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from my_project.eval_critic import CriticEval


class FakeLearner:
    """Records the sets it is scored on; q = rewards, y = rewards + 1 (TD error 1 everywhere)."""

    def __init__(self, n_step=1, num_agents=2):
        self.cfg = SimpleNamespace(std_label_names=["grasp", "insert", "screw"], std_by_label="eval_state/phase/id",
                                   exploration_scale_key=None, discount_factor=0.9)
        self.n = n_step
        self.num_agents = num_agents
        self.seen = []

    def init_side_inputs(self):
        self._side_inputs = {}

    def _demo_names(self):
        return ["observations", "actions", "rewards", "next_observations", "terminated", "truncated"]

    def _n_step(self):
        return self.n

    def critic_values(self, sets):
        self.seen.append(sets)
        return [None if s is None else (s["rewards"].reshape(-1), s["rewards"].reshape(-1) + 1.0) for s in sets]

    health_of = staticmethod(__import__("robonuke_rl_core.learners.sac", fromlist=["SAC"]).SAC.health_of)


def _cfg():
    return OmegaConf.create({"trainer": {"learner": "ddpg"}, "ddpg": {"demo_transitions": None}})


def _run(collector, labels, alive, terminated):
    """T steps of 4 envs; observation of env e at step t is 10*t + e (so consecutive steps hand off)."""
    collector.start_round()
    T = len(labels) - 1
    for t in range(T):
        obs = torch.tensor([[10.0 * t + e] for e in range(4)])
        collector.step(observations=obs, states=None, actions=torch.zeros(4, 1), rewards=torch.full((4, 1), float(t)),
                       next_observations=obs + 10.0, next_states=None, terminated=torch.tensor(terminated[t]),
                       truncated=torch.zeros(4, dtype=torch.bool), infos={"eval_state": {"phase/id": torch.tensor(labels[t + 1])}},
                       alive=torch.tensor(alive[t]), start_label=torch.tensor(labels[t]))


def test_routed_rows_keep_the_phase_end_the_switch_terminal_and_the_rollout_end_truncated():
    insert, screw = FakeLearner(), FakeLearner()
    collector = CriticEval(_cfg(), None, {"insert": insert, "screw": screw}, num_agents=2, num_envs=4)
    # env 0: insert 2 steps then screw; env 1: insert to the end; envs 2, 3 (agent 1): grasp, screw, dies
    labels = [[1, 1, 0, 2], [1, 1, 0, 2], [2, 1, 0, 2], [2, 1, 0, 2]]
    alive = [[True, True, True, True], [True, True, True, False], [True, True, True, False]]
    _run(collector, labels, alive, [[False] * 4] * 3)
    out = collector.results()
    rows = insert.seen[0][0]  # agent 0's insert rows: env 0 steps 0, 1 then env 1 steps 0, 1, 2
    assert rows["rewards"].reshape(-1).tolist() == [0.0, 1.0, 0.0, 1.0, 2.0]
    assert rows["terminated"].reshape(-1).tolist() == [False, True, False, False, False]  # env 0's switch to screw
    assert rows["truncated"].reshape(-1).tolist() == [False, False, False, False, True]  # env 1's rollout stopped
    assert insert.seen[0][1] is None  # agent 1 never in insert
    assert screw.seen[0][1]["rewards"].reshape(-1).tolist() == [0.0]  # env 3's step 0 only (then dead)
    assert out[0]["critic_insert/rows_eval"] == 5 and out[0]["critic_insert/td_abs_mean_eval"] == 1.0
    assert "critic_insert/explained_variance" in out[0] and "critic_insert/rows_demo" not in out[0]
    assert "critic_insert/rows" not in out[1] and out[1]["critic_screw/rows"] == 1
    assert collector.rounds == []


def test_one_learner_rows_fold_into_n_step_and_rounds_join():
    learner = FakeLearner(n_step=2)
    collector = CriticEval(_cfg(), None, {None: learner}, num_agents=2, num_envs=4)
    labels = [[0] * 4] * 3
    _run(collector, labels, [[True] * 4, [True, True, True, False]], [[False, False, False, True], [False] * 4])
    _run(collector, labels, [[True] * 4] * 2, [[False] * 4] * 2)
    out = collector.results()
    rows = learner.seen[0][0]  # agent 0: 2 envs x 2 steps x 2 rounds
    assert rows["rewards"].shape[0] == 8 and "discounts" in rows
    assert torch.allclose(rows["rewards"][:2].reshape(-1), torch.tensor([0.0 + 0.9 * 1.0, 1.0]))
    assert out[1]["critic_all/rows_eval"] == 7  # env 3 ended at its first step in round 1
