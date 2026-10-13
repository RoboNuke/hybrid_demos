"""The observation history's buffer and the demo-transition fingerprint check, on CPU fakes."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from my_project.cfg import HistoryCfg
from my_project.wrappers import HistoryBuffer, ObservationHistory
from my_project import residual_demos


def make_history(length: int = 3, index=(0, 2), num_envs: int = 2) -> HistoryBuffer:
    return HistoryBuffer(torch.tensor(index), length, num_envs, "cpu")


def test_an_episode_start_repeats_the_current_value_then_the_history_shifts_oldest_first():
    history = make_history(length=3, index=(0, 2), num_envs=1)
    obs = lambda v: torch.tensor([[v, 99.0, 10 * v]])
    step = lambda v, fresh: history.step(obs(v), torch.tensor([fresh]))[0].tolist()
    assert step(1.0, True) == [1.0, 10.0] * 3
    assert step(2.0, False) == [1.0, 10.0, 1.0, 10.0, 1.0, 10.0]
    assert step(3.0, False) == [1.0, 10.0, 1.0, 10.0, 2.0, 20.0]
    assert step(4.0, False) == [1.0, 10.0, 2.0, 20.0, 3.0, 30.0]
    assert history.width == 6


def test_only_the_env_that_reset_is_refilled():
    history = make_history(length=2, index=(0,), num_envs=2)
    history.step(torch.tensor([[1.0], [5.0]]), torch.tensor([True, True]))
    history.step(torch.tensor([[2.0], [6.0]]), torch.tensor([False, False]))
    past = history.step(torch.tensor([[3.0], [7.0]]), torch.tensor([False, True]))
    assert past.tolist() == [[1.0, 2.0], [7.0, 7.0]]


class FakeEnv:
    def __init__(self, num_envs: int = 2) -> None:
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
        self.unwrapped = self
        self.next = None

    def produce(self):
        return {k: v.clone() for k, v in self.next.items()}


def test_the_wrapper_appends_the_same_history_to_the_observation_and_the_critic_state():
    env = FakeEnv(1)
    wrapper = ObservationHistory.__new__(ObservationHistory)
    wrapper.env = env
    wrapper.history = make_history(length=2, index=(0,), num_envs=1)
    wrapper._original_get_observations = env.produce
    env.next = {"policy": torch.tensor([[4.0, 1.0]]), "critic": torch.zeros(1, 3)}
    out = wrapper._get_observations()
    assert out["policy"].tolist() == [[4.0, 1.0, 4.0, 4.0]]
    assert out["critic"].tolist() == [[0.0, 0.0, 0.0, 4.0, 4.0]]


@pytest.mark.parametrize(
    "fields, match",
    [({"length": 0}, "length"), ({"terms": []}, "terms"), ({"terms": ["ft_force", "ft_force"]}, "terms")],
)
def test_history_cfg_rejects_a_bad_value(fields, match):
    with pytest.raises(ValueError, match=match):
        HistoryCfg(enabled=True, **fields).validate(None)


def test_history_terms_must_be_in_the_obs_order():
    cfg = SimpleNamespace(task_cfg=SimpleNamespace(obs_order=["fingertip_pos", "ft_force"]))
    HistoryCfg(enabled=True).validate(cfg)
    with pytest.raises(ValueError, match="joint_pos"):
        HistoryCfg(enabled=True, terms=["joint_pos"]).validate(cfg)


# ---------------------------------------------------------------------- the demo fingerprint
def fake_cfg(path, **changes):
    root = {
        "task": {"name": "T", "cfg": {"episode_length_s": 15.0, "seed": 1, "scene": {"num_envs": 1}, "obs_order": ["a"]}},
        "action": {"world_pose": True},
        "residual": {"enabled": True, "base_run": "runs/bc", "action_scale": None},
        "history": {"enabled": False, "length": 8},
        "hybrid": {"enabled": False},
    }
    for dotted, value in changes.items():
        node = root
        *parents, leaf = dotted.split(".")
        for p in parents:
            node = node[p]
        node[leaf] = value
    return SimpleNamespace(
        root=OmegaConf.create(root),
        task_name="T",
        trainer=SimpleNamespace(learner="sac"),
        sac=SimpleNamespace(demo_transitions=str(path)),
    )


def test_a_file_made_for_another_setup_is_refused_and_cosmetic_task_fields_are_not(tmp_path):
    path = tmp_path / "demos.pt"
    torch.save({"transitions": {}, "meta": {"fingerprint": residual_demos.demo_fingerprint(fake_cfg(path))}}, path)
    residual_demos.check_demo_transitions(fake_cfg(path))
    # episode length, seed and env count do not change a replayed transition
    residual_demos.check_demo_transitions(
        fake_cfg(path, **{"task.cfg.episode_length_s": 30.0, "task.cfg.seed": 7, "task.cfg.scene.num_envs": 64})
    )
    for change in ({"history.enabled": True}, {"residual.base_run": "runs/other"}, {"task.cfg.obs_order": ["b"]}):
        with pytest.raises(ValueError, match="different setup"):
            residual_demos.check_demo_transitions(fake_cfg(path, **change))


def test_a_missing_file_is_left_to_the_learner(tmp_path):
    residual_demos.check_demo_transitions(fake_cfg(tmp_path / "absent.pt"))


def test_a_field_added_later_at_its_default_does_not_invalidate_a_file(tmp_path):
    path = tmp_path / "demos.pt"
    made = residual_demos.demo_fingerprint(fake_cfg(path))
    torch.save({"transitions": {}, "meta": {"fingerprint": made}}, path)
    # the run now carries a residual field the file predates, at its default
    residual_demos.check_demo_transitions(fake_cfg(path, **{"residual.action_scale_fraction": 0.1}))
    with pytest.raises(ValueError, match="action_scale_fraction"):
        residual_demos.check_demo_transitions(fake_cfg(path, **{"residual.action_scale_fraction": 0.3}))
