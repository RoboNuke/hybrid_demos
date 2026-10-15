"""The env-family split (``my_project/families.py``, ``my_project/envhooks.py``) and the
FurnitureBench branch of the demo conversion, on CPU fakes -- no Isaac Sim, no Forge."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import gymnasium as gym
import pytest
import torch

from my_project import families, residual_demos
from my_project.cfg import HistoryCfg, ResidualCfg
from my_project.envhooks import DirectHooks, ManagerHooks, hooks_for

FB_LAYOUT = (("ee_pos", 3), ("ee_quat", 4), ("ee_pos_vel", 3), ("ee_ori_vel", 3), ("gripper_width", 1), ("parts_poses", 35))


@pytest.fixture
def furniture_layout(monkeypatch):
    """``furniture_lab`` is not importable here: stand in its layout constant."""
    pkg = ModuleType("furniture_lab")
    tasks = ModuleType("furniture_lab.tasks")
    assembly = ModuleType("furniture_lab.tasks.assembly")
    cfg = ModuleType("furniture_lab.tasks.assembly.assembly_env_cfg")
    cfg.FB_OBS_LAYOUT = FB_LAYOUT
    cfg.FBABS_OBS_LAYOUT = (("ee_pos", 3), ("ee_rot6d", 6), ("ee_pos_vel", 3), ("ee_ori_vel", 3), ("gripper_width", 1), ("parts_poses", 45))
    for name, mod in (("furniture_lab", pkg), ("furniture_lab.tasks", tasks),
                      ("furniture_lab.tasks.assembly", assembly), ("furniture_lab.tasks.assembly.assembly_env_cfg", cfg)):
        monkeypatch.setitem(sys.modules, name, mod)
    return FB_LAYOUT


# ---------------------------------------------------------------------- families
def test_the_family_comes_from_the_task_name():
    assert families.family_of("Isaac-Furniture-OneLeg-FB-v0") == families.FURNITURE
    assert families.family_of("Isaac-Forge-PegInsert-Direct-v0") == families.FORGE
    assert families.family_of("Fake-Task-v0") == families.FORGE  # the project's default family


def test_the_furniture_observation_layout_is_furniturebenchs_and_the_action_is_8_wide(furniture_layout):
    slices = families.obs_term_slices_of("Isaac-Furniture-OneLeg-FB-v0", None)
    assert slices["ee_pos"] == slice(0, 3) and slices["parts_poses"] == slice(14, 49)
    layout = families.action_layout("Isaac-Furniture-OneLeg-FB-v0")
    assert layout["dim"] == 8 and layout["quat_xyzw"] == slice(3, 7) and layout["gripper"] == 7
    assert families.action_layout("forge-x") == {"dim": 4, "pos": slice(0, 3), "yaw": 3}
    # the absolute rot6d interface
    assert families.is_abs("Isaac-Furniture-OneLeg-FBAbs-v0") and not families.is_abs("Isaac-Furniture-OneLeg-FB-v0")
    assert families.obs_term_slices_of("Isaac-Furniture-OneLeg-FBAbs-v0", None)["parts_poses"] == slice(16, 61)
    assert families.action_layout("Isaac-Furniture-OneLeg-FBAbs-v0") == {"dim": 10, "pos": slice(0, 3), "rot6d": slice(3, 9), "gripper": 9}
    assert families.action_metric_groups("Isaac-Furniture-OneLeg-FBAbs-v0") == {"pos": [0, 1, 2], "rot": [3, 4, 5, 6, 7, 8], "gripper": [9]}


def test_history_terms_are_checked_against_the_familys_layout(furniture_layout):
    cfg = SimpleNamespace(task_name="Isaac-Furniture-OneLeg-FB-v0", task_cfg=SimpleNamespace())
    HistoryCfg(enabled=True, terms=["ee_pos", "ee_pos_vel"]).validate(cfg)
    with pytest.raises(ValueError, match="fingertip_pos"):
        HistoryCfg(enabled=True).validate(cfg)  # the Forge default terms


def residual_cfg_for(task_name: str, world_pose: bool, action_scale=None):
    return SimpleNamespace(
        task_name=task_name,
        action=SimpleNamespace(world_pose=world_pose),
        trainer=SimpleNamespace(learner="sac"),
    ), ResidualCfg(enabled=True, base_run="runs/bc", action_scale=action_scale)


def test_bc_metric_groups_default_to_the_familys_action_layout(furniture_layout):
    from my_project.cfg import BCCfg

    def bc_cfg(task_name):
        return SimpleNamespace(
            task_name=task_name,
            trainer=SimpleNamespace(learner="diffusion_bc", total_timesteps=1000, checkpoint_interval=1000),
            experiment=SimpleNamespace(num_agents=1),
            model=SimpleNamespace(architecture="diffusion_transformer"),
        )

    bc = BCCfg(dataset="/demos")
    bc.validate(bc_cfg("Isaac-Furniture-OneLeg-FB-v0"))
    assert bc.action_metric_groups == {"pos": [0, 1, 2], "rot": [3, 4, 5, 6], "gripper": [7]}
    bc = BCCfg(dataset="/demos")
    bc.validate(bc_cfg("forge"))
    assert bc.action_metric_groups == {"pos": [0, 1, 2], "yaw": [3]}


def test_the_residual_needs_world_pose_on_forge_only_and_a_scale_per_action_dim():
    cfg, rc = residual_cfg_for("forge", world_pose=False)
    with pytest.raises(ValueError, match="world_pose"):
        rc.validate(cfg)
    cfg, rc = residual_cfg_for("Isaac-Furniture-OneLeg-FB-v0", world_pose=False)
    rc.validate(cfg)
    cfg, rc = residual_cfg_for("Isaac-Furniture-OneLeg-FB-v0", world_pose=True)
    with pytest.raises(ValueError, match="Forge interface"):
        rc.validate(cfg)
    cfg, rc = residual_cfg_for("Isaac-Furniture-OneLeg-FB-v0", world_pose=False, action_scale=[0.1] * 4)
    with pytest.raises(ValueError, match="8 positive"):
        rc.validate(cfg)
    cfg, rc = residual_cfg_for("Isaac-Furniture-OneLeg-FB-v0", world_pose=False, action_scale=[0.1] * 8)
    rc.validate(cfg)


# ---------------------------------------------------------------------- env hooks
class FakeDirect:
    def __init__(self):
        self.cfg = SimpleNamespace(observation_space=5, state_space=7, action_space=4)
        self.num_envs, self.device = 2, "cpu"
        self.episode_length_buf = torch.tensor([0, 3])
        self.actions = torch.ones(2, 4)
        self.rebuilt = 0

    def _configure_gym_env_spaces(self):
        self.rebuilt += 1
        self.actions = torch.zeros(2, self.cfg.action_space)
        self.observation_space = gym.spaces.Box(-1, 1, (self.num_envs, self.cfg.observation_space))
        self.action_space = gym.spaces.Box(-1, 1, (self.num_envs, self.cfg.action_space))

    def _get_observations(self):
        return {"policy": torch.zeros(2, 5), "critic": torch.zeros(2, 7)}


class FakeManager:
    def __init__(self):
        self.single_observation_space = gym.spaces.Dict({"policy": gym.spaces.Box(-1, 1, (49,))})
        self.single_action_space = gym.spaces.Box(-1, 1, (8,))
        self.num_envs, self.device = 2, "cpu"
        self.episode_length_buf = torch.tensor([0, 3])
        # Isaac Lab's signature: compute(update_history=False)
        self.observation_manager = SimpleNamespace(compute=lambda update_history=False: {"policy": torch.zeros(2, 49)})
        self.cfg = SimpleNamespace()


def test_direct_hooks_widen_both_spaces_and_keep_the_action_buffer():
    env = FakeDirect()
    hooks = hooks_for(env)
    assert isinstance(hooks, DirectHooks) and hooks.obs_width() == 5 and hooks.action_width() == 4 and hooks.has_critic()
    hooks.add_obs_dims(3)
    assert env.cfg.observation_space == 8 and env.cfg.state_space == 10 and env.rebuilt == 1
    assert torch.equal(env.actions, torch.ones(2, 4))  # not the rebuild's zeros
    hooks.hook_observations(lambda o: {"policy": o["policy"] + 1, **{k: v for k, v in o.items() if k != "policy"}})
    assert env._get_observations()["policy"].tolist() == [[1.0] * 5] * 2
    assert hooks.fresh().tolist() == [True, False]


def test_manager_hooks_widen_the_policy_group_and_post_process_the_manager():
    env = FakeManager()
    hooks = hooks_for(env)
    assert isinstance(hooks, ManagerHooks) and hooks.obs_width() == 49 and hooks.action_width() == 8 and not hooks.has_critic()
    hooks.add_obs_dims(8)
    assert env.single_observation_space["policy"].shape == (57,)
    assert env.observation_space["policy"].shape == (2, 57)
    hooks.hook_observations(lambda o: {"policy": torch.cat((o["policy"], torch.ones(2, 8)), dim=-1)})
    assert env.observation_manager.compute()["policy"].shape == (2, 57)
    assert env.observation_manager.compute(update_history=True)["policy"].shape == (2, 57)  # as reset/step call it


def test_an_unknown_env_is_refused():
    with pytest.raises(TypeError):
        hooks_for(SimpleNamespace(cfg=SimpleNamespace()))


# ---------------------------------------------------------------------- the furniture conversion
class FakeNormalizer:
    action_min = torch.full((8,), -1.0)
    action_max = torch.ones(8)

    def normalize(self, x, key):
        return x

    def denormalize(self, x, key):
        return x


class FakePolicy:
    """The base: action 8, always zero; chunk of 2."""

    obs_dim, action_dim, pred_horizon, action_horizon = 49, 8, 2, 2
    normalizer = FakeNormalizer()
    device = torch.device("cpu")
    stats = {"action": {"min": torch.full((8,), -1.0), "max": torch.ones(8)}}

    def __call__(self, obs):
        return torch.zeros(obs.shape[0], 2, 8)

    def eval(self):
        return self


def furniture_frames(length: int, success: bool, offset: float = 0.0):
    t = torch.arange(length + 1, dtype=torch.float32) + offset
    state = t[:, None].expand(-1, 49).clone()
    reward = torch.zeros(length, 1)
    if success:
        reward[-1] = 1.0
    return {
        "state": state,
        "critic": state,
        "action": torch.full((length, 8), 0.5),
        "reward": reward,
        "transitions": length,
        "success": success,
    }


def test_furniture_demos_yield_a_transition_per_action_with_the_recorded_reward(monkeypatch, furniture_layout):
    from my_project.models import diffusion_policy

    class ChunkedBase:
        def __init__(self, policy, n):
            self.policy, self.n = policy, n

        def actions(self, obs, fresh):
            return torch.zeros(self.n, 8)

    monkeypatch.setattr(diffusion_policy, "ChunkedBase", ChunkedBase)
    monkeypatch.setattr(residual_demos, "forge_rewards", lambda *a, **k: pytest.fail("Forge's reward must not run on furniture demos"))
    cfg = SimpleNamespace(
        task_name="Isaac-Furniture-OneLeg-FB-v0",
        task_cfg=SimpleNamespace(sim=SimpleNamespace(dt=1 / 120), decimation=12),
        residual=SimpleNamespace(action_scale=[0.5] * 8, action_scale_fraction=0.1, demo_actions_beyond_scale="store"),
        history=SimpleNamespace(enabled=False),
        hybrid=SimpleNamespace(enabled=False),
    )
    episodes = [{"episode_index": 0}, {"episode_index": 1}]
    out, stats = residual_demos.convert(cfg, FakePolicy(), episodes, [furniture_frames(4, True), furniture_frames(3, False, 100.0)], torch.device("cpu"))
    assert stats["transitions"] == 4 + 3 and stats["demo_steps"] == 7
    assert out["observations"][:4, 0].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert out["next_observations"][:4, 0].tolist() == [1.0, 2.0, 3.0, 4.0]  # the last action's next state too
    # the recorded 1 at assembly, paid as the env pays it: x step_dt (0.1 at 10 Hz)
    assert out["rewards"].flatten().tolist() == pytest.approx([0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0])
    assert out["terminated"].flatten().tolist() == [False, False, False, True, False, False, False]
    assert torch.all(out["actions"] == 1.0)  # (0.5 - 0) / 0.5
    assert stats["success_rate"] == 0.5 and out["states"].shape == (7, 49 + 8)


def test_the_episode_limit_is_the_managers_full_length_on_furniture():
    task = SimpleNamespace(episode_length_s=30.0, sim=SimpleNamespace(dt=1 / 120), decimation=6)
    assert residual_demos._episode_limit(SimpleNamespace(task_name="Isaac-Furniture-OneLeg-FB-v0", task_cfg=task)) == 600
    assert residual_demos._episode_limit(SimpleNamespace(task_name="forge", task_cfg=task)) == 599


def test_furniture_abs_demos_read_the_importers_absolute_columns(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    T = 3
    table = pa.table({
        "observation.state": pa.array([[0.0] * 49] * T, pa.list_(pa.float32(), 49)),
        "action": pa.array([[0.0] * 8] * T, pa.list_(pa.float32(), 8)),
        "observation.next_state": pa.array([[0.0] * 49] * T, pa.list_(pa.float32(), 49)),
        "observation.state_abs": pa.array([[float(t)] * 61 for t in range(T)], pa.list_(pa.float32(), 61)),
        "action.abs": pa.array([[0.5] * 10] * T, pa.list_(pa.float32(), 10)),
        "observation.next_state_abs": pa.array([[float(t + 1)] * 61 for t in range(T)], pa.list_(pa.float32(), 61)),
        "reward": pa.array([0.0, 0.0, 1.0], pa.float32()),
    })
    pq.write_table(table, tmp_path / "ep.parquet")
    episode = {"file": "ep.parquet", "success": True}
    out = residual_demos._read_episode(tmp_path, episode, 61, torch.device("cpu"), families.FURNITURE, absolute=True)
    assert out["state"].shape == (T + 1, 61) and out["action"].shape == (T, 10)
    assert out["state"][:, 0].tolist() == [0.0, 1.0, 2.0, 3.0] and out["transitions"] == T
    assert out["success"] and out["reward"].flatten().tolist() == [0.0, 0.0, 1.0]
    with pytest.raises(ValueError, match="base policy takes 61"):
        residual_demos._read_episode(tmp_path, episode, 61, torch.device("cpu"), families.FURNITURE, absolute=False)
