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
    cfg.FBABS_RESIP_OBS_LAYOUT = cfg.FBABS_OBS_LAYOUT[:-1] + (("parts_poses_b", 45), ("obstacle_xy", 2))
    cfg.FBREL_RESIP_OBS_LAYOUT = (
        ("ee_pos_vel", 3), ("ee_ori_vel", 3), ("gripper_width", 1), ("parts_poses_ee", 45), ("obstacle_pos_ee", 3)
    )

    def fbabs_obs_layout(env_cfg):  # the real one's rule: the policy group's obstacle term
        policy = getattr(getattr(env_cfg, "observations", None), "policy", None)
        return cfg.FBABS_RESIP_OBS_LAYOUT if getattr(policy, "obstacle_xy", None) is not None else cfg.FBABS_OBS_LAYOUT

    cfg.fbabs_obs_layout = fbabs_obs_layout
    for name, mod in (("furniture_lab", pkg), ("furniture_lab.tasks", tasks),
                      ("furniture_lab.tasks.assembly", assembly), ("furniture_lab.tasks.assembly.assembly_env_cfg", cfg)):
        monkeypatch.setitem(sys.modules, name, mod)
    return FB_LAYOUT


# ---------------------------------------------------------------------- families
def test_the_relative_interface_is_its_own_task_with_the_abs_action_layout(furniture_layout):
    rel, ab = "Isaac-Furniture-OneLeg-FBRel-ResiP-v0", "Isaac-Furniture-OneLeg-FBAbs-ResiP-v0"
    assert families.family_of(rel) == families.FURNITURE and families.is_rel(rel) and not families.is_abs(rel)
    assert not families.is_rel(ab) and not families.is_rel("Isaac-Furniture-OneLeg-FB-v0")
    # [dpos 3, drot6d 6, gripper]: the absolute target's dims, so metrics and checks read it the same way
    assert families.action_layout(rel) == families.action_layout(ab) and families.action_layout(rel)["dim"] == 10
    assert families.action_metric_groups(rel) == {"pos": [0, 1, 2], "rot": [3, 4, 5, 6, 7, 8], "gripper": [9]}
    layout = families.obs_layout(rel, None)
    assert [n for n, _ in layout] == ["ee_pos_vel", "ee_ori_vel", "gripper_width", "parts_poses_ee", "obstacle_pos_ee"]
    assert sum(w for _, w in layout) == 55
    assert not families.observes_obstacle(rel)  # the FBAbs rule, unchanged (the importer handles FBRel itself)


def test_the_delta_interface_is_the_resip_observation_with_a_7_wide_action(furniture_layout):
    delta, ab = "Isaac-Furniture-OneLeg-FBDelta-ResiP-v0", "Isaac-Furniture-OneLeg-FBAbs-ResiP-v0"
    assert families.family_of(delta) == families.FURNITURE and families.is_delta(delta)
    assert not families.is_abs(delta) and not families.is_rel(delta) and not families.is_delta(ab)
    # [dpos 3, drot 3 (axis-angle), gripper]
    assert families.action_layout(delta) == {"dim": 7, "pos": slice(0, 3), "axis_angle": slice(3, 6), "gripper": 6}
    assert families.action_metric_groups(delta) == {"pos": [0, 1, 2], "rot": [3, 4, 5], "gripper": [6]}
    # the observation is the FBAbs ResiP task's, read from the env cfg as for that task
    resip_cfg = SimpleNamespace(observations=SimpleNamespace(policy=SimpleNamespace(obstacle_xy=object())))
    assert families.obs_layout(delta, resip_cfg) == families.obs_layout(ab, resip_cfg)
    assert sum(w for _, w in families.obs_layout(delta, resip_cfg)) == 63
    assert not families.observes_obstacle(delta)  # the FBAbs rule, unchanged (the importer handles FBDelta itself)


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

    def normalize(self, x, key, widen=False):  # a [-1, 1] range: widening never applies
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
        reward[-1] = 1.0  # the recording's own reward: the conversion no longer reads it
    return {  # no "critic": the env has no critic group
        "state": state,
        "next_state": state[1:],
        "action": torch.full((length, 8), 0.5),
        "reward": reward,
        "transitions": length,
        "success": success,
    }


def newly_assembled(env):  # stand-ins for the env's reward terms: the conversion reads their names
    raise AssertionError("never called")


def each_step(env):
    raise AssertionError("never called")


def furniture_rewards_cfg(step_weight: float = 0.0, **extra):
    """The env cfg's ``rewards``: the assembly at weight 1, the per-step term at ``step_weight``."""
    return SimpleNamespace(
        assembled=SimpleNamespace(func=newly_assembled, weight=1.0),
        step=SimpleNamespace(func=each_step, weight=step_weight),
        **extra,
    )


class DoublingNormalizer(FakeNormalizer):
    """Not the identity on the observation: tells the agent's normalized observation from the raw one."""

    def normalize(self, x, key, widen=False):
        return 2.0 * x if key == "obs" else x


def _convert_furniture(monkeypatch, rewards, assembled_from=None, normalizer=None, **residual):
    """Demo 0 (4 steps, a success in its recording) and demo 1 (3 steps, a failure). The env's
    assembled test is stood in: demo 0's leg tests assembled after step ``assembled_from[0]`` on."""
    from my_project.models import diffusion_policy

    class ChunkedBase:
        def __init__(self, policy, n):
            self.policy, self.n = policy, n

        def actions(self, obs, fresh):
            return torch.zeros(self.n, 8)

    assembled_from = {0.0: 2} if assembled_from is None else assembled_from

    def assembled(cfg, frames):
        steps, start = frames["transitions"], assembled_from.get(float(frames["state"][0, 0]))
        now = torch.zeros(steps, 1, dtype=torch.bool)
        if start is not None:
            now[start:] = True
        return now

    monkeypatch.setattr(diffusion_policy, "ChunkedBase", ChunkedBase)
    monkeypatch.setattr(residual_demos, "forge_rewards", lambda *a, **k: pytest.fail("Forge's reward must not run on furniture demos"))
    monkeypatch.setattr(residual_demos, "assembled_after_each_step", assembled)
    cfg = SimpleNamespace(
        task_name="Isaac-Furniture-OneLeg-FB-v0",
        task_cfg=SimpleNamespace(sim=SimpleNamespace(dt=1 / 120), decimation=12, rewards=rewards),
        residual=ResidualCfg(action_scale=[0.5] * 8, action_scale_fraction=0.1, demo_actions_beyond_scale="store", **residual),
        history=SimpleNamespace(enabled=False),
        hybrid=SimpleNamespace(enabled=False),
    )
    episodes = [{"episode_index": 0}, {"episode_index": 1}]
    policy = FakePolicy()
    if normalizer is not None:
        policy.normalizer = normalizer
    return residual_demos.convert(cfg, policy, episodes, [furniture_frames(4, True), furniture_frames(3, False, 100.0)], torch.device("cpu"))


def test_without_a_critic_group_the_states_are_the_agents_observations(monkeypatch, furniture_layout):
    """The env's critic state is the agent's observation there: the file's states must be the same
    columns, raw (the learner's state_preprocessor normalizes both alike)."""
    out, _ = _convert_furniture(monkeypatch, furniture_rewards_cfg(), normalizer=DoublingNormalizer())
    assert out["observations"][:3, 0].tolist() == [0.0, 1.0, 2.0]  # raw: the recorded frame, no normalizer applied
    assert torch.equal(out["states"], out["observations"]) and torch.equal(out["next_states"], out["next_observations"])
    assert out["states"].shape == (6, 49 + 8)


def test_the_demo_critic_state_does_not_depend_on_what_the_agent_observes(monkeypatch, furniture_layout):
    """residual.observe_base_action shapes the agent's observation only: the file's states (the critic state) are the
    observation and the base action whatever it says."""
    ref, _ = _convert_furniture(monkeypatch, furniture_rewards_cfg(), normalizer=DoublingNormalizer())
    out, _ = _convert_furniture(monkeypatch, furniture_rewards_cfg(), normalizer=DoublingNormalizer(),
                                observe_base_action=False)
    assert torch.equal(out["states"], ref["states"]) and torch.equal(out["next_states"], ref["next_states"])
    assert out["observations"].shape == (6, 49) and out["observations"][:3, 0].tolist() == [0.0, 1.0, 2.0]  # raw, no base action


def test_furniture_demos_end_where_the_env_would_with_the_reward_the_terms_define(monkeypatch, furniture_layout):
    """Demo 0's leg tests assembled after step 2: the env ends the episode there, so step 2 is
    terminal and pays the assembly's weight (1: no step_dt, which the package divides out), and
    the recording's step 3 is dropped. Demo 1 never assembles and keeps every step."""
    out, stats = _convert_furniture(monkeypatch, furniture_rewards_cfg())
    assert stats["transitions"] == 3 + 3 and stats["demo_steps"] == 6
    assert stats["rows_after_env_success"] == 1
    assert out["observations"][:3, 0].tolist() == [0.0, 1.0, 2.0]
    assert out["next_observations"][:3, 0].tolist() == [1.0, 2.0, 3.0]
    assert out["rewards"].flatten().tolist() == pytest.approx([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    assert out["terminated"].flatten().tolist() == [False, False, True, False, False, False]
    assert out["truncated"].flatten().tolist() == [False, False, False, False, False, True]  # demo 1's end
    assert torch.all(out["actions"] == 1.0)  # (0.5 - 0) / 0.5
    assert stats["success_rate"] == 0.5 and out["states"].shape == (6, 49 + 8)


def test_the_per_step_term_reaches_the_demo_rewards_as_the_env_pays_it(monkeypatch, furniture_layout):
    """rewards.step at weight -1: every step costs 1, the success step pays both."""
    out, stats = _convert_furniture(monkeypatch, furniture_rewards_cfg(step_weight=-1.0))
    assert out["rewards"].flatten().tolist() == pytest.approx([-1.0, -1.0, 0.0, -1.0, -1.0, -1.0])
    assert out["terminated"].flatten().tolist() == [False, False, True, False, False, False]
    assert stats["success_rate"] == 0.5  # the env's verdict, not a positive total


def test_a_recorded_success_the_envs_test_never_ends_is_refused(monkeypatch, furniture_layout):
    with pytest.raises(ValueError, match="never passes"):
        _convert_furniture(monkeypatch, furniture_rewards_cfg(), assembled_from={})


def test_a_weighted_reward_term_the_conversion_cannot_recompute_is_refused(monkeypatch, furniture_layout):
    def keypoint_distance(env):
        raise AssertionError("never called")

    unset = furniture_rewards_cfg(shaping=None, dense=SimpleNamespace(func=keypoint_distance, weight=0.0))
    _convert_furniture(monkeypatch, unset)  # unset and zero-weight terms pay nothing, online too
    weighted = furniture_rewards_cfg(dense=SimpleNamespace(func=keypoint_distance, weight=0.5))
    with pytest.raises(ValueError, match="rewards.dense .*keypoint_distance"):
        _convert_furniture(monkeypatch, weighted)


def _rot6d_module(monkeypatch):
    """The env repo's rot6d (pure torch) under its own name, without importing its Isaac Lab
    package (``furniture_lab.tasks.assembly.mdp`` imports isaaclab)."""
    import importlib.util
    from pathlib import Path

    root = Path(importlib.util.find_spec("furniture_lab").submodule_search_locations[0])
    spec = importlib.util.spec_from_file_location("furniture_lab.tasks.assembly.mdp.rot6d", root / "tasks/assembly/mdp/rot6d.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("furniture_lab.tasks.assembly", "furniture_lab.tasks.assembly.mdp"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, "furniture_lab.tasks.assembly.mdp.rot6d", module)
    return module


def test_the_envs_assembled_test_runs_on_the_recorded_part_poses(monkeypatch):
    """The leg at one of the top's assembled poses (in the top's OBJ frame) tests assembled; 2 cm
    off it, or tilted 30 degrees (cos 0.87 < ori_bound 0.94), it does not."""
    import math

    from furniture_lab.tasks.furniture import square_table as st

    rot6d = _rot6d_module(monkeypatch)
    fcfg = SimpleNamespace(success_mode="soft", part_names=list(st.PART_NAMES), task="one_leg",
                           assembled_pos_threshold=(0.010, 0.005, 0.010), ori_bound=0.94)
    target = torch.as_tensor(st.rel_pose_targets(st.pairs_for("one_leg"), "obj")[0, 0], dtype=torch.float32)
    top = torch.tensor([0.5, 0.1, 0.02])
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    half = math.radians(30) / 2
    tilted = rot6d.quat_to_rot6d(torch.tensor([[math.cos(half), math.sin(half), 0.0, 0.0]]))[0]

    def state(leg_pos, leg_rot):
        poses = [torch.cat((top, identity))] + [torch.cat((torch.zeros(3), identity))] * 3 + [torch.cat((leg_pos, leg_rot))]
        return torch.cat([torch.zeros(16)] + poses)  # 16 leading dims, then the 5 part poses

    states = torch.stack([
        state(top + target, identity),
        state(top + target + torch.tensor([0.02, 0.0, 0.0]), identity),
        state(top + target, tilted),
    ])
    out = residual_demos.furniture_pairs_assembled(SimpleNamespace(furniture=fcfg), slice(16, 16 + 45), states)
    assert out.flatten().tolist() == [True, False, False]
    with pytest.raises(ValueError, match="'soft' only"):
        strict = SimpleNamespace(**{**vars(fcfg), "success_mode": "strict"})
        residual_demos.furniture_pairs_assembled(SimpleNamespace(furniture=strict), slice(16, 61), states)


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


class FakeAbsEnv(gym.Env):
    """A manager-based FBAbs env in miniature: obs 61, action 10; step returns what it was sent."""

    def __init__(self):
        self.single_observation_space = gym.spaces.Dict({"policy": gym.spaces.Box(-1, 1, (61,))})
        self.single_action_space = gym.spaces.Box(-1, 1, (10,))
        self.observation_space = self.single_observation_space
        self.action_space = self.single_action_space
        self.num_envs, self.device = 2, "cpu"
        self.episode_length_buf = torch.tensor([0, 0])
        self.observation_manager = SimpleNamespace(compute=lambda update_history=False: {"policy": torch.zeros(2, 61)})
        self.cfg = SimpleNamespace()
        self.sent = None

    def reset(self, seed=None, options=None):
        return self.observation_manager.compute(update_history=True), {}

    def step(self, action):
        self.sent = action
        self.episode_length_buf = self.episode_length_buf + 1
        obs = self.observation_manager.compute(update_history=True)
        return obs, torch.zeros(2), torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool), {}


def test_the_residual_steps_the_absolute_rot6d_action_and_logs_its_rotation(monkeypatch):
    import dataclasses

    from my_project import residual as residual_mod
    from my_project.models.cfg import DiffusionUnetActorCfg
    from my_project.models.diffusion_policy import DiffusionPolicy

    torch.manual_seed(0)
    actor = DiffusionUnetActorCfg(horizon=8, n_obs_steps=2, action_horizon=4, down_dims=[16, 32], diffusion_step_embed_dim=16,
                                  n_groups=4, inference_steps=2, num_diffusion_iters=10, warm_start=False)
    policy = DiffusionPolicy(actor, 61, 10)
    policy.normalizer.fit("obs", torch.randn(20, 61))
    policy.normalizer.fit("action", torch.randn(20, 10))
    monkeypatch.setattr(residual_mod, "load_base_policy", lambda *a, **k: policy.eval())
    cfg = SimpleNamespace(
        task_name="Isaac-Furniture-OneLeg-FBAbs-ResiP-v0",
        residual=ResidualCfg(base_run="r", base_checkpoint="c", action_scale=None, action_scale_fraction=0.1),
        hybrid=SimpleNamespace(extra_dims=0, enabled=False),
    )
    env = residual_mod.ResidualPolicyWrapper(FakeAbsEnv(), cfg)
    obs, _ = env.reset()
    assert obs["policy"].shape == (2, 61 + 10)
    _, _, _, _, info = env.step(torch.ones(2, 10))
    assert env.env.sent.shape == (2, 10)
    m = info["metrics_to_log"]
    assert m["residual/position_mm"].shape == (2,) and m["residual/rot_deg"].shape == (2,)
    assert torch.all(m["residual/rot_deg"] >= 0) and torch.all(m["residual/rot_deg"] < 180)
    _, _, _, _, info = env.step(torch.zeros(2, 10))
    assert torch.allclose(info["metrics_to_log"]["residual/rot_deg"], torch.zeros(2), atol=1e-2)  # no residual, no rotation


def test_the_resip_fbabs_layout_appends_the_obstacle(furniture_layout):
    from types import SimpleNamespace

    resip = SimpleNamespace(observations=SimpleNamespace(policy=SimpleNamespace(obstacle_xy=object())))
    layout = families.obs_layout("Isaac-Furniture-OneLeg-FBAbs-ResiP-v0", resip)
    assert layout[-2:] == [("parts_poses_b", 45), ("obstacle_xy", 2)] and sum(w for _, w in layout) == 63
    assert sum(w for _, w in families.obs_layout("Isaac-Furniture-OneLeg-FBAbs-v0", SimpleNamespace())) == 61
    assert families.observes_obstacle("Isaac-Furniture-OneLeg-FBAbs-ResiP-v0")
    assert not families.observes_obstacle("Isaac-Furniture-OneLeg-FBAbs-v0")


class FakePoseAbsEnv(FakeAbsEnv):
    """FakeAbsEnv whose observation holds a valid EE pose (0.5 m forward, identity orientation): the pose residual
    reads the EE."""

    def __init__(self):
        super().__init__()
        obs = torch.zeros(2, 61)
        obs[:, 0] = 0.5
        obs[:, 3], obs[:, 7] = 1.0, 1.0  # rot6d of the identity
        self.observation_manager = SimpleNamespace(compute=lambda update_history=False: {"policy": obs.clone()})


def test_the_chained_additive_target_persists_until_the_base_replans(monkeypatch):
    """action_space: pose, combine: add, action_target: target: a constant r adds up step after step on the base's
    targets (the chained target starts at the EE at each chunk start and follows the base's increments), the current
    target joins the agent's and the critics' observation, and the chain starts over when the base replans."""
    from my_project import residual as residual_mod
    from my_project.models.cfg import DiffusionUnetActorCfg
    from my_project.models.diffusion_policy import DiffusionPolicy

    torch.manual_seed(0)
    actor = DiffusionUnetActorCfg(horizon=8, n_obs_steps=2, action_horizon=4, down_dims=[16, 32], diffusion_step_embed_dim=16,
                                  n_groups=4, inference_steps=2, num_diffusion_iters=10, warm_start=False)
    policy = DiffusionPolicy(actor, 61, 10)
    policy.normalizer.fit("obs", torch.randn(20, 61))
    policy.normalizer.fit("action", torch.randn(20, 10))
    monkeypatch.setattr(residual_mod, "load_base_policy", lambda *a, **k: policy.eval())
    cfg = SimpleNamespace(
        task_name="Isaac-Furniture-OneLeg-FBAbs-ResiP-v0",
        residual=ResidualCfg(base_run="r", base_checkpoint="c", action_space="pose", combine="add", action_target="target",
                             rotation_param="euler", critic_state="observation", critic_action="full", step_scale_pos=0.02,
                             step_scale_rot_deg=6.0),
        hybrid=SimpleNamespace(extra_dims=0, enabled=False),
    )
    env = residual_mod.ResidualPolicyWrapper(FakePoseAbsEnv(), cfg)
    obs, _ = env.reset()
    assert obs["policy"].shape == (2, 61 + 7 + 6) and obs["critic"].shape == (2, 61 + 6)
    assert env.side.shape == (2, 35)
    r = torch.zeros(2, 6)
    r[:, 0] = 1.0  # +2 cm along the robot base's x every step
    offsets = []
    for _ in range(6):
        base = env.base_action.clone()
        env.step(r)
        offsets.append(float((env.env.sent[0, 0] - base[0, 0]) * 1000))
    # steps 1-4 add up within the base's 4-step chunk; the replan at step 5 starts the chain over at the EE
    assert offsets == pytest.approx([20.0, 40.0, 60.0, 80.0, 20.0, 40.0], abs=1e-3)
