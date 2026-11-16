"""The env-family split (``my_project/families.py``, ``my_project/envhooks.py``) and the
FurnitureBench branch of the demo conversion, on CPU fakes -- no Isaac Sim, no Forge."""

from __future__ import annotations

import math
import sys
from types import ModuleType, SimpleNamespace

import gymnasium as gym
import pytest
import torch

from my_project import families
from my_project.cfg import HistoryCfg
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


def test_the_resip_fbabs_layout_appends_the_obstacle(furniture_layout):
    from types import SimpleNamespace

    resip = SimpleNamespace(observations=SimpleNamespace(policy=SimpleNamespace(obstacle_xy=object())))
    layout = families.obs_layout("Isaac-Furniture-OneLeg-FBAbs-ResiP-v0", resip)
    assert layout[-2:] == [("parts_poses_b", 45), ("obstacle_xy", 2)] and sum(w for _, w in layout) == 63
    assert sum(w for _, w in families.obs_layout("Isaac-Furniture-OneLeg-FBAbs-v0", SimpleNamespace())) == 61
    assert families.observes_obstacle("Isaac-Furniture-OneLeg-FBAbs-ResiP-v0")
    assert not families.observes_obstacle("Isaac-Furniture-OneLeg-FBAbs-v0")


