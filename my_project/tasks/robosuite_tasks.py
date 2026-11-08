"""ResFiT's robosuite tasks (robomimic Can and Square, DexMimicGen BoxCleanup, CanSort and Coffee) as their code builds
them, low-dim, behind the surface of an Isaac Lab ``DirectRLEnv`` so the package and the project wrappers take them
unchanged (``task.simulator: standalone``: Isaac Sim never starts).

ResFiT (resfit/dexmg/environments/dexmg.py at their commit 66262b0) makes every task with
``robosuite.make(env_name, robots=ENV_ROBOTS[env_name], controller_configs=load_composite_controller_config(robot=robots[0]),
control_freq=20, horizon=<the task's>, ...)``, a composite controller's ``ik_input_ref_frame`` set to ``world``, and
robosuite 1.5.1's defaults otherwise (sparse reward, ``reward_shaping`` off); an episode is terminated on success and
truncated at the horizon. Theirs observes cameras and the robot state (:data:`ROBOT_KEYS`); this one observes the same
robot state then the objects' (which theirs saw only through the cameras):

* ``object`` -- robosuite's ``object-state`` (robomimic's low-dim ``object``: Can's and Square's object poses and their
  poses relative to the gripper), as the robomimic low-dim datasets store it;
* ``object_joints`` -- the DexMimicGen tasks publish no object observable: the position of every joint that is not the
  robot's or its grippers' (a free object's position and quaternion, an articulated part's angle), read from the sim, so
  a recorded MuJoCo state gives exactly what the env gives (``my_project/demos/robomimic_hdf5.py`` replays them).

Each of the ``scene.num_envs`` envs is its own robosuite env, stepped one after another on the CPU; the tensors in and out
live on ``device``. Like a Direct env: ``step`` returns ``({"policy": obs}, reward, terminated, truncated, extras)``
with the finished envs already reset (same-step auto-reset: the returned observation of a finished env is its next
episode's first), ``episode_length_buf`` counts each env's steps (0 right after a reset), ``_reset_idx`` is the only
reset, and ``_get_observations`` / ``cfg.observation_space`` are the hook points ``my_project/envhooks.py`` uses.
``extras["episode_metrics_to_log"]["episode/success"]`` holds each env's last episode outcome (read only for the envs
that finished this step).
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import gymnasium as gym
import numpy as np
import torch
from isaaclab.utils import configclass

__all__ = [
    "RobosuiteTaskCfg", "RobosuiteCanCfg", "RobosuiteSquareCfg", "RobosuiteBoxCleanupCfg", "RobosuiteCanSortCfg",
    "RobosuiteCoffeeCfg", "RobosuiteTaskEnv", "make_robosuite_env", "obs_row", "ROBOT_KEYS",
]

#: ResFiT's robot state per robot layout (dexmg.py _get_expected_low_dim_keys)
ROBOT_KEYS = {
    "panda_single": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
    "panda_two_arm": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos",
                      "robot1_eef_pos", "robot1_eef_quat", "robot1_gripper_qpos"],
    "humanoid": ["robot0_right_eef_pos", "robot0_right_eef_quat", "robot0_right_gripper_qpos",
                 "robot0_left_eef_pos", "robot0_left_eef_quat", "robot0_left_gripper_qpos"],
}
#: joints of the robot, its grippers and its mount: everything else is the scene's
_ROBOT_JOINT_PREFIXES = ("robot", "gripper", "mount")


def _object_joint_slices(sim) -> List[slice]:
    """The qpos slices of every joint that is not the robot's (in model order)."""
    model = sim.model
    widths = {0: 7, 1: 4, 2: 1, 3: 1}  # mjJNT_FREE, BALL, SLIDE, HINGE
    out = []
    for j in range(model.njnt):
        name = model.joint_id2name(j)
        if name is not None and name.startswith(_ROBOT_JOINT_PREFIXES):
            continue
        start = int(model.jnt_qposadr[j])
        out.append(slice(start, start + widths[int(model.jnt_type[j])]))
    return out


def obs_row(env, obs: Dict[str, np.ndarray], keys: Sequence[str]) -> np.ndarray:
    """One robosuite env's observation row: ``keys`` from its observation dict (``object`` is robosuite's
    ``object-state``), ``object_joints`` from its sim (:func:`_object_joint_slices`)."""
    out = []
    for key in keys:
        if key == "object_joints":
            qpos = env.sim.data.qpos
            value = np.concatenate([qpos[s] for s in _object_joint_slices(env.sim)])
        elif key == "object":
            value = obs["object-state"]
        else:
            value = obs[key]
        out.append(np.asarray(value, dtype=np.float32).reshape(-1))
    return np.concatenate(out)


def make_robosuite_env(cfg: "RobosuiteTaskCfg"):
    """One robosuite env as ResFiT's ``RobosuiteGymWrapper`` makes it (no cameras; the horizon is counted outside)."""
    import robosuite as suite
    from robosuite.controllers import load_composite_controller_config

    if cfg.import_module:
        import importlib

        importlib.import_module(cfg.import_module)  # registers the DexMimicGen envs
    robots = list(cfg.robots)
    controller = load_composite_controller_config(robot=robots[0])
    if "composite_controller_specific_configs" in controller:
        controller["composite_controller_specific_configs"]["ik_input_ref_frame"] = "world"
    return suite.make(
        env_name=cfg.env_name,
        robots=robots,
        controller_configs=controller,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        ignore_done=True,  # the horizon is counted by the batched env (episode_length_buf), as a Direct env does
        control_freq=int(cfg.control_freq),
        horizon=int(cfg.horizon),
        reward_shaping=False,
        lite_physics=bool(cfg.lite_physics),
    )


@configclass
class RobosuiteSceneCfg:
    num_envs: int = 1


@configclass
class RobosuiteTaskCfg:
    seed: int = 42
    scene: RobosuiteSceneCfg = RobosuiteSceneCfg()
    #: the robosuite env (``robosuite.make``'s env_name)
    env_name: str = "PickPlaceCan"
    #: its robots (ResFiT's ENV_ROBOTS)
    robots: List[str] = ["Panda"]
    #: a module to import before making it ("dexmimicgen" registers the DexMimicGen envs; "" none)
    import_module: str = ""
    #: steps per episode (ResFiT's horizon for the task): truncated after it
    horizon: int = 200
    #: control steps per second
    control_freq: int = 20
    #: robosuite's lite_physics (ResFiT leaves robosuite 1.5.1's default, True)
    lite_physics: bool = True
    #: the observation's terms, in order (module docstring)
    obs_keys: List[str] = ROBOT_KEYS["panda_single"] + ["object"]
    #: end the episode on the step the task succeeds (ResFiT: terminated = reward == 1)
    terminate_on_success: bool = True
    #: the observation's width (checked against obs_keys at the env's first reset)
    observation_space: int = 23
    action_space: int = 7
    state_space: int = 0


@configclass
class RobosuiteCanCfg(RobosuiteTaskCfg):
    pass


@configclass
class RobosuiteSquareCfg(RobosuiteTaskCfg):
    env_name: str = "NutAssemblySquare"
    horizon: int = 300


@configclass
class RobosuiteBoxCleanupCfg(RobosuiteTaskCfg):
    env_name: str = "TwoArmBoxCleanup"
    robots: List[str] = ["PandaDexRH", "PandaDexLH"]
    import_module: str = "dexmimicgen"
    horizon: int = 300
    obs_keys: List[str] = ROBOT_KEYS["panda_two_arm"] + ["object_joints"]
    observation_space: int = 52
    action_space: int = 24


@configclass
class RobosuiteCanSortCfg(RobosuiteTaskCfg):
    env_name: str = "TwoArmCanSortRandom"
    robots: List[str] = ["GR1FixedLowerBody"]
    import_module: str = "dexmimicgen"
    horizon: int = 400
    obs_keys: List[str] = ROBOT_KEYS["humanoid"] + ["object_joints"]
    observation_space: int = 57
    action_space: int = 24


@configclass
class RobosuiteCoffeeCfg(RobosuiteTaskCfg):
    env_name: str = "TwoArmCoffee"
    robots: List[str] = ["GR1FixedLowerBody"]
    import_module: str = "dexmimicgen"
    horizon: int = 400
    obs_keys: List[str] = ROBOT_KEYS["humanoid"] + ["object_joints"]
    observation_space: int = 51
    action_space: int = 24


# MuJoCo runs these tasks, not Isaac Sim (the package's task.simulator: standalone); set after the classes, as
# configclass turns every class attribute into a field
for _cls in (RobosuiteTaskCfg, RobosuiteCanCfg, RobosuiteSquareCfg, RobosuiteBoxCleanupCfg, RobosuiteCanSortCfg, RobosuiteCoffeeCfg):
    _cls.SIMULATOR = "standalone"


class RobosuiteTaskEnv(gym.Env):
    """``scene.num_envs`` robosuite envs of one task as one batched, Direct-env-shaped env (module docstring)."""

    is_vector_env = True
    metadata = {"render_modes": [None]}

    def __init__(self, cfg: RobosuiteTaskCfg, render_mode: str | None = None, **kwargs: Any) -> None:
        self.cfg = cfg
        self.num_envs = int(cfg.scene.num_envs)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.max_episode_length = int(cfg.horizon)
        self.step_dt = 1.0 / float(cfg.control_freq)
        self.render_mode = render_mode
        # robosuite 1.5.1 has no seed argument: its placement samplers draw from NumPy's global generator
        np.random.seed(int(cfg.seed))
        self.envs = [make_robosuite_env(cfg) for _ in range(self.num_envs)]
        if int(self.envs[0].action_dim) != int(cfg.action_space):
            raise ValueError(f"cfg.action_space is {cfg.action_space}; {cfg.env_name} takes {self.envs[0].action_dim}")
        self._raw = [env.reset() for env in self.envs]
        width = int(obs_row(self.envs[0], self._raw[0], cfg.obs_keys).shape[0])
        if int(cfg.observation_space) != width:
            raise ValueError(f"cfg.observation_space is {cfg.observation_space}; {list(cfg.obs_keys)} of {cfg.env_name} are {width} wide")
        self._configure_gym_env_spaces()
        self.actions = torch.zeros((self.num_envs, int(cfg.action_space)), device=self.device)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.reset_terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.reset_time_outs = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.success = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.extras: Dict[str, Any] = {}

    # ------------------------------------------------------------------ Direct-env surface
    def _configure_gym_env_spaces(self) -> None:
        from gymnasium.vector.utils import batch_space

        cfg = self.cfg
        self.single_observation_space = gym.spaces.Dict(
            {"policy": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(int(cfg.observation_space),))}
        )
        if int(cfg.state_space):
            self.single_observation_space["critic"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(int(cfg.state_space),))
        self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(int(cfg.action_space),))
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        self.action_space = batch_space(self.single_action_space, self.num_envs)
        self.state_space = self.single_observation_space["critic"] if int(cfg.state_space) else None

    @property
    def unwrapped(self) -> "RobosuiteTaskEnv":
        return self

    def seed(self, seed: int = -1) -> int:
        if seed is not None and seed >= 0:
            np.random.seed(int(seed))
            torch.manual_seed(int(seed))
        return seed

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        rows = np.stack([obs_row(env, o, self.cfg.obs_keys) for env, o in zip(self.envs, self._raw)])
        return {"policy": torch.as_tensor(rows, dtype=torch.float32, device=self.device)}

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        ids = env_ids.tolist() if torch.is_tensor(env_ids) else list(env_ids)
        for i in ids:
            self._raw[i] = self.envs[i].reset()
        if ids:
            index = torch.as_tensor(ids, dtype=torch.long, device=self.device)
            self.episode_length_buf[index] = 0

    def reset(self, seed: int | None = None, options: Dict[str, Any] | None = None):
        if seed is not None:
            self.seed(seed)
        self._reset_idx(range(self.num_envs))
        self.extras = {}
        return self._get_observations(), self.extras

    def step(self, action: torch.Tensor):
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device).reshape(self.num_envs, -1)
        self.actions = action.clone()
        # sent as it is, as ResFiT's wrapper sends it: robosuite's controllers bound their own inputs, and its action_spec
        # is not every controller's range (GR1's finger joint targets reach 1.57 rad against a spec of +-1)
        sent = action.cpu().numpy()
        reward = np.zeros(self.num_envs, dtype=np.float32)
        success = np.zeros(self.num_envs, dtype=bool)
        for i, env in enumerate(self.envs):
            self._raw[i], r, _, _ = env.step(sent[i])
            reward[i] = float(r)
            success[i] = bool(env._check_success())
        self.episode_length_buf += 1
        success_t = torch.as_tensor(success, device=self.device)
        self.reset_terminated = success_t if self.cfg.terminate_on_success else torch.zeros_like(success_t)
        self.reset_time_outs = (self.episode_length_buf >= self.max_episode_length) & ~self.reset_terminated
        done = self.reset_terminated | self.reset_time_outs
        # an episode's outcome: did the task succeed on its last step (it ends there with terminate_on_success)
        self.success = torch.where(done, success_t.to(torch.float32), self.success)
        self.extras = {"episode_metrics_to_log": {"episode/success": self.success.clone()}}
        finished = done.nonzero(as_tuple=False).squeeze(-1)
        if finished.numel():
            self._reset_idx(finished)
        reward_t = torch.as_tensor(reward, device=self.device)
        return self._get_observations(), reward_t, self.reset_terminated.clone(), self.reset_time_outs.clone(), self.extras

    def render(self):
        return None

    def close(self) -> None:
        for env in self.envs:
            env.close()
