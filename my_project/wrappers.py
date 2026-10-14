"""Project policy-interface wrappers: what the policy's action means. (What it observes is
config: ``task.cfg.obs_order``, with the extra terms of ``my_project/obs_terms.py``.)

:func:`apply_policy_wrappers` is the one entry point: the demo recorder, demo replay and the BC
rollouts all call it right after making the env, so a dataset's ``action`` and what a trained
policy sends in rollouts are the same thing.

``action.world_pose`` makes the action a **world pose** ``[x, y, z, yaw]``: the fingertip
target in the env (robot base) frame, metres and radians (yaw over Forge's [-180, 90] deg).
:class:`WorldPoseAction` converts it into Forge's 7-d action
(``my_project.demos.teleop.world_pose_to_forge_action``): relative to the env's hole estimate,
which Forge adds back so its noise cancels; roll / pitch 0 (Forge overrides them); the success
prediction -1 (unused). The env's action space becomes 4-wide; Forge's own action buffer stays
7-wide.

It also replaces the previous action Forge appends to the policy observation (its smoothed,
hole-relative, normalized 7-d action) with the **same smoothed action in world units**: the
world pose ``[x, y, z, yaw]`` Forge actually targets after its action smoothing
(``ctrl.ema_factor_range``), 4 dims, in the policy action's units. At an episode's first step
that is the pose Forge seeds its smoothing with (the fingertip's starting pose).
``action.observe_previous_action: false`` drops the previous action from the observation
altogether (Forge's 7 dims removed, nothing appended). ``action.bolt_relative`` makes the
position part relative to the env's bolt (hole) estimate instead of the env origin: Forge's own
frame, so it is only rescaled.

``hybrid.enabled`` attaches the package's force/position controller innermost
(``ForgeControllerWrapper``, Forge's 7-wide pose block first); the world pose then carries the
controller's selection and force blocks after its own 4 dims, untouched
(:class:`WorldPoseAction`'s ``extra_dims``).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import torch
from robonuke_rl_core.envs.forge.controller import ForgeControllerWrapper

__all__ = [
    "WorldPoseAction",
    "ResetOnSuccess",
    "hybrid_demo_block",
    "HybridController",
    "ObservationHistory",
    "HistoryBuffer",
    "history_index",
    "apply_policy_wrappers",
    "project_env_wrappers",
    "obs_term_slices",
]


def _rebuild_spaces(unwrapped: Any) -> None:
    """Re-derive the gym spaces from ``cfg``; keep Forge's 7-wide action buffer (the rebuild
    re-allocates it at the cfg's action width)."""
    native = unwrapped.actions
    unwrapped._configure_gym_env_spaces()
    unwrapped.actions = native


def _require_forge(unwrapped: Any, what: str) -> None:
    from isaaclab_tasks.direct.forge.forge_env import ForgeEnv

    if not isinstance(unwrapped, ForgeEnv):
        raise TypeError(f"{what} needs a Forge env; {type(unwrapped).__name__} is not a ForgeEnv")


class WorldPoseAction(gym.Wrapper):
    """Take ``[x, y, z, yaw]`` world-pose actions; send Forge its 7-d action; observe Forge's
    smoothed action as a world pose instead of Forge's normalized previous action."""

    DIMS = 4
    #: Forge's previous action: the last 7 policy-observation dims
    FORGE_DIMS = 7

    def __init__(self, env: Any, observe_previous: bool = True, bolt_relative: bool = False, extra_dims: int = 0) -> None:
        super().__init__(env)
        #: positions relative to the env's bolt estimate instead of the env origin
        self.bolt_relative = bool(bolt_relative)
        #: action dims after the world pose, passed through after Forge's 7 (``hybrid``: the
        #: controller's selection and force blocks)
        self.extra_dims = int(extra_dims)
        unwrapped = env.unwrapped
        _require_forge(unwrapped, "action.world_pose")
        self.bounds = torch.tensor(unwrapped.cfg.ctrl.pos_action_bounds, device=unwrapped.device)
        unwrapped.cfg.action_space = self.DIMS + self.extra_dims
        self.observe_previous = bool(observe_previous)
        unwrapped.cfg.observation_space += (self.DIMS if self.observe_previous else 0) - self.FORGE_DIMS
        _rebuild_spaces(unwrapped)
        self.action_space = unwrapped.action_space
        self.observation_space = unwrapped.observation_space
        self._original_get_observations = unwrapped._get_observations
        unwrapped._get_observations = self._get_observations

    def applied_pose(self) -> torch.Tensor:
        """The world pose of Forge's smoothed action buffer: what the env targets after
        smoothing (at a reset, Forge's seed)."""
        from my_project.demos.teleop import action_to_yaw

        unwrapped = self.env.unwrapped
        actions = unwrapped.actions
        position = actions[:, 0:3] * self.bounds
        if not self.bolt_relative:
            position = position + self.hole_estimate()
        return torch.cat((position, action_to_yaw(actions[:, 5:6])), dim=-1)

    def hole_estimate(self) -> torch.Tensor:
        """The env's bolt (hole) estimate, env frame: what Forge's actions are relative to."""
        unwrapped = self.env.unwrapped
        return unwrapped.fixed_pos_obs_frame + unwrapped.init_fixed_pos_obs_noise

    def policy_action(self, world_pose: torch.Tensor) -> torch.Tensor:
        """A world pose ``[x, y, z, yaw]`` (env frame) as this wrapper's action."""
        if not self.bolt_relative:
            return world_pose
        offset = torch.zeros_like(world_pose)
        offset[..., 0:3] = self.hole_estimate().to(world_pose.device).reshape(offset[..., 0:3].shape)
        return world_pose - offset

    def _get_observations(self):
        observations = self._original_get_observations()
        policy = observations["policy"][:, : -self.FORGE_DIMS]
        if self.observe_previous:
            policy = torch.cat((policy, self.applied_pose()), dim=-1)
        observations["policy"] = policy
        return observations

    def forge_action(self, pose: torch.Tensor) -> torch.Tensor:
        from my_project.demos.teleop import world_pose_to_forge_action

        unwrapped = self.env.unwrapped
        if pose.shape[-1] != self.DIMS:
            raise ValueError(f"a world-pose action is [x, y, z, yaw]; got width {pose.shape[-1]}")
        pose = pose.to(unwrapped.device)
        if self.bolt_relative:
            # relative to the estimate already: Forge's own frame, so the estimate is 0 here
            return world_pose_to_forge_action(pose, torch.zeros_like(pose[..., 0:3]), self.bounds)
        return world_pose_to_forge_action(pose, self.hole_estimate(), self.bounds)

    def step(self, action):
        if not self.extra_dims:
            return self.env.step(self.forge_action(action))
        pose, extra = action[..., : self.DIMS], action[..., self.DIMS :]
        return self.env.step(torch.cat((self.forge_action(pose), extra.to(self.env.unwrapped.device)), dim=-1))


def obs_term_slices(unwrapped: Any, task_name: str | None = None) -> dict:
    """``{term: slice}`` of each leading term of the policy observation: Forge's ``obs_order``
    with Factory's width table, or FurnitureBench's fixed layout (``my_project.families``;
    anything appended after the terms is not a term)."""
    from my_project.families import obs_term_slices_of

    if task_name is None:
        task_name = "Isaac-Furniture" if hasattr(unwrapped, "observation_manager") else "forge"
    return obs_term_slices_of(task_name, unwrapped.cfg)


def history_index(unwrapped_cfg: Any, terms, task_name: str = "forge") -> torch.Tensor:
    """Column indices of ``terms`` in the policy observation (their positions in the layout)."""
    slices = obs_term_slices(SimpleNamespace(cfg=unwrapped_cfg), task_name)
    missing = [t for t in terms if t not in slices]
    if missing:
        raise ValueError(f"history.terms {missing} are not observation terms of the env: {list(slices)}")
    return torch.cat([torch.arange(slices[t].start, slices[t].stop) for t in terms])


class HistoryBuffer:
    """The previous ``length`` values of observation columns ``index``, per env, oldest first.

    :meth:`step` takes this step's observation and the envs whose episode just began, and
    returns the flattened history to append (before this step's value joins it). A fresh env's
    history is its current value repeated. One definition for :class:`ObservationHistory` and
    the offline demo conversion.
    """

    def __init__(self, index: torch.Tensor, length: int, num_envs: int, device: Any) -> None:
        self.index = index.to(device)
        self.length = int(length)
        self.buffer = torch.zeros((int(num_envs), self.length, int(self.index.numel())), device=device)

    @property
    def width(self) -> int:
        return self.length * int(self.index.numel())

    def step(self, obs: torch.Tensor, fresh: torch.Tensor) -> torch.Tensor:
        current = obs[:, self.index]
        fresh = fresh.to(torch.bool).reshape(-1)
        if bool(fresh.any()):
            self.buffer[fresh] = current[fresh].unsqueeze(1).expand(-1, self.length, -1)
        past = self.buffer.reshape(self.buffer.shape[0], -1)
        self.buffer = torch.cat((self.buffer[:, 1:], current.unsqueeze(1)), dim=1)
        return past


class ObservationHistory(gym.Wrapper):
    """Append the previous ``history.length`` values of ``history.terms`` to the agent's
    observation and, when the env has one, to the critic state (:class:`HistoryBuffer`).

    Outermost of the project wrappers: it reads the terms at their positions in the
    observation the agent gets (with the residual on, the base's normalized observation,
    whose layout is the same), so the base policy's own input is untouched. An env whose
    episode just began gets its current value repeated. Either env family
    (``my_project/envhooks.py``).
    """

    def __init__(self, env: Any, cfg: Any) -> None:
        super().__init__(env)
        from my_project.envhooks import hooks_for

        hooks = hooks_for(env.unwrapped)
        hc = cfg.history
        self.history = HistoryBuffer(
            history_index(env.unwrapped.cfg, hc.terms, cfg.task_name), hc.length, hooks.num_envs, hooks.device
        )
        added = self.history.width
        hooks.add_obs_dims(added)
        self.observation_space = hooks.observation_space
        self.action_space = hooks.action_space
        self._hooks = hooks
        hooks.hook_observations(self._observations)
        print(f"[history] {self.history.length} previous steps of {list(hc.terms)} (+{added} dims)", flush=True)

    def _observations(self, observations):
        past = self.history.step(observations["policy"], self._hooks.fresh())
        observations["policy"] = torch.cat((observations["policy"], past), dim=-1)
        if "critic" in observations:
            observations["critic"] = torch.cat((observations["critic"], past), dim=-1)
        return observations


class ResetOnSuccess(gym.Wrapper):
    """``episode.reset_on_success``: end an env's episode the step it first succeeds.

    Wraps the env's ``_get_dones``: an env that meets the task's own success check (the one
    Factory's ``_get_rewards`` applies the same step, with the wrist rotation for nut threading)
    is reported done like the time limit, terminated and truncated, so ``DirectRLEnv.step``
    resets it through ``_reset_idx`` and SAC still bootstraps through the step. The step's
    reward and ``ep_succeeded`` are computed before that reset, so the success counts.
    """

    def __init__(self, env: Any) -> None:
        super().__init__(env)
        unwrapped = env.unwrapped
        _require_forge(unwrapped, "episode.reset_on_success")
        if unwrapped.num_envs != 1:
            raise ValueError(f"episode.reset_on_success needs one env (Factory resets all envs together); got {unwrapped.num_envs}")
        self._original_get_dones = unwrapped._get_dones
        unwrapped._get_dones = self._get_dones

    def _get_dones(self):
        terminated, time_out = self._original_get_dones()  # Factory: refreshes the state first
        unwrapped = self.env.unwrapped
        task = unwrapped.cfg_task
        success = unwrapped._get_curr_successes(task.success_threshold, check_rot=task.name == "nut_thread")
        return terminated | success, time_out | success


def apply_policy_wrappers(cfg: Any, env: Any, extra_dims: int = 0) -> Any:
    """Wrap ``env`` as the ``action`` section says. Call right after ``gym.make``.
    ``extra_dims``: action dims passed through after the world pose (``hybrid``)."""
    if getattr(cfg, "action", None) is None:
        raise ValueError("the action config section is not registered (scripts' setup())")
    if cfg.action.world_pose:
        env = WorldPoseAction(
            env,
            observe_previous=cfg.action.observe_previous_action,
            bolt_relative=cfg.action.bolt_relative,
            extra_dims=extra_dims,
        )
    elif extra_dims:
        raise ValueError("extra action dims after the world pose need action.world_pose")
    return env


class HybridController(ForgeControllerWrapper):
    """``hybrid.enabled``: the package's controller, position-controlled through resets.

    Factory's own reset closes the gripper through ``generate_ctrl_signals``, which the
    controller replaces, so the last step's selection and force target would rule the next
    episode's grasp. A reset env gets every axis back on position control and a zero force
    target; its first action sets them anew.
    """

    def _install(self) -> None:
        super()._install()
        unwrapped = self.env.unwrapped
        self._original_reset_idx = unwrapped._reset_idx
        unwrapped._reset_idx = self._reset_idx

    def _reset_idx(self, env_ids):
        self._selection[env_ids] = 0.0
        self._force_target[env_ids] = 0.0
        self._original_reset_idx(env_ids)


def hybrid_demo_block(axes: int, rows: int, device: Any) -> torch.Tensor:
    """The selection and force blocks (``axes`` wide each) of an action taken under position
    control (the demos): every selection bit -1 (position), every force target 0 (unused there)."""
    block = torch.zeros((rows, 2 * int(axes)), device=device)
    block[:, : int(axes)] = -1.0
    return block


def project_env_wrappers(cfg: Any, env: Any, task_name: str) -> Any:
    """The project's wrappers for every env the package builds (``register_env_wrapper`` in
    the scripts' ``setup``): the hybrid controller when enabled (innermost), the policy
    interface, the success reset when ``episode.reset_on_success``, the residual policy when
    enabled, then the agent's observation history when enabled (outermost, so the base never
    sees it)."""
    hybrid = getattr(cfg, "hybrid", None)
    if hybrid is None:
        raise ValueError("the hybrid config section is not registered (scripts' setup())")
    if hybrid.enabled:
        env = HybridController(env, hybrid.controller_cfg(), task_name)
    env = apply_policy_wrappers(cfg, env, extra_dims=hybrid.extra_dims)
    episode = getattr(cfg, "episode", None)
    if episode is None:
        raise ValueError("the episode config section is not registered (scripts' setup())")
    if episode.reset_on_success:
        env = ResetOnSuccess(env)
    residual = getattr(cfg, "residual", None)
    if residual is not None and residual.enabled:
        from my_project.residual import ResidualPolicyWrapper

        env = ResidualPolicyWrapper(env, cfg)
    history = getattr(cfg, "history", None)
    if history is None:
        raise ValueError("the history config section is not registered (scripts' setup())")
    if history.enabled:
        env = ObservationHistory(env, cfg)
    from my_project.residual_demos import check_demo_transitions

    check_demo_transitions(cfg)  # sac.demo_transitions made for this exact setup
    return env
