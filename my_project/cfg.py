"""The project's config sections. Registered in the entry scripts' ``setup`` with
``register_section("demo", DemoCfg)``; experiment YAMLs then set ``demo.*`` fields and
the resolved config records them like any package section. Follow the package CLAUDE.md's
"Add a config class" recipe (OmegaConf-supported types, validate() for cross-field rules,
every field documented in this repo's README)."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

#: the package's replay learners with Q critics (robonuke_rl_core train.py's ``q_learners``): each reads its own section
Q_LEARNERS = ("sac", "ddpg")


def q_section(cfg: Any) -> Any:
    """The running Q learner's config section (``cfg.sac`` or ``cfg.ddpg``: demos, routes, discount), or
    ``cfg.sac`` when the learner has none (its defaults: no demos, no routes)."""
    learner = getattr(getattr(cfg, "trainer", None), "learner", "sac")
    return getattr(cfg, learner if learner in Q_LEARNERS else "sac", None)


@dataclass
class DemoCfg:
    """Example section -- replace with what your project actually configures."""

    #: example field, read by your project code via ``cfg.demo.reward_scale``
    reward_scale: float = 1.0

    def validate(self, cfg: Any) -> None:
        if self.reward_scale <= 0:
            raise ValueError(f"demo.reward_scale must be > 0, got {self.reward_scale}")


@dataclass
class ActionCfg:
    """What the policy's action means (``my_project/wrappers.py``). Applied by the demo
    recorder, demo replay and the BC rollouts alike."""

    #: the action is the fingertip's world pose [x, y, z, yaw] (env / robot base frame, m and
    #: rad) instead of Forge's normalized 7-d action; converted into Forge's action by a
    #: wrapper, which also makes the observation's previous action the smoothed action Forge
    #: applied, as a world pose (4 dims)
    world_pose: bool = False
    #: with world_pose: keep the previous action (the smoothed world pose the env applied, 4)
    #: at the end of the policy observation; false drops it (and Forge's own 7) entirely
    observe_previous_action: bool = True
    #: with world_pose: the position [x, y, z] is relative to the env's bolt (hole) estimate,
    #: the frame Forge itself acts in, instead of the env origin (yaw stays absolute)
    bolt_relative: bool = False

    def validate(self, cfg: Any) -> None:
        if self.bolt_relative and not self.world_pose:
            raise ValueError("action.bolt_relative is a world_pose option: set action.world_pose too")


@dataclass
class EpisodeCfg:
    """When an episode ends, beyond the task's time limit (``my_project/wrappers.py``,
    :class:`~my_project.wrappers.ResetOnSuccess`). Applied to every env the package builds
    (train, eval, debug); never to the demo recorder, replay or BC rollouts."""

    #: reset an env the step it first meets the task's own success check (Factory's
    #: ``_get_curr_successes`` at ``success_threshold``, with the wrist rotation for nut
    #: threading). Reported like the time limit (terminated and truncated), so SAC bootstraps
    #: through it: the objective is unchanged, the rest of a solved episode is just not
    #: simulated. One env only: Factory's reset assumes every env resets together.
    reset_on_success: bool = False

    def validate(self, cfg: Any) -> None:
        if self.reset_on_success and cfg is not None and cfg.task_cfg.scene.num_envs != 1:
            raise ValueError(
                f"episode.reset_on_success needs task.cfg.scene.num_envs: 1 (got {cfg.task_cfg.scene.num_envs}): "
                "Factory's _reset_idx assumes every env resets together"
            )


@dataclass
class ForceObsCfg:
    """The measured end-effector wrench in the RL agent's observation (``my_project/wrappers.py``,
    :class:`~my_project.wrappers.ForceObservation`): appended last, after the history, so the base
    policy's input never changes, and to the critic state when the env has one.

    The furniture family reads the env's joint-torque estimate (``env.wrench.filtered``, robot
    base frame); Forge its smoothed wrist sensor (``force_sensor_smooth``). Raw N and N m, not
    normalized: the learner's own normalizer sees it like any other input, and
    ``force_selection`` (``my_project/losses.py``) reads it from the stored observation.
    """

    enabled: bool = False
    #: "no_torque": force only, 3 dims; "with_torque": force then torque, 6 dims
    mode: str = "no_torque"

    MODES = {"no_torque": 3, "with_torque": 6}

    @property
    def width(self) -> int:
        return self.MODES[self.mode] if self.enabled else 0

    def validate(self, cfg: Any) -> None:
        if self.mode not in self.MODES:
            raise ValueError(f"force_obs.mode must be one of {list(self.MODES)}, got {self.mode!r}")
        if cfg is None:
            return
        terms = [t for t in getattr(getattr(cfg, "losses", None), "terms", []) or [] if t.name == "force_selection"]
        if terms and not self.enabled:
            raise ValueError("losses term force_selection reads the force from the observation: set force_obs.enabled: true")
        for term in terms:
            mode = dict(term.kwargs).get("mode", "no_torque")
            if mode != self.mode:
                raise ValueError(
                    f"losses term force_selection has mode {mode!r} but force_obs.mode is {self.mode!r}: "
                    "the loss reads the force from the observation's last columns, laid out by force_obs.mode"
                )


@dataclass
class HybridCfg:
    """Hybrid force/position control under the world-pose action (``my_project/wrappers.py``).

    The package's own controller (``ForgeControllerWrapper``: ``tau = J^T [(I - S)(K e - D v)
    + S K_f (f_d - f)]``) is attached innermost, and the world-pose action (and the residual)
    pass its blocks through untouched. The package's ``controller`` section cannot be used for
    this: it requires the selection block right after Forge's 7-wide action, and the policy's
    pose block here is the 4-wide world pose. The agent's action is

        ``[x, y, z, yaw | selection | force target]``

    with one selection bit (``> 0`` is force control) and one force target (``[-1, 1]``, times
    ``force_target_bounds``) per axis of ``force_axes``. Gains are constant: the pose gains are
    Forge's own, the force gains ``default_force_gains``. Applied to every env the package
    builds (train, eval, debug); never to the demo recorder, replay or BC rollouts.

    **On the furniture family** the env owns the controller: the task id
    ``Isaac-Furniture-OneLeg-FBAbs-ResiP-Hybrid-v0`` (Robonuke_Furniture_Bench ``mdp/hybrid.py``)
    runs the same law on joint torques, with its fields under ``task.cfg.hybrid.*``. This section
    then only says the agent's action carries the blocks, after the 10-wide abs target:
    ``[goal_pos 3, goal_rot6d 6, gripper | selection | force target]``; its own force fields
    stay at their defaults there (the config refuses them).
    """

    enabled: bool = False
    #: ``[x, y, z, Rx, Ry, Rz]`` axes the force branch may take (the package's
    #: ``controller.force_axes``); the selection and force blocks are as wide as its sum
    force_axes: List[int] = field(default_factory=lambda: [1, 1, 1, 0, 0, 0])
    #: the force action in [-1, 1] scales to +/- these, N and Nm (``controller.force_target_bounds``)
    force_target_bounds: List[float] = field(default_factory=lambda: [50.0, 50.0, 50.0, 5.0, 5.0, 5.0])
    #: K_f per axis (``controller.default_force_gains``)
    default_force_gains: List[float] = field(default_factory=lambda: [0.1, 0.1, 0.1, 0.01, 0.01, 0.01])

    #: the world pose's width, the block before the selection (Forge)
    POSE_DIMS = 4
    #: the furniture hybrid task's pose block: the abs target [goal_pos 3, goal_rot6d 6, gripper]
    FURNITURE_POSE_DIMS = 10
    #: this section's force fields: Forge only
    FORGE_FIELDS = ("force_axes", "force_target_bounds", "default_force_gains")

    def controller_cfg(self):
        """The package ``ControllerCfg`` this section attaches: pose and force, constant gains,
        Forge's native 7-wide action as the pose block."""
        from robonuke_rl_core.envs.cfg import ControllerCfg

        return ControllerCfg(
            enabled=True,
            use_pose=True,
            use_force=True,
            gain_mapping="constant",
            force_axes=list(self.force_axes),
            force_target_bounds=list(self.force_target_bounds),
            default_force_gains=list(self.default_force_gains),
        )

    @property
    def axes(self) -> int:
        return int(sum(self.force_axes))

    @property
    def selection_dims(self) -> List[int]:
        """The selection block's indices in the agent's action: the actor's Bernoulli dims."""
        return list(range(self.POSE_DIMS, self.POSE_DIMS + self.axes))

    @property
    def extra_dims(self) -> int:
        """Action dims after the world pose: the selection and force blocks."""
        return 2 * self.axes if self.enabled else 0

    # ---- either family: these read the env's own controller cfg on the furniture family
    @staticmethod
    def is_furniture(cfg: Any) -> bool:
        from my_project.families import FURNITURE, family_of

        return family_of(cfg.task_name) == FURNITURE

    def axes_for(self, cfg: Any) -> int:
        if self.is_furniture(cfg):
            return int(sum(cfg.task_cfg.hybrid.force_axes))
        return self.axes

    def pose_dims_for(self, cfg: Any) -> int:
        return self.FURNITURE_POSE_DIMS if self.is_furniture(cfg) else self.POSE_DIMS

    def selection_dims_for(self, cfg: Any) -> List[int]:
        start = self.pose_dims_for(cfg)
        return list(range(start, start + self.axes_for(cfg)))

    def extra_dims_for(self, cfg: Any) -> int:
        """The env's action dims after the base's pose block (0 when off)."""
        return 2 * self.axes_for(cfg) if self.enabled else 0

    def validate(self, cfg: Any) -> None:
        if cfg is not None and self.is_furniture(cfg):
            self._validate_furniture(cfg)
            return
        if not self.enabled:
            return
        self.controller_cfg().validate(None)  # the package's own field checks
        if cfg is None:
            return
        if not cfg.action.world_pose:
            raise ValueError("hybrid.enabled needs action.world_pose; without it use the package's controller section")
        if cfg.controller.enabled:
            raise ValueError("hybrid.enabled attaches the package's controller itself; set controller.enabled: false")
        declared = sorted(int(d) for d in (cfg.model.actor.bernoulli_action_dims or []))
        if declared != self.selection_dims:
            raise ValueError(
                f"model.actor.bernoulli_action_dims must be exactly the selection block {self.selection_dims} "
                f"(after the {self.POSE_DIMS}-wide world pose), got {declared}"
            )

    def _validate_furniture(self, cfg: Any) -> None:
        owns = hasattr(cfg.task_cfg, "hybrid")
        if self.enabled != owns:
            raise ValueError(
                f"hybrid.enabled is {self.enabled} but {cfg.task_name} "
                + ("runs the hybrid controller (task.cfg.hybrid): set hybrid.enabled: true" if owns
                   else "has no hybrid controller: use Isaac-Furniture-OneLeg-FBAbs-ResiP-Hybrid-v0")
            )
        if not self.enabled:
            return
        defaults = HybridCfg()
        changed = [f for f in self.FORGE_FIELDS if list(getattr(self, f)) != list(getattr(defaults, f))]
        if changed:
            raise ValueError(f"hybrid.{changed} are Forge fields; on {cfg.task_name} set task.cfg.hybrid.{changed[0]} instead")
        from my_project.families import is_abs

        if not is_abs(cfg.task_name):
            raise ValueError(f"hybrid on the furniture family needs the abs interface (FBAbs), got {cfg.task_name}")
        declared = sorted(int(d) for d in (cfg.model.actor.bernoulli_action_dims or []))
        if declared != self.selection_dims_for(cfg):
            raise ValueError(
                f"model.actor.bernoulli_action_dims must be exactly the selection block {self.selection_dims_for(cfg)} "
                f"(after the {self.FURNITURE_POSE_DIMS}-wide abs target), got {declared}"
            )


# ---------------------------------------------------------------------------- residual RL wrappers
#: one_leg's task phases in phase/id order (furniture_lab assembly/phases.py); ids from len(PHASE_NAMES) on are an
#: episode's end codes
PHASE_NAMES = ("corner", "grasp", "insert", "screw")
#: rotation parametrisations the wrappers read and write (``my_project/wrappers/pose_math.py``)
ROT_PARAMS = ("rotvec", "euler", "rot6d", "quat_xyzw", "quat_wxyz")


def _span(pair: Any, what: str) -> None:
    if len(pair) != 2 or not 0 <= int(pair[0]) < int(pair[1]):
        raise ValueError(f"{what} is a [start, stop) column pair, got {list(pair)}")


def _rot_span(pair: Any, param: str, what: str) -> None:
    from my_project.wrappers.pose_math import rot_width

    _span(pair, what)
    if param not in ROT_PARAMS:
        raise ValueError(f"{what}: a rotation parametrisation is one of {ROT_PARAMS}, got {param!r}")
    if int(pair[1]) - int(pair[0]) != rot_width(param):
        raise ValueError(f"{what} spans {int(pair[1]) - int(pair[0])} columns; {param} is {rot_width(param)}")


@dataclass
class BasePolicyCfg:
    """The frozen diffusion policy under a residual (``my_project/wrappers/residual_policy/base_policy.py``): innermost of
    the residual wrappers, it runs the base on the env's own observation, re-plans its chunk every ``action_horizon``
    steps or when an env's episode starts, and appends the base's action to the agent's observation (the critics see
    the observation without it). Every wrapper above reads the base's action, in its own action space."""

    enabled: bool = False
    #: the base BC run: a local run directory (``base_run``), or on wandb ``base_entity`` (default ``wandb.entity``) /
    #: ``base_project`` / ``base_run_name``, downloaded into the package's run cache
    base_run: Optional[str] = None
    base_project: Optional[str] = None
    base_run_name: Optional[str] = None
    base_entity: Optional[str] = None
    base_checkpoint: str = "actor_chkpt_best_success_rate.pt"
    #: the base's executed horizon and DDIM steps; null: the base run's own
    base_action_horizon: Optional[int] = None
    base_inference_steps: Optional[int] = None
    #: each plan warm-started from the last one's tail; null: the base run's own setting
    base_warm_start: Optional[bool] = None

    def validate(self, cfg: Any) -> None:
        if not self.enabled:
            return
        if not self.base_run and not (self.base_project and self.base_run_name):
            raise ValueError("base_policy needs base_run, or base_project and base_run_name")
        if self.base_run and (self.base_project or self.base_run_name):
            raise ValueError("base_policy: set base_run or base_project / base_run_name, not both")
        if cfg is None:
            return
        if not self.base_run and self.base_project == cfg.wandb.project:
            raise ValueError(f"base_policy.base_project is this run's own wandb.project ({cfg.wandb.project!r}): a residual "
                             "run logs to another project than its base BC run")
        if cfg.trainer.learner == BC_LEARNER:
            raise ValueError(f"base_policy.enabled trains a residual with sac / ddpg / ppo, not {BC_LEARNER}")


@dataclass
class DropColumnsCfg:
    """``my_project/wrappers/action_spaces/drop_columns.py``: removes observation columns (from the agent's and the
    critics' observation) and action dims. A dropped action dim is the base's: the agent no longer controls it, and it
    leaves the base's action too."""

    enabled: bool = False
    #: ``[start, stop)`` observation column pairs, indexed as the observation reaches this wrapper (the env's own
    #: columns: it sits right above the base policy)
    observation: List[List[int]] = field(default_factory=list)
    #: action dims, indexed in the env's action
    action: List[int] = field(default_factory=list)

    def validate(self, cfg: Any) -> None:
        for pair in self.observation:
            _span(pair, "drop_columns.observation entries")
        if len(set(int(a) for a in self.action)) != len(self.action) or any(int(a) < 0 for a in self.action):
            raise ValueError(f"drop_columns.action is a list of distinct dims >= 0, got {list(self.action)}")


@dataclass
class TargetCfg:
    """One tracked target (``target_tracking.targets``): a position-like part -- ``action_pos`` in the action, the body's
    current value at ``obs_pos`` in the observation (an arm's position in m, or a finger's joints) -- and optionally a
    rotation (``action_rot`` / ``obs_rot``, each with its parametrisation)."""

    action_pos: List[int] = field(default_factory=lambda: [0, 3])
    obs_pos: List[int] = field(default_factory=lambda: [0, 3])
    action_rot: Optional[List[int]] = None
    action_rot_param: str = "rot6d"
    obs_rot: Optional[List[int]] = None
    obs_rot_param: str = "rot6d"

    def check(self, what: str) -> None:
        _span(self.action_pos, f"{what}.action_pos")
        _span(self.obs_pos, f"{what}.obs_pos")
        if int(self.action_pos[1]) - int(self.action_pos[0]) != int(self.obs_pos[1]) - int(self.obs_pos[0]):
            raise ValueError(f"{what}: action_pos and obs_pos must be as wide")
        if (self.action_rot is None) != (self.obs_rot is None):
            raise ValueError(f"{what}: set both action_rot and obs_rot, or neither")
        if self.action_rot is not None:
            _rot_span(self.action_rot, self.action_rot_param, f"{what}.action_rot")
            _rot_span(self.obs_rot, self.obs_rot_param, f"{what}.obs_rot")


@dataclass
class TargetTrackingCfg:
    """``my_project/wrappers/action_spaces/target_tracking.py``: the action sent to the env is an absolute target per
    tracked body; the agent's action is each target's change. Each target starts at the body's current pose when an
    episode starts (and, with ``reset_at_chunk``, when the base starts a new chunk); its change per step is the base's
    own planned change (``base_t - base_{t-1}``, or ``base_0 - current`` at a reset) -- the base action the wrappers above
    see -- and the sent target is the current one moved by the agent's change: positions added, rotations ``Exp(drot)
    R`` (axis-angle in the base frame's axes). Action dims outside every target pass through as they are."""

    enabled: bool = False
    targets: List[TargetCfg] = field(default_factory=lambda: [TargetCfg()])
    #: each target starts over at the body's pose whenever the base re-plans (smol2), not only at an episode's start
    reset_at_chunk: bool = True
    #: the current targets, each relative to its body (``R^T (p_target - p)``, ``log(R^T R_target)``; a joint target
    #: minus the joints), appended to the agent's / the critics' observation
    target_in_policy: bool = True
    target_in_critic: bool = False

    def validate(self, cfg: Any) -> None:
        if not self.enabled:
            return
        if not self.targets:
            raise ValueError("target_tracking.targets is empty")
        for i, target in enumerate(self.targets):
            target.check(f"target_tracking.targets[{i}]")
        used = []
        for target in self.targets:
            for pair in (target.action_pos, target.action_rot):
                if pair is not None:
                    used += list(range(int(pair[0]), int(pair[1])))
        if len(set(used)) != len(used):
            raise ValueError("target_tracking.targets overlap in the action")


@dataclass
class PoseColumnsCfg:
    """A pose in the observation (``ee_relative.poses``): its position and rotation columns and the rotation's
    parametrisation."""

    pos: List[int] = field(default_factory=lambda: [0, 3])
    rot: List[int] = field(default_factory=lambda: [3, 9])
    param: str = "rot6d"


@dataclass
class EERelativeCfg:
    """``my_project/wrappers/observation_spaces/ee_relative.py``: the observation re-expressed relative to the EE, in the
    EE's own axes -- every listed pose ``(R_ee^T (p - p_ee), R_ee^T R)`` in its own columns and parametrisation, every
    listed vector rotated ``R_ee^T v`` -- and the EE's own columns removed. Columns are indexed as the observation
    reaches this wrapper."""

    enabled: bool = False
    ee_pos: List[int] = field(default_factory=lambda: [0, 3])
    ee_rot: List[int] = field(default_factory=lambda: [3, 9])
    ee_rot_param: str = "rot6d"
    poses: List[PoseColumnsCfg] = field(default_factory=list)
    vectors: List[List[int]] = field(default_factory=list)

    def validate(self, cfg: Any) -> None:
        if not self.enabled:
            return
        _span(self.ee_pos, "ee_relative.ee_pos")
        _rot_span(self.ee_rot, self.ee_rot_param, "ee_relative.ee_rot")
        for i, pose in enumerate(self.poses):
            _span(pose.pos, f"ee_relative.poses[{i}].pos")
            _rot_span(pose.rot, pose.param, f"ee_relative.poses[{i}].rot")
        for pair in self.vectors:
            _span(pair, "ee_relative.vectors entries")
            if int(pair[1]) - int(pair[0]) != 3:
                raise ValueError(f"ee_relative.vectors entries are 3 wide, got {list(pair)}")


@dataclass
class AdditiveResidualCfg:
    """``my_project/wrappers/residual_policy/additive.py``: the agent's ``r`` added to the base's action, ``sent = base +
    scale * r`` on every dim of the action space it is given. The critics take the sent action."""

    enabled: bool = False
    #: the scale per action dim; null: ``scale_fraction`` x the base's demo half range per dim (only right above the
    #: base policy, where the action is the base's own), or the routed phases' scales from ``phase_stats``
    scale: Optional[List[float]] = None
    #: ResFiT's rule: ``scale_fraction x max(half range, half_range_floor)`` of the base's demo actions per dim (the
    #: floor null: the base normalizer's half range)
    scale_fraction: float = 0.1
    half_range_floor: Optional[float] = None
    #: per routed phase (the learner's ``route_labels``, phases read from the env's tracker) its own scale, from this
    #: file (a phase without a residual gets scale 0); the critics' scale rides in their side input
    phase_stats: Optional[str] = None
    #: clamp every sent dim to ``center +- clip x half range`` of the base's demo actions (ResFiT's widened range); null:
    #: no clamp. Only right above the base policy
    clip: Optional[float] = None

    def validate(self, cfg: Any) -> None:
        if not self.enabled:
            return
        if self.scale is not None and (self.phase_stats is not None or any(float(v) < 0 for v in self.scale)):
            raise ValueError("additive_residual.scale: numbers >= 0, and not with phase_stats")
        if self.scale_fraction <= 0:
            raise ValueError(f"additive_residual.scale_fraction must be > 0, got {self.scale_fraction}")
        if self.half_range_floor is not None and self.half_range_floor <= 0:
            raise ValueError("additive_residual.half_range_floor must be > 0 or null")
        if self.clip is not None and self.clip <= 0:
            raise ValueError("additive_residual.clip must be > 0 or null")
        if cfg is not None and self.phase_stats is not None and not list(getattr(q_section(cfg), "route_labels", []) or []):
            raise ValueError("additive_residual.phase_stats scales the routed phases: set the learner's route_labels")


@dataclass
class MultiplyResidualCfg:
    """``my_project/wrappers/residual_policy/multiply.py``: the agent's ``r`` multiplies the base's action, ``sent = r *
    base`` on every dim of the action space it is given (drop a dim it must not scale, e.g. the gripper, with
    ``drop_columns``). r's range and start are the learner's (``ddpg.action_range``, ``model.actor.output_bias``); the
    base alone is r = 1."""

    enabled: bool = False
    #: a demo's r is ``executed / base`` per dim, 1 where ``|base|`` is under this (one number, or one per dim), then
    #: clamped to the learner's action_range
    invert_floor: List[float] = field(default_factory=lambda: [1e-6])

    def validate(self, cfg: Any) -> None:
        if not self.enabled:
            return
        if not self.invert_floor or any(float(v) <= 0 for v in self.invert_floor):
            raise ValueError("multiply_residual.invert_floor: one or more numbers > 0")
        if cfg is not None and cfg.trainer.learner != "ddpg":
            raise ValueError("multiply_residual reads r as the multiplier itself: trainer.learner: ddpg, whose action_range "
                             "sets what it can be")


def residual_wrappers_check(cfg: Any) -> None:
    """The residual wrappers together (called from :meth:`BasePolicyCfg`'s owner check in ``project_env_wrappers``):
    every one of them needs the base policy, and exactly one combine runs with it."""
    combines = [name for name in ("additive_residual", "multiply_residual") if getattr(cfg, name).enabled]
    others = [name for name in ("drop_columns", "target_tracking", "ee_relative") if getattr(cfg, name).enabled]
    if not cfg.base_policy.enabled:
        if combines or others:
            raise ValueError(f"{combines + others} need base_policy.enabled")
        return
    if len(combines) != 1:
        raise ValueError(f"base_policy runs with exactly one of additive_residual / multiply_residual enabled, got {combines}")


@dataclass
class HistoryCfg:
    """A short history of some observation terms for the RL agent (``my_project/wrappers.py``,
    :class:`~my_project.wrappers.ObservationHistory`).

    Applied outermost, after the residual wrapper, so the frozen base policy's input never
    changes: the agent's observation (and the critic state, when the env has one) gains the
    previous ``length`` values of each term, oldest first. At an episode's first step the
    history is the current value repeated.
    """

    enabled: bool = False
    #: previous steps kept per term (15 Hz control: 8 is ~0.5 s)
    length: int = 8
    #: observation term names (Forge: ``task.cfg.obs_order``, where ``fingertip_pos`` is the
    #: noisy fingertip estimate and ``ft_force`` the noisy 3-d force; FurnitureBench:
    #: ``ee_pos, ee_quat, ee_pos_vel, ee_ori_vel, gripper_width, parts_poses``)
    terms: List[str] = field(default_factory=lambda: ["fingertip_pos", "ft_force"])

    def validate(self, cfg: Any) -> None:
        if not self.enabled:
            return
        if self.length < 1:
            raise ValueError(f"history.length must be >= 1, got {self.length}")
        if not self.terms or len(set(self.terms)) != len(self.terms):
            raise ValueError(f"history.terms must be distinct obs_order names, got {list(self.terms)}")
        if cfg is None:
            return
        from my_project.families import obs_layout

        order = [name for name, _ in obs_layout(cfg.task_name, cfg.task_cfg)]
        unknown = [t for t in self.terms if t not in order]
        if unknown:
            raise ValueError(f"history.terms {unknown} are not observation terms of {cfg.task_name}: {order}")


#: the ``trainer.learner`` value of the diffusion-policy behaviour cloning loop
BC_LEARNER = "diffusion_bc"
#: the file the BC loop keeps its best rollout success rate in (``my_project/bc/train.py``):
#: what eval / debug's ``--checkpoint best`` means for a BC run (the scripts register it with
#: ``register_eval_policy(..., best_checkpoint=...)``)
BC_BEST_CHECKPOINT = "actor_chkpt_best_success_rate.pt"


@dataclass
class BCCfg:
    """Behaviour cloning of the diffusion policy (``scripts/train_bc.py``).

    Only what the training loop alone uses. It reads, and never repeats:

    * ``model.actor.*`` -- horizons, the diffusion schedule, warm start, obs clip;
    * ``trainer.total_timesteps`` -- total **gradient steps**; ``trainer.checkpoint_interval``
      -- gradient steps between numbered checkpoints (0: none); ``trainer.output_dir``;
    * ``eval.num_rollouts`` and ``task.cfg.scene.num_envs`` -- the in-training rollouts;
    * ``experiment.seed`` -- the episode split, the init and the sampling.

    ``trainer.write_interval`` is not used: metrics are published once per epoch.
    """

    #: the demo dataset directory (``scripts/record_demos.py``'s ``record.dataset_dir``);
    #: required for a BC run
    dataset: Optional[str] = None
    #: parquet column holding the per-frame observation vector
    obs_key: str = "observation.state"
    #: parquet column holding the per-frame action vector
    action_key: str = "action"
    #: training demonstrations, drawn at random from the dataset (seeded by experiment.seed);
    #: null: every episode
    num_demos: Optional[int] = 50
    #: test episodes = ceil(test_split * num_demos), drawn from the episodes not trained on;
    #: with every episode in use, that many are held out of them instead
    test_split: float = 0.05

    batch_size: int = 256
    #: gradient steps per epoch; ``trainer.total_timesteps`` must be a multiple
    steps_per_epoch: int = 1000
    lr: float = 1.0e-4
    #: AdamW weight decay, on every parameter (as the reference trains it)
    weight_decay: float = 1.0e-3
    #: a diffusers ``get_scheduler`` name, e.g. "cosine", "constant_with_warmup"
    lr_schedule: str = "cosine"
    warmup_steps: int = 1000
    #: gradient-norm clip; 0 disables (the norm is still logged)
    max_grad_norm: float = 0.0

    #: epochs between test-loss passes (also when checkpoints-by-score, rollouts, the
    #: sampled-action metric and the early stopper get their chance)
    eval_every: int = 10
    #: epochs between sampled-action MSE metrics (checked on eval epochs)
    sample_every: int = 100
    #: epochs between rollouts in the env; 0 disables rollouts (and the env is not built)
    rollout_every: int = 10
    #: rollouts run only once the test loss is below this
    rollout_loss_threshold: float = 0.1
    #: show each rollout's progress -- a tqdm bar of episode steps out of eval.num_rollouts x the
    #: episode length, an episode that ends early adding its unused steps -- and print its
    #: success rate when it ends
    rollout_progress: bool = False

    #: stop once training has run this long, hours of wall-clock time from the first epoch
    #: (checked after each epoch, whose checkpoints are saved first); null: no limit
    max_hours: Optional[float] = None
    #: keep (and upload to wandb) only actor_chkpt_best_success_rate.pt: written at the first
    #: rollout, then overwritten only when a rollout's success rate beats the best so far. No
    #: last / periodic / best-test-loss checkpoints. Needs rollouts (bc.rollout_every > 0)
    save_only_best_ckpt: bool = False

    #: how observations and actions are scaled to [-1, 1] from the demos' min / max.
    #: ``per_dim``: every dim its own range (robust-rearrangement's). ``per_unit``: every dim of one
    #: physical unit shares one range, taken over the observation AND the action together --
    #: positions (EE, parts, obstacle, the action's target), rot6d, linear velocity, angular
    #: velocity, finger width, the gripper command (``families.obs_units`` / ``action_units``):
    #: a near-constant dim stays near-constant, and the target and the EE scale alike. Needs a
    #: furniture FBAbs task
    normalization: str = "per_dim"
    #: false: the action is used as it is, never rescaled -- its stats are fixed at [-1, 1], so
    #: normalizing is the identity and a zero output is the zero action (the -FBRel- interface:
    #: hold). Every demo action must lie in [-1, 1]. Not with ``normalization: per_unit``
    normalize_action: bool = True
    #: with ``normalize_action``: scale each action dim by ``s = max(|min|, |max|)`` of the demos'
    #: actions to [-s, s] -> [-1, 1] instead of min-max, so a zero action stays a zero output (the
    #: EE-frame actions: "stay at the pose") while every dim still spans the network's range. Not
    #: with ``normalization: per_unit``
    symmetric_action: bool = False

    #: early stopping on an EMA of the test loss; .inf never stops (the metrics still log)
    early_stop_patience: float = float("inf")
    early_stop_smooth: float = 0.9

    #: eval / debug: draw each env's predicted chunk as spheres in the scene (viewer and
    #: recorded videos): executing now yellow, the executed part's waypoints still to come green
    #: and those already done grey, the tail never executed red
    show_waypoints: bool = False
    #: m: those spheres' radius (at the recorder's 240 px width one pixel is ~3.6 mm at 1 m)
    waypoint_radius: float = 0.0015

    #: named action-dim groups for the sampled-action MSE breakdown
    #: (``action_sample/<split>_action_mse_error_<name>``); null: the task family's own
    #: (Forge ``pos`` [0, 1, 2] / ``yaw`` [3]; FurnitureBench ``pos`` / ``rot`` [3..6] /
    #: ``gripper`` [7]). A config's entries merge over the default, so set it only to add.
    action_metric_groups: Optional[Dict[str, List[int]]] = None

    def validate(self, cfg: Any) -> None:
        if not self.waypoint_radius > 0:
            raise ValueError(f"bc.waypoint_radius must be > 0, got {self.waypoint_radius}")
        if cfg is None or cfg.trainer.learner != BC_LEARNER:
            return  # the section is registered everywhere but only rules a BC run
        if not self.dataset:
            raise ValueError("bc.dataset is required: the demo dataset directory to train on")
        if cfg.experiment.num_agents != 1:
            raise ValueError(
                f"trainer.learner: {BC_LEARNER} trains one policy; experiment.num_agents must "
                f"be 1, got {cfg.experiment.num_agents}"
            )
        from my_project.models.cfg import DIFFUSION_ARCHITECTURES

        if cfg.model.architecture not in DIFFUSION_ARCHITECTURES:
            raise ValueError(
                f"trainer.learner: {BC_LEARNER} needs a diffusion model.architecture "
                f"({sorted(DIFFUSION_ARCHITECTURES)}), got {cfg.model.architecture!r}"
            )
        for name in ("batch_size", "steps_per_epoch", "eval_every", "sample_every"):
            if getattr(self, name) < 1:
                raise ValueError(f"bc.{name} must be >= 1, got {getattr(self, name)}")
        for name in ("warmup_steps", "rollout_every"):
            if getattr(self, name) < 0:
                raise ValueError(f"bc.{name} must be >= 0, got {getattr(self, name)}")
        if self.save_only_best_ckpt and self.rollout_every == 0:
            raise ValueError(
                "bc.save_only_best_ckpt keeps the best rollout's checkpoint, but bc.rollout_every is 0 "
                "(no rollouts): nothing would ever be saved"
            )
        if self.normalization not in ("per_dim", "per_unit"):
            raise ValueError(f"bc.normalization must be per_dim or per_unit, got {self.normalization!r}")
        if self.symmetric_action and not self.normalize_action:
            raise ValueError("bc.symmetric_action scales the action; bc.normalize_action: false keeps it as it is. Choose one")
        if self.symmetric_action and self.normalization == "per_unit":
            raise ValueError("bc.symmetric_action scales each action dim by its own range; bc.normalization: per_unit shares them. Choose one")
        if self.normalization == "per_unit" and not self.normalize_action:
            raise ValueError(
                "bc.normalization: per_unit fits the action's range together with the observation's; "
                "bc.normalize_action: false keeps the action as it is. Choose one"
            )
        if self.normalization == "per_unit":
            from my_project.families import is_abs

            if not is_abs(str(cfg.task_name)):
                raise ValueError(
                    f"bc.normalization: per_unit needs a furniture FBAbs task (its layout names every "
                    f"dim's unit), got task {cfg.task_name!r}"
                )
        if self.max_hours is not None and self.max_hours <= 0:
            raise ValueError(f"bc.max_hours must be > 0 or null, got {self.max_hours}")
        if self.num_demos is not None and self.num_demos < 1:
            raise ValueError(f"bc.num_demos must be >= 1 or null, got {self.num_demos}")
        if not 0.0 < self.test_split < 1.0:
            raise ValueError(f"bc.test_split must be in (0, 1), got {self.test_split}")
        if self.lr <= 0 or self.weight_decay < 0 or self.max_grad_norm < 0:
            raise ValueError(
                f"bc.lr must be > 0 and bc.weight_decay, bc.max_grad_norm >= 0; got "
                f"{self.lr}, {self.weight_decay}, {self.max_grad_norm}"
            )
        if not self.early_stop_patience > 0 or not 0.0 <= self.early_stop_smooth < 1.0:
            raise ValueError(
                f"bc.early_stop_patience must be > 0 and bc.early_stop_smooth in [0, 1); got "
                f"{self.early_stop_patience}, {self.early_stop_smooth}"
            )
        from diffusers.optimization import SchedulerType

        names = [s.value for s in SchedulerType]
        if self.lr_schedule not in names:
            raise ValueError(f"bc.lr_schedule must be one of {names}, got {self.lr_schedule!r}")
        total, interval = cfg.trainer.total_timesteps, cfg.trainer.checkpoint_interval
        for name, value in (("total_timesteps", total), ("checkpoint_interval", interval)):
            if value % self.steps_per_epoch:
                raise ValueError(
                    f"trainer.{name} ({value}) counts gradient steps for {BC_LEARNER} and must "
                    f"be a multiple of bc.steps_per_epoch ({self.steps_per_epoch})"
                )
        frame = getattr(getattr(cfg.model, "actor", None), "action_frame", "absolute")
        if frame != "absolute":
            from my_project.families import is_abs
            from my_project.models.action_frame import METRIC_GROUPS

            if not is_abs(str(cfg.task_name)):
                raise ValueError(
                    f"model.actor.action_frame: {frame} reads the EE pose from an FBAbs observation and turns "
                    f"its chunk into the FBAbs absolute action; got task {cfg.task_name!r}"
                )
            if self.normalization == "per_unit":
                raise ValueError(
                    f"bc.normalization: per_unit shares the FBAbs action's ranges with the observation; "
                    f"model.actor.action_frame: {frame} predicts a different action. Use per_dim"
                )
            if self.action_metric_groups is None:  # the network's [dpos, drot, gripper]
                self.action_metric_groups = {k: list(v) for k, v in METRIC_GROUPS.items()}
        if self.action_metric_groups is None:
            from my_project.families import action_metric_groups

            self.action_metric_groups = action_metric_groups(cfg.task_name)
        for group, dims in self.action_metric_groups.items():
            if not dims or min(dims) < 0:
                raise ValueError(f"bc.action_metric_groups.{group} must list dims >= 0, got {dims}")


@dataclass
class ExpertCfg:
    """The scripted expert (``record.input: expert``, ``my_project/demos/expert.py``).

    It steers with ground truth. Distances in m, speeds in m/s, forces in N.
    """

    #: the peg first lands this far from the hole axis, on the block's top surface. The block
    #: is 40 mm across and the peg 8 mm: 10-14 mm keeps it on the surface and off the opening.
    surface_radius_min: float = 0.010
    surface_radius_max: float = 0.014
    #: the peg's bottom flies this far above the hole top before descending
    approach_clearance: float = 0.010
    descend_speed: float = 0.02
    #: contact once the force's magnitude passes this
    contact_force: float = 1.5
    slide_speed: float = 0.01
    #: the height target stays this far below the fingertip while sliding and inserting: a
    #: gentle push (about task gain x push_depth)
    push_depth: float = 0.003
    #: the same lead once the peg is in the hole (null: push_depth). With the action smoothing's
    #: lag it sets the insertion speed; it may exceed push_depth, which must stay gentle while
    #: sliding (a harder push makes friction stop the slide short of the hole)
    insert_push_depth: Optional[float] = None
    #: inserting once the peg's bottom is this far below the hole top
    drop_depth: float = 0.002

    def validate(self, cfg: Any) -> None:
        if not 0 < self.surface_radius_min <= self.surface_radius_max:
            raise ValueError(
                f"record.expert.surface_radius_min / _max must satisfy 0 < min <= max, got "
                f"{self.surface_radius_min}, {self.surface_radius_max}"
            )
        if self.insert_push_depth is not None and self.insert_push_depth <= 0:
            raise ValueError(f"record.expert.insert_push_depth must be > 0 or null, got {self.insert_push_depth}")
        for name in ("approach_clearance", "descend_speed", "contact_force", "slide_speed",
                     "push_depth", "drop_depth"):
            if getattr(self, name) <= 0:
                raise ValueError(f"record.expert.{name} must be > 0, got {getattr(self, name)}")


@dataclass
class SearchExpertCfg:
    """The searching expert (``record.input: search_expert``, ``my_project/demos/search_expert.py``).

    It sees only the noisy observations and finds the hole by a spiral search. Distances in m,
    speeds in m/s, forces in N.
    """

    #: fingertip height above the believed hole top for the approach (the peg bottom is
    #: ~32.5 mm below the fingertip, so 45 mm clears the surface)
    approach_height: float = 0.045
    descend_speed: float = 0.02
    #: contact once the 3-step mean of the noisy force's magnitude passes this (the force
    #: noise is ~1 N per axis)
    contact_force: float = 3.0
    #: the spiral covers the peg's offset in the gripper (up to 3 mm) plus the hole estimate's
    #: per-episode error (~1 mm per axis)
    search_radius: float = 0.005
    #: distance between spiral turns: small enough not to step over the hole's catch zone
    search_pitch: float = 0.0008
    search_speed: float = 0.01
    #: the height target stays this far below the fingertip while searching: a gentle push
    push_depth: float = 0.003
    #: inserting once the fingertip is this far below the contact height
    drop_depth: float = 0.002
    #: the same lead once inserting
    insert_push_depth: float = 0.003

    def validate(self, cfg: Any) -> None:
        for name in ("approach_height", "descend_speed", "contact_force", "search_radius",
                     "search_pitch", "search_speed", "push_depth", "drop_depth", "insert_push_depth"):
            if getattr(self, name) <= 0:
                raise ValueError(f"record.search_expert.{name} must be > 0, got {getattr(self, name)}")


@dataclass
class ForgeV1Cfg:
    """Replaying the forge-v1 NutThread demos in our env (``scripts/import_forge_v1.py``,
    ``my_project/demos/import_forge_v1.py``). Read only by that script."""

    #: the downloaded forge-v1 dataset (LeRobot v2.1: data/chunk-*/episode_*.parquet, meta/)
    source_dir: Optional[str] = None
    #: where the successful replays are written (DemoWriter format); required
    dataset_dir: Optional[str] = None
    #: stop once this many replays have succeeded
    num_demos: int = 50
    #: force (N) above which a recorded step counts as the first contact
    contact_force: float = 2.0
    #: a replay is kept once the env's success check has held for this many steps
    success_hold_steps: int = 5

    def validate(self, cfg: Any) -> None:
        if self.num_demos < 1 or self.success_hold_steps < 1 or self.contact_force <= 0:
            raise ValueError("forge_v1.num_demos and success_hold_steps must be >= 1, contact_force > 0")


@dataclass
class RecordCfg:
    """Teleoperated demo recording (``scripts/record_demos.py``). Read only by the recorder.

    The env is whatever ``task.*`` says (``configs/base/forge_fixed_dyn.yaml`` for fixed dynamics),
    run bare: no package wrappers. The control rate is the env's own
    (``task.cfg.sim.dt * task.cfg.decimation``).
    """

    #: where the dataset is written (created, or appended to); required to record
    dataset_dir: Optional[str] = None
    #: "gamepad" (an Xbox-style controller, GUI only), "expert" (the oracle in record.expert,
    #: which steers with ground truth) or "search_expert" (record.search_expert: noisy
    #: observations only, finds the hole by a spiral search); the experts run headless too
    input: str = "gamepad"
    expert: ExpertCfg = field(default_factory=ExpertCfg)
    search_expert: SearchExpertCfg = field(default_factory=SearchExpertCfg)
    #: fingertip speed at full stick, m/s, and yaw rate, rad/s
    max_speed_pos: float = 0.03
    max_speed_yaw: float = 0.5
    #: speed multiplier while the fine-mode button (RB) is held
    fine_scale: float = 0.25
    #: the setpoint stays within this of the fingertip (m, per axis; rad): releasing the stick
    #: stops the arm, and a blocked push is limited to about gain * leash
    leash_pos: float = 0.01
    leash_yaw: float = 0.15
    #: stick deflection below this reads as 0
    stick_dead_zone: float = 0.1
    #: sign per command axis (vx, vy, vz, yaw) in case a direction feels inverted
    axis_signs: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    #: save automatically once the env's success check has held for this many steps; 0: only
    #: the save button saves
    success_hold_steps: int = 5
    #: keep an episode that hit the env's time limit (marked unsuccessful) instead of dropping it
    save_timeouts: bool = False
    #: pace the loop to the control rate in wall-clock time (off for the expert)
    realtime: bool = True
    #: stop after this many saved episodes this session; 0: until quit
    num_episodes: int = 0

    def validate(self, cfg: Any) -> None:
        if self.input not in ("gamepad", "expert", "search_expert"):
            raise ValueError(
                f"record.input must be 'gamepad', 'expert' or 'search_expert', got {self.input!r}"
            )
        self.expert.validate(cfg)
        self.search_expert.validate(cfg)
        for name in ("max_speed_pos", "max_speed_yaw", "fine_scale", "leash_pos", "leash_yaw"):
            if getattr(self, name) <= 0:
                raise ValueError(f"record.{name} must be > 0, got {getattr(self, name)}")
        if not 0.0 <= self.stick_dead_zone < 1.0:
            raise ValueError(f"record.stick_dead_zone must be in [0, 1), got {self.stick_dead_zone}")
        if len(self.axis_signs) != 4 or any(abs(s) != 1.0 for s in self.axis_signs):
            raise ValueError(f"record.axis_signs must be four of +1 / -1, got {list(self.axis_signs)}")
        if self.success_hold_steps < 0 or self.num_episodes < 0:
            raise ValueError("record.success_hold_steps and record.num_episodes must be >= 0")
