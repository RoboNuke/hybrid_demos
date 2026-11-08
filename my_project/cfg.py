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


def check_observation_form_demos(path: str) -> None:
    """``residual.critic_state: observation``: the demo file must give the critics the observation. A raw demo episode
    file (scripts/make_raw_demos.py) does -- the demo loader builds every input through the run's action space, routed
    (one phase per learner) or not (every phase) -- as does a ready transition file in the observation form (meta
    ``form: "observation"``); refused is a ready file in another form (make_demo_transitions.py's residual critic state).
    A path that is not (yet) a file is left to the learner, which refuses it at load."""
    from pathlib import Path

    file = Path(path)
    if not file.is_file():
        return
    import torch

    from my_project.raw_demos import RAW_FORMAT

    data = torch.load(file, map_location="cpu", weights_only=False, mmap=True)
    if isinstance(data, dict) and data.get("format") == RAW_FORMAT:
        return
    form = ((data.get("meta") if isinstance(data, dict) else None) or {}).get("form")
    if form != "observation":
        raise ValueError(f"residual.critic_state: observation reads demos in the observation form; {file} is form {form!r} "
                         "(scripts/make_demo_transitions.py writes the residual's critic state): rebuild them in the "
                         "observation form, or as raw demo episodes (scripts/make_raw_demos.py)")


@dataclass
class ResidualCfg:
    """Residual RL on a frozen diffusion policy (``my_project/residual.py``).

    The learner (``trainer.learner``: sac or ppo) trains a residual: the env runs the frozen
    base, and the agent's action ``r`` in [-1, 1] per dim is added as ``scale * r`` to the base
    action (``[x, y, z, yaw]``, needs ``action.world_pose``). The agent observes the base's
    observation and the base action of the current step, both normalized with the base
    policy's normalizer.
    """

    enabled: bool = False
    #: the base policy's BC run as a local run directory (holds ``checkpoints/``; its resolved
    #: config, there or one level up, gives the architecture). Null: the wandb run
    #: ``base_project`` / ``base_run_name`` instead (one of the two, not both)
    base_run: Optional[str] = None
    #: the base policy's BC run on wandb: its project and run name (or run id), under
    #: ``base_entity`` (null: this run's ``wandb.entity``). Its ``resolved_config.yaml`` and
    #: ``base_checkpoint`` are downloaded fresh whenever an env is built. The project must not
    #: be this run's ``wandb.project``: residual runs and their bases live in separate projects
    base_project: Optional[str] = None
    base_run_name: Optional[str] = None
    base_entity: Optional[str] = None
    #: the base's checkpoint file (in ``<base_run>/checkpoints/``, or the wandb run's files)
    base_checkpoint: str = "actor_chkpt_best_success_rate.pt"
    #: largest correction per step, one value per action dim (Forge: [x, y, z, yaw] in m and
    #: rad; FurnitureBench: [dpos 3, dquat 4, gripper]); null: action_scale_fraction
    #: of the base policy's normalized action range per dim
    action_scale: Optional[List[float]] = None
    #: with action_scale null: the correction bound as a fraction of the base's normalized
    #: action range, i.e. fraction * (max - min) / 2 of its training actions per dim
    #: (robust-rearrangement's ``action_scale: 0.1``; Policy Decorator's alpha)
    action_scale_fraction: float = 0.1
    #: demo transitions (``scripts/make_demo_transitions.py``) whose residual action
    #: ``(a_demo - a_base) / scale`` lies beyond the policy's [-1, 1] in any dim: "store" them
    #: as they are (the critic takes the action as an input, so the value is still true) or
    #: "drop" them. Never clipped: a clipped action did not produce the stored next state
    demo_actions_beyond_scale: str = "store"
    #: what the SAC critics take as the action: "residual", the agent's own r in [-1, 1]
    #: (the base action reaches them through the observation), or "full", the executed pose
    #: action normalized as the base's, normalize(a_base + scale * r) = normalize(a_base) +
    #: scale / half_range * r (``residual_critic_action``); with ``steps_per_chunk`` k, the
    #: chunk's k executed actions, one row each, step-major. A demo transition's is then exactly
    #: normalize(a_demo), however far beyond the residual's reach it lies
    critic_action: str = "residual"
    #: what the agent's action is (``my_project/residual_ee.py``). "absolute": ``scale * r`` added to the
    #: base action, every dim (above). "ee" (furniture FBAbs tasks): ``r = [r_pos 3, r_rot 3]``, a correction
    #: of the base's target in the end-effector's frame -- ``ee_pos_scale * R_ee r_pos`` along the EE's axes,
    #: a rotation ``Exp(ee_rot_scale * r_rot)`` about the target's own -- the gripper the base's
    #: "target" (furniture FBAbs tasks, ``my_project/residual_target.py``): Policy Decorator's meaning of
    #: ``r`` on an absolute base, ``r = [r_pos 3, r_rot 3, r_grip 1]`` per step: a relative pose
    #: ``(ee_pos_scale * r_pos, Exp(ee_rot_scale * r_rot))`` composed on the base's target in the target's own
    #: frame, ``[p_base + R_base dp, rot6d(R_base dR)]``, and ``scale_grip * r_grip`` added to its gripper (the
    #: absolute frame's gripper scale). ManiSkill2's ``pd_ee_delta_pose`` with ``res_scale`` 0.1 is 0.01 m and
    #: 0.01 rad (0.573 deg) per unit, about the EE's axes (here the base target's, which the chunk's critic
    #: action can rebuild)
    #: "multiply": see multiply_min / multiply_reference below
    action_frame: str = "absolute"
    #: m per unit ``r_pos`` per step (action_frame: ee / target)
    ee_pos_scale: float = 0.01
    #: deg per unit ``r_rot`` per step (action_frame: ee / target)
    ee_rot_scale_deg: float = 5.0
    #: action_frame: ee: add each step's correction to an offset that resets to zero whenever the base
    #: starts a new chunk, instead of re-sending it every step; the offset (in the EE frame, over its
    #: largest reach in one chunk) joins the agent's and the critic's observation
    accumulate: bool = False
    #: env steps per agent action (Policy Decorator's residual chunk). 1: the agent acts every env
    #: step. k > 1: it acts once every k env steps with a (k x A)-wide action, step-major (row j
    #: is sub-step j's residual, A the per-step width); the wrapper runs the k env steps, each
    #: with the base's own next action plus its row, sums the rewards and ORs the dones. The
    #: base's executed ``action_horizon`` must be a multiple of k (a chunk never straddles a
    #: base re-plan), so the observed base action is the chunk's k (step-major) and
    #: ``critic_action: full`` is the chunk's k executed actions. No ``hybrid``, no demo transitions
    steps_per_chunk: int = 1
    #: override the base's executed ``model.actor.action_horizon`` (Policy Decorator's act
    #: horizon 4); null: the base run's own. At most its ``horizon``
    base_action_horizon: Optional[int] = None
    #: override the base's DDIM ``model.actor.inference_steps`` (Policy Decorator's 4); null:
    #: the base run's own. At most its ``num_diffusion_iters``
    base_inference_steps: Optional[int] = None
    #: append the base action to the agent's observation and the critic state; false: the
    #: agent sees the observation alone (Policy Decorator's ``actor_input: obs``), which rules
    #: out ``critic_action: full`` (it reads the base action there). Part of the demo fingerprint
    observe_base_action: bool = True
    #: what the SAC critics observe. "residual": the residual's own critic state (``critic_state``: the env's,
    #: or the base observation, then the base action(s)). "observation": the agent's
    #: own observation, no critic state (Policy Decorator's critic); ``critic_action: full`` then reads the
    #: base action(s) from per-transition side inputs the critics never see (``infos["residual_base"]`` /
    #: ``["next_residual_base"]``, the learner's ``side_inputs``). Furniture envs (no critic group of their
    #: own), not with ``action_frame: ee`` or demo transitions
    critic_state: str = "residual"
    #: Policy Decorator's command form for the base action the agent observes and the executed action the
    #: critics take (``action_frame: target``, ``critic_state: observation``; ``my_project/residual_target.py``
    #: ``EERelative``): each of the chunk's absolute targets against the EE pose of the chunk's observation, in the
    #: EE's frame -- ``[R_ee^T (p - p_ee), log(R_ee^T R), gripper]``, 7 per step (m, rad, raw; ManiSkill2's
    #: ``pd_ee_delta_pose``, ``frame: ee``). The observed base action (``observe_base_action``) and the critics' ``full``
    #: action are then the same form, the agent's ``k x 7`` and the critics' ``k x 7``; the critic state stays the
    #: observation without the base action (Policy Decorator's critic input). False: the absolute form
    ee_relative_actions: bool = False
    #: action_frame: target: the axes the correction is expressed in. "target": the base target's own frame (r_pos
    #: along its axes, Exp(r_rot) about them). "base": the robot-base frame the absolute targets are in (p_base +
    #: ee_pos_scale r_pos, Exp(ee_rot_scale r_rot) R_base): r = 1 is a fixed absolute correction (ee_pos_scale m,
    #: ee_rot_scale_deg deg) whatever the target's orientation. Not with ee_relative_actions
    target_axes: str = "target"
    #: override the base run's ``model.actor.warm_start`` (null keeps it): false plans every chunk from pure noise
    base_warm_start: Optional[bool] = None
    #: one_leg's task phase (``furniture.track_phases``: corner, grasp, insert, screw) from which the residual acts:
    #: an agent step whose observation's phase is earlier (or an episode-end code) runs the base alone, the agent's
    #: action zeroed, and ``infos["residual_active"]`` (1 / 0 per env) says which steps used it (set
    #: ``sac.executed_action_mask: residual_active`` so the memory stores the executed zeros, or ``sac.store_labels`` to
    #: store only the phases it acts in). "insert": only after the leg is grasped. Train and eval alike. None: every step
    active_from_phase: Optional[str] = None
    #: columns of the base's observation the residual never sees, as ``[start, stop)`` pairs: removed from the agent's
    #: observation and from the critic state (an env without a critic group of its own), never from the base's input.
    #: one_leg -FBAbs-ResiP (63): ``[25, 52]`` legs 1-3 (parts 1-3 of ``parts 5 x 9`` at column 16), ``[61, 63]`` the
    #: front obstacle. Empty: the whole observation
    observation_drop: List[List[int]] = field(default_factory=list)
    #: action_frame: target: the residual corrects the gripper too (``r_grip``, the 7th per-step dim). False: the
    #: residual is the pose alone (6 per step) and the gripper is the base's
    residual_gripper: bool = True
    #: action_frame: target: what ``r_rot`` is. "rotvec": an axis-angle, ``Exp(ee_rot_scale r_rot)``. "euler": roll,
    #: pitch, yaw, ``Rz(yaw) Ry(pitch) Rx(roll)`` of ``ee_rot_scale_deg r_rot`` (``residual_target.euler_rpy``)
    target_rotation: str = "rotvec"
    #: action_frame: target: the position's scale per axis is the absolute frame's (``action_scale``, or
    #: ``action_scale_fraction`` of the base's half range, per axis) instead of ``ee_pos_scale``: with fraction 1, r_pos
    #: is in the base's normalized action units (ResFiT's residual). The gripper's is the absolute frame's either way
    target_scale_from_base: bool = False
    #: action_frame: target: clamp the sent position and gripper to the base's normalized ``[-executed_clip,
    #: executed_clip]`` (ResFiT clamps the combined action to its normalizer's range, widened 10%: 1.1 here); the critics'
    #: executed action is rebuilt the same way. Null: no clamp. The rotation is never clamped. action_frame: absolute:
    #: clamp every sent dim to ``center +- executed_clip x`` the residual scale's half range (``action_half_range_floor``'s
    #: floored one when set): ResFiT's clamp to ``mid +- (1 + action_scale) x max(half range, floor)`` with executed_clip
    #: = action_scale_fraction; the critics take the unclamped action, as ResFiT stores it
    executed_clip: Optional[float] = None

    #: action_frame: target: the gripper command per unit ``r_grip`` (the env's own gripper units; FurnitureBench's
    #: gripper is +-1, the sign closing / opening it through the action term's hysteresis), as ee_pos_scale and
    #: ee_rot_scale_deg are for the pose. Null: the absolute frame's scale (action_scale, or action_scale_fraction of
    #: the base's gripper half range)
    ee_grip_scale: Optional[float] = None

    #: action_frame: absolute: the smallest half range of an action dim in the residual's scale (``scale =
    #: action_scale_fraction x max(half range, floor)``, the half range the base's demo dataset's own), ResFiT's
    #: ActionScaler floor (min_action_range / 2 = 0.05). Null: the base normalizer's half range, which is 1 for a
    #: dim constant in the demos (robosuite Coffee's idle fingers: 1.2 rad per unit r instead of ResFiT's 0.06).
    #: Robosuite tasks only
    action_half_range_floor: Optional[float] = None
    #: per routed phase (``ddpg.route_labels``) its own action scale, read from ``phase_stats``; the env wrapper reads the
    #: phase of every env's observation from the env's phase tracker and applies that phase's (a phase without a residual
    #: gets no correction). The critics' scale from r to the executed action rides in their side input, per transition.
    #: action_frame: absolute, critic_state: observation, critic_action: full, one step per chunk, furniture tasks
    phase_scales: bool = False
    #: phase_scales' file: every routed phase's action scale
    phase_stats: Optional[str] = None
    #: phase_scales: what one unit of r is per action dim. "half_range": ResFiT's ``action_scale_fraction x max(the
    #: phase's demo half range, action_half_range_floor or 0.05)``; "median_correction": the phase's median demo
    #: correction ``|a_demo - a_base|`` (the base run along the demos), so r is in multiples of a typical correction on
    #: every dim; a dim whose median is at most ``scale_zero_below`` (one the demos never correct, e.g. the gripper) gets
    #: scale 0, no correction
    action_scale_source: str = "half_range"
    #: action_scale_source: median_correction: a median correction at or below this gives the dim scale 0
    scale_zero_below: float = 1e-6

    #: action_frame: target, target_axes: base: the residual is a per-step increment of an offset the base's targets
    #: carry (``TargetResidual.accumulate``), as a delta-pose command's perturbation persists: each step adds to the
    #: offset, and the offset goes back to zero whenever the base replans (a new chunk, every ``action_horizon`` steps,
    #: and at an episode start). The offset, scaled by a chunk of full-reach steps, joins the agent's observation and the
    #: critics' (it is state), and the critics' side input (the executed target is rebuilt from it). False: the
    #: residual offsets the base's target for its own step only (the base's next target pulls it back)
    target_accumulate: bool = False
    #: action_frame: target: m per unit r_pos, one per axis (x, y, z), in place of ee_pos_scale (and of
    #: target_scale_from_base). Null: those
    target_pos_scale: Optional[List[float]] = None
    #: action_frame: target: deg per unit r_rot, one per axis (roll, pitch, yaw with target_rotation: euler), in place of
    #: ee_rot_scale_deg. Null: that
    target_rot_scale_deg: Optional[List[float]] = None
    #: action_frame: multiply (``my_project/residual_multiply.py``): the base's motion multiplied per dim by ``m =
    #: multiply_min + (r + 1) / 2 (multiply_max - multiply_min)`` (the gripper the base's): ``r = -1`` is ``multiply_min``,
    #: ``r = 1`` ``multiply_max``, the actor starts at ``m = 1`` (the base's own). Its observation is relative to the EE
    multiply_min: float = -0.05
    multiply_max: float = 1.25
    #: action_frame: multiply: what m multiplies. "ee": the base's target relative to the EE, in the EE's frame, sent from
    #: the EE. "increment": the base's planned step in the robot-base frame, ``base_t (-) base_{t-1}``, chained onto the
    #: current target, which is the EE at a chunk's first step (``d = base_0 (-) ee``): m = 1 is the base's trajectory,
    #: m = 0 holds the current target, and a held step never grows the next; the agent also observes the current target
    #: relative to the EE (6). The critics take the resulting target relative to the EE and the gripper (7)
    multiply_reference: str = "ee"
    #: action_frame: multiply: the warm-up's std (``ddpg.random_action_mode: actor_gaussian``) and the online
    #: exploration's, both in multiplier units (0.05: the base's step +-5%); ``ddpg.random_action_scale`` and
    #: ``ddpg.exploration_std`` are these over the multiplier per unit r, ``(multiply_max - multiply_min) / 2``
    multiply_warmup_std: float = 0.05
    multiply_exploration_std: float = 0.025

    TARGET_ROTATIONS = ("rotvec", "euler")
    DEMO_ACTIONS_BEYOND_SCALE = ("store", "drop")
    CRITIC_ACTIONS = ("residual", "full")
    ACTION_FRAMES = ("absolute", "ee", "target", "multiply")
    CRITIC_STATES = ("residual", "observation")
    PHASES = ("corner", "grasp", "insert", "screw")

    def _validate_multiply(self, cfg: Any) -> None:
        """action_frame: multiply's setup: ResFiT's observation critic, the obstacle's columns dropped, and the learner and
        the actor in the multiplier's units (the actor starts at m = 1; the warm-up and the exploration as
        multiply_warmup_std / multiply_exploration_std say)."""
        import math

        if not self.multiply_min < 1.0 < self.multiply_max:
            raise ValueError(f"residual.multiply_min ({self.multiply_min}) and multiply_max ({self.multiply_max}) must hold 1 inside")
        if self.multiply_reference not in ("ee", "increment"):
            raise ValueError(f"residual.multiply_reference must be 'ee' or 'increment', got {self.multiply_reference!r}")
        if self.multiply_warmup_std < 0 or self.multiply_exploration_std < 0:
            raise ValueError("residual.multiply_warmup_std and multiply_exploration_std must be >= 0")
        self._validate_chain("multiply")
        if cfg is None or not self.enabled:
            return
        self._validate_multiply_learner(cfg)

    def _validate_chain(self, frame: str) -> None:
        """action_frame: multiply's setup (the EE-relative observation, the observation critic, the per-phase routing)."""
        needs = {"critic_state: observation": self.critic_state == "observation", "critic_action: full": self.critic_action == "full",
                 "steps_per_chunk: 1": int(self.steps_per_chunk) == 1, "no phase_scales (no action scale)": not self.phase_scales,
                 "observe_base_action": bool(self.observe_base_action),
                 "no ee_relative_actions": not self.ee_relative_actions, "no executed_clip": self.executed_clip is None,
                 "no active_from_phase (its zeroed r is not the base's action)": self.active_from_phase is None,
                 "the obstacle's columns 61, 62 in observation_drop": {61, 62} <= {c for a, b in self.observation_drop for c in range(int(a), int(b))}}
        missing = [k for k, ok in needs.items() if not ok]
        if missing:
            raise ValueError(f"residual.action_frame: {frame} needs {missing}")

    def _validate_multiply_learner(self, cfg: Any) -> None:
        """action_frame: multiply: the learner and the actor in the multiplier's units."""
        import math

        half = (self.multiply_max - self.multiply_min) / 2.0
        neutral = (1.0 - self.multiply_min) / half - 1.0
        sec, actor = q_section(cfg), cfg.model.actor
        want = {
            "trainer.learner": (cfg.trainer.learner, "ddpg"),
            "ddpg.actor_action_scale": (float(sec.actor_action_scale), 1.0), "ddpg.action_clip": (sec.action_clip, 1.0),
            "ddpg.random_action_mode": (sec.random_action_mode, "actor_gaussian"),
            "ddpg.random_action_scale": (float(sec.random_action_scale), self.multiply_warmup_std / half),
            "ddpg.exploration_std": (float(sec.exploration_std), self.multiply_exploration_std / half),
            "ddpg.exploration_std_end": (sec.exploration_std_end, None),
            "ddpg.warmup_episodes": (int(sec.warmup_episodes), 0),
            "model.actor.output_init_scale": (float(actor.output_init_scale), 0.0),
            "model.actor.output_bias": (actor.output_bias, math.atanh(neutral)),
        }
        off = {}
        for name, (have, need) in want.items():
            same = have == need if not isinstance(need, float) or have is None else abs(float(have) - need) <= 1e-6 * max(1.0, abs(need))
            if not same:
                off[name] = (have, need)
        if off:
            raise ValueError(
                "residual.action_frame: multiply runs the actor in the multiplier's units (m = 1 at the start; r in [-1, 1] "
                f"is [{self.multiply_min}, {self.multiply_max}], {half} per unit r); set these (have -> need): "
                + ", ".join(f"{k}: {h!r} -> {n!r}" for k, (h, n) in off.items())
            )

    def validate(self, cfg: Any) -> None:
        if self.critic_action not in self.CRITIC_ACTIONS:
            raise ValueError(f"residual.critic_action must be one of {self.CRITIC_ACTIONS}, got {self.critic_action!r}")
        if self.action_scale_fraction <= 0:
            raise ValueError(f"residual.action_scale_fraction must be > 0, got {self.action_scale_fraction}")
        if self.demo_actions_beyond_scale not in self.DEMO_ACTIONS_BEYOND_SCALE:
            raise ValueError(
                f"residual.demo_actions_beyond_scale must be one of {self.DEMO_ACTIONS_BEYOND_SCALE}, "
                f"got {self.demo_actions_beyond_scale!r}"
            )
        if self.action_frame not in self.ACTION_FRAMES:
            raise ValueError(f"residual.action_frame must be one of {self.ACTION_FRAMES}, got {self.action_frame!r}")
        if self.critic_state not in self.CRITIC_STATES:
            raise ValueError(f"residual.critic_state must be one of {self.CRITIC_STATES}, got {self.critic_state!r}")
        if self.active_from_phase is not None and self.active_from_phase not in self.PHASES:
            raise ValueError(f"residual.active_from_phase must be one of {self.PHASES} or null, got {self.active_from_phase!r}")
        if self.ee_relative_actions:
            if self.action_frame != "target" or self.critic_state != "observation":
                raise NotImplementedError(
                    "residual.ee_relative_actions is built for action_frame: target with critic_state: observation"
                )
        for pair in self.observation_drop:
            if len(pair) != 2 or not 0 <= int(pair[0]) < int(pair[1]):
                raise ValueError(f"residual.observation_drop entries are [start, stop) column pairs, got {list(pair)}")
        if not self.residual_gripper and self.action_frame != "target":
            raise ValueError("residual.residual_gripper: false drops r_grip from the target-frame residual: set action_frame: target")
        if self.target_rotation not in self.TARGET_ROTATIONS:
            raise ValueError(f"residual.target_rotation must be one of {self.TARGET_ROTATIONS}, got {self.target_rotation!r}")
        target_only = {"target_rotation": self.target_rotation != "rotvec", "target_scale_from_base": self.target_scale_from_base,
                       "ee_grip_scale": self.ee_grip_scale is not None, "target_accumulate": self.target_accumulate,
                       "target_pos_scale": self.target_pos_scale is not None,
                       "target_rot_scale_deg": self.target_rot_scale_deg is not None}
        used = sorted(name for name, on in target_only.items() if on)
        if used and (self.action_frame != "target" or self.ee_relative_actions):
            raise ValueError(f"residual.{used} shape action_frame: target's residual: set action_frame: target, without ee_relative_actions")
        if self.action_scale_source not in ("half_range", "median_correction"):
            raise ValueError(f"residual.action_scale_source is half_range or median_correction, got {self.action_scale_source!r}")
        if self.phase_scales:
            if not self.phase_stats:
                raise ValueError("residual.phase_scales reads and writes its per-phase stats at residual.phase_stats: set it")
            needs = {"action_frame: absolute": self.action_frame == "absolute", "critic_state: observation": self.critic_state == "observation",
                     "critic_action: full": self.critic_action == "full",
                     "steps_per_chunk: 1": int(self.steps_per_chunk) == 1, "action_scale: null": self.action_scale is None,
                     "no executed_clip": self.executed_clip is None}
            missing = [k for k, ok in needs.items() if not ok]
            if missing:
                raise ValueError(f"residual.phase_scales needs {missing}")
        elif self.action_scale_source != "half_range":
            raise ValueError("residual.action_scale_source: median_correction is phase_scales' (per-phase) scale: set phase_scales")
        if self.action_half_range_floor is not None:
            if not float(self.action_half_range_floor) > 0:
                raise ValueError(f"residual.action_half_range_floor must be > 0 or null, got {self.action_half_range_floor}")
            if self.action_frame != "absolute" or self.action_scale is not None:
                raise ValueError("residual.action_half_range_floor floors action_frame: absolute's half-range scale (action_scale: null)")
        if self.enabled and cfg is not None and (self.phase_scales or self.action_half_range_floor is not None):
            from my_project.families import FURNITURE, ROBOSUITE, family_of

            family = family_of(cfg.task_name)
            if self.phase_scales:
                if family != FURNITURE:
                    raise NotImplementedError("residual.phase_scales reads one_leg's phase tracker: furniture tasks only")
                sec = q_section(cfg)
                if not list(getattr(sec, "route_labels", []) or []):
                    raise ValueError("residual.phase_scales scales each routed phase: set the learner's route_labels")
            # robosuite and furniture tasks only (the floor acts on action_frame: absolute's scale only); Forge neither
            if family not in (ROBOSUITE, FURNITURE):
                raise NotImplementedError("residual.phase_scales / action_half_range_floor are for robosuite and furniture tasks")
        for name in ("target_pos_scale", "target_rot_scale_deg"):
            value = getattr(self, name)
            if value is not None and (len(value) != 3 or min(float(v) for v in value) <= 0):
                raise ValueError(f"residual.{name} is 3 positive values (one per axis), got {value}")
        if self.target_pos_scale is not None and self.target_scale_from_base:
            raise ValueError("residual.target_pos_scale and target_scale_from_base both set the position scale: pick one")
        if self.target_accumulate:
            if self.target_axes != "base":
                raise ValueError("residual.target_accumulate adds up the correction in the robot-base frame: set target_axes: base")
            if self.steps_per_chunk != 1 or self.critic_state != "observation":
                raise NotImplementedError("residual.target_accumulate is built for steps_per_chunk: 1 and critic_state: observation")
        if self.action_frame == "multiply":
            self._validate_multiply(cfg)
        if self.ee_grip_scale is not None and self.ee_grip_scale <= 0:
            raise ValueError(f"residual.ee_grip_scale must be > 0 or null, got {self.ee_grip_scale}")
        if self.executed_clip is not None and self.executed_clip <= 0:
            raise ValueError(f"residual.executed_clip must be > 0 or null, got {self.executed_clip}")
        if self.executed_clip is not None and (self.action_frame not in ("target", "absolute") or self.ee_relative_actions):
            raise ValueError("residual.executed_clip clamps action_frame: target's or absolute's executed action (without ee_relative_actions)")
        if self.target_axes not in ("target", "base"):
            raise ValueError(f"residual.target_axes must be 'target' or 'base', got {self.target_axes!r}")
        if self.target_axes != "target" and (self.action_frame != "target" or self.ee_relative_actions):
            raise ValueError("residual.target_axes: base is action_frame: target's absolute variant, without ee_relative_actions")
        if self.critic_state == "observation" and self.action_frame == "ee":
            raise NotImplementedError(
                "residual.critic_state: observation with action_frame: ee: the EE-frame critic action reads the EE "
                "pose and the offset from the residual's critic state"
            )
        if self.ee_pos_scale <= 0 or self.ee_rot_scale_deg <= 0:
            raise ValueError(f"residual.ee_pos_scale and ee_rot_scale_deg must be > 0, got {self.ee_pos_scale}, {self.ee_rot_scale_deg}")
        if self.accumulate and self.action_frame != "ee":
            raise ValueError("residual.accumulate adds up the EE-frame correction: it needs residual.action_frame: ee")
        if self.steps_per_chunk < 1:
            raise ValueError(f"residual.steps_per_chunk must be >= 1, got {self.steps_per_chunk}")
        for name in ("base_action_horizon", "base_inference_steps"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"residual.{name} must be null or >= 1, got {value}")
        if self.base_action_horizon is not None and self.base_action_horizon % self.steps_per_chunk:
            raise ValueError(
                f"residual.base_action_horizon ({self.base_action_horizon}) must be a multiple of "
                f"residual.steps_per_chunk ({self.steps_per_chunk}): a residual chunk never straddles a base re-plan"
            )
        if self.critic_action == "full" and self.steps_per_chunk > 1 and self.action_frame == "ee":
            raise NotImplementedError(
                f"residual.critic_action: full with action_frame: ee and steps_per_chunk {self.steps_per_chunk} is not "
                "supported: the EE frame turns between the chunk's sub-steps, so the executed offsets after the "
                "first sub-step are not determined by the chunk's observation"
            )
        if not self.enabled or cfg is None:
            return
        if self.steps_per_chunk > 1 and cfg.hybrid.enabled:
            raise NotImplementedError(
                f"residual.steps_per_chunk {self.steps_per_chunk} with hybrid.enabled is not supported: the chunk "
                "action carries no per-step selection and force blocks"
            )
        if self.action_frame in ("ee", "target", "multiply"):
            from my_project.families import is_abs

            if not is_abs(str(cfg.task_name)):
                raise ValueError(f"residual.action_frame: {self.action_frame} corrects an FBAbs absolute target; got task {cfg.task_name!r}")
            if cfg.hybrid.enabled:
                raise ValueError(f"residual.action_frame: {self.action_frame} has no hybrid blocks; switch hybrid off")
        if self.action_frame == "ee" and self.action_scale is not None:
            raise ValueError("residual.action_scale is the absolute frame's per-dim scale; action_frame: ee uses ee_pos_scale / ee_rot_scale_deg")
        if self.critic_state == "observation":
            from my_project.families import FORGE, family_of

            if family_of(cfg.task_name) == FORGE:
                raise NotImplementedError("residual.critic_state: observation: a Forge env has a critic state of its own")
            sac = q_section(cfg)
            if sac is not None and sac.demo_transitions:
                check_observation_form_demos(str(sac.demo_transitions))
        remote = self.base_project is not None or self.base_run_name is not None
        if bool(self.base_run) == remote:
            raise ValueError(
                "residual needs the base policy's BC run, one way: residual.base_run (a local run directory) "
                "or residual.base_project and residual.base_run_name (a wandb run); "
                f"got base_run={self.base_run!r}, base_project={self.base_project!r}, base_run_name={self.base_run_name!r}"
            )
        if remote and not (self.base_project and self.base_run_name):
            raise ValueError(
                f"residual.base_project and residual.base_run_name name the base's wandb run together; got "
                f"base_project={self.base_project!r}, base_run_name={self.base_run_name!r}"
            )
        if remote and self.base_project == cfg.wandb.project:
            raise ValueError(
                f"residual.base_project is this run's own wandb.project ({cfg.wandb.project!r}): a residual run "
                "logs to another project than its base BC run"
            )
        if cfg.trainer.learner == BC_LEARNER:
            raise ValueError(f"residual.enabled trains a residual with sac / ppo, not {BC_LEARNER}")
        if self.critic_action == "full" and cfg.trainer.learner not in Q_LEARNERS:
            raise ValueError(
                f"residual.critic_action: full is the Q critics' action input ({Q_LEARNERS}); {cfg.trainer.learner} has no Q critic"
            )
        from my_project.families import FORGE, action_layout, family_of

        family = family_of(cfg.task_name)
        if family == FORGE and not cfg.action.world_pose:
            raise ValueError("residual.enabled needs action.world_pose: the base acts in world poses")
        if family != FORGE and cfg.action.world_pose:
            raise ValueError(f"action.world_pose is a Forge interface; {cfg.task_name} acts in its own (FurnitureBench) action")
        width = action_layout(cfg.task_name)["dim"]
        if self.action_scale is not None and (
            len(self.action_scale) != width or min(self.action_scale) <= 0
        ):
            raise ValueError(
                f"residual.action_scale must be {width} positive values (one per action dim of {cfg.task_name}), "
                f"got {self.action_scale}"
            )


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
