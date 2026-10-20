"""The project's config sections. Registered in the entry scripts' ``setup`` with
``register_section("demo", DemoCfg)``; experiment YAMLs then set ``demo.*`` fields and
the resolved config records them like any package section. Follow the package CLAUDE.md's
"Add a config class" recipe (OmegaConf-supported types, validate() for cross-field rules,
every field documented in this repo's README)."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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
    #: the BC run directory of the base policy (holds ``checkpoints/``; its resolved config,
    #: there or one level up, gives the architecture)
    base_run: Optional[str] = None
    #: the checkpoint file in ``<base_run>/checkpoints/``
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

    DEMO_ACTIONS_BEYOND_SCALE = ("store", "drop")

    def validate(self, cfg: Any) -> None:
        if self.action_scale_fraction <= 0:
            raise ValueError(f"residual.action_scale_fraction must be > 0, got {self.action_scale_fraction}")
        if self.demo_actions_beyond_scale not in self.DEMO_ACTIONS_BEYOND_SCALE:
            raise ValueError(
                f"residual.demo_actions_beyond_scale must be one of {self.DEMO_ACTIONS_BEYOND_SCALE}, "
                f"got {self.demo_actions_beyond_scale!r}"
            )
        if not self.enabled or cfg is None:
            return
        if not self.base_run:
            raise ValueError("residual.base_run is required: the BC run directory of the base policy")
        if cfg.trainer.learner == BC_LEARNER:
            raise ValueError(f"residual.enabled trains a residual with sac / ppo, not {BC_LEARNER}")
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

    #: early stopping on an EMA of the test loss; .inf never stops (the metrics still log)
    early_stop_patience: float = float("inf")
    early_stop_smooth: float = 0.9

    #: eval / debug: draw each env's predicted chunk as spheres in the scene (viewer and
    #: recorded videos): executing now purple, the rest of the executed part green, the tail red
    show_waypoints: bool = False

    #: named action-dim groups for the sampled-action MSE breakdown
    #: (``action_sample/<split>_action_mse_error_<name>``); null: the task family's own
    #: (Forge ``pos`` [0, 1, 2] / ``yaw`` [3]; FurnitureBench ``pos`` / ``rot`` [3..6] /
    #: ``gripper`` [7]). A config's entries merge over the default, so set it only to add.
    action_metric_groups: Optional[Dict[str, List[int]]] = None

    def validate(self, cfg: Any) -> None:
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
