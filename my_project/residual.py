"""Residual RL on a frozen diffusion policy: the env runs the base, the agent learns a correction.

:class:`ResidualPolicyWrapper` sits outermost (``register_env_wrapper`` in every script's
``setup``), on top of the world-pose action. Every step:

1. The frozen base (:class:`~my_project.models.diffusion_policy.ChunkedBase`) gives this
   step's base action ``a_base = [x, y, z, yaw]`` from the current observation. Each env keeps
   its own chunk: a new one is predicted when ``action_horizon`` actions are used up or the
   env's episode just began. ``residual.base_action_horizon`` / ``base_inference_steps``
   override the base run's executed horizon and DDIM steps (:func:`override_base`).
2. The agent observes ``[obs, a_base]`` raw: the base's observation and the base action (:func:`residual_observation`);
   no scaling here -- the learner's ``state_preprocessor`` normalizes every input. ``residual.observe_base_action:
   false`` leaves the base action out of it. The **critic state** is the residual's own (:func:`critic_state`): the env's
   critic state or, without one, the base observation, then the base action(s); the wrapper creates a critic group for
   an env that has none. ``critic_action: full`` reads it.
3. The agent's action ``r`` (``[-1, 1]`` per dim for a squashed policy) becomes
   ``a_base + residual.action_scale * r``, sent on to the world-pose action. With
   ``hybrid.enabled`` the agent's action also holds the controller's selection and force blocks
   after its 4 pose dims; the base has none, so they go to the controller as the agent gave them.

**Chunks** (``residual.steps_per_chunk: k > 1``, Policy Decorator's residual): the agent acts once
every ``k`` env steps with a ``(k x A)`` action, step-major (row ``j`` is sub-step ``j``'s residual,
``A`` the per-step width). :meth:`ResidualPolicyWrapper.step` runs the ``k`` env steps itself, each
with the base's own next action plus its row, sums the rewards, ORs the dones and returns the last
observation. The base's executed ``action_horizon`` must be a multiple of ``k``, so a chunk never
straddles a base re-plan -- and so, at the chunk boundary, the base's next ``k`` actions are known:
the observation's base-action block is those ``k`` actions normalized, step-major
(:meth:`~my_project.models.diffusion_policy.ChunkedBase.block`; ``k x A``, the rows the residual
rows are added to), the critic state gets the same block, and ``residual.critic_action: full``
gives the critics the chunk's executed actions, ``normalize(a_base_j) + gain * r_j`` per row
(:func:`full_action_transform`). The envs step together, so an env whose episode ends at sub-step ``j < k -
1`` cannot stop: Isaac Lab resets it and it runs the chunk's remaining sub-steps on the base alone
(zero residual), their rewards, dones and metrics not counted, and its base re-plans at the next
chunk boundary so the new episode's chunks line up again; the agent's first observation of that
episode is the one after sub-step ``k - 1``. Both observation hooks (this one and the history's)
run every env step, as they must for the base's per-step actions.

``residual/*`` per-env metrics report the size of the correction (per sub-step means in a chunk).

``residual.action_space``: ``per_dim`` (above: ``scale * r`` added per action dim, any task) or ``pose``
(``my_project/pose_residual.py``, furniture absolute targets): ``r`` acts on the base's target expressed in a chosen
frame, rotation convention and parametrisation, added or multiplied, optionally through a chained target; the agent
observes the observation re-expressed in its own frame, the base's motion and -- with a target -- the current target.
The action map lives in ``my_project/action_spaces.py``, shared with the demo loader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import gymnasium as gym
import torch

__all__ = [
    "load_base_policy", "base_run_dir", "base_source", "override_base", "ResidualPolicyWrapper", "residual_scale",
    "residual_observation", "critic_state", "find_residual", "full_action_transform", "residual_scale_and_clip",
    "residual_critic_action", "residual_exploration_scale",
    "SideInputTransform", "PoseSideInput", "SIDE_INPUT",
]

#: ``residual.critic_state: observation``: the per-transition side input (the learner's ``side_inputs``) the
#: critics' executed action is rebuilt from -- this agent step's base action(s), normalized, step-major; the
#: env publishes it and its ``next_`` value in ``infos`` every agent step
SIDE_INPUT = "residual_base"

#: one_leg's task phases in phase/id order (furniture_lab assembly/phases.py); ids from len(PHASE_NAMES) on are an
#: episode's end codes
PHASE_NAMES = ("corner", "grasp", "insert", "screw")
#: how often (env steps) the latched phase-label check is read to the host and raised on (one synchronization)
LABEL_CHECK_EVERY = 100
#: the residual's per-step correction sizes it logs (``residual/<name>``), split per phase when the phases are tracked
RESIDUAL_SIZES = ("position_mm", "rot_deg", "yaw_deg", "multiplier")


def base_source(cfg: Any) -> str:
    """Where the base BC run comes from, as the logs show it: ``entity/project/run`` on wandb
    (``residual.base_project`` / ``base_run_name``) or the local ``residual.base_run``."""
    rc = cfg.residual
    if rc.base_run:
        return str(rc.base_run)
    return f"{rc.base_entity or cfg.wandb.entity}/{rc.base_project}/{rc.base_run_name}"


def base_run_dir(cfg: Any) -> Path:
    """The base BC run as a local run directory: ``residual.base_run``, or the wandb run
    downloaded (its ``resolved_config.yaml`` and ``residual.base_checkpoint``, fresh, into the
    package's run cache)."""
    rc = cfg.residual
    if rc.base_run:
        return Path(rc.base_run).expanduser()
    from robonuke_rl_core.evaluation import fetch_wandb_run

    return fetch_wandb_run(base_source(cfg), rc.base_checkpoint)


def load_base_policy(run_dir: str | Path, checkpoint: str, device: Any):
    """The frozen diffusion policy of a BC run: architecture from the run's resolved config,
    weights and normalizer stats from ``<run_dir>/checkpoints/<checkpoint>``. The checkpoint's
    training step and best success rate ride along as ``policy.checkpoint_step`` /
    ``policy.checkpoint_success_rate`` (-1 / nan when it has none)."""
    from omegaconf import OmegaConf

    from robonuke_rl_core.evaluation import resolved_config_path

    from my_project.models.cfg import DIFFUSION_ARCHITECTURES, actor_cfg_from
    from my_project.models.diffusion_policy import DiffusionPolicy

    run_dir = Path(run_dir).expanduser()
    path = run_dir / "checkpoints" / checkpoint
    if not path.is_file():
        available = sorted(p.name for p in (run_dir / "checkpoints").glob("*.pt"))
        raise FileNotFoundError(f"no base checkpoint {path}; {run_dir / 'checkpoints'} has {available}")
    resolved = OmegaConf.load(resolved_config_path(run_dir))
    if resolved.model.architecture not in DIFFUSION_ARCHITECTURES:
        raise ValueError(f"{run_dir} is not a diffusion policy run (model.architecture={resolved.model.architecture})")
    actor_cfg = actor_cfg_from(resolved.model)
    if actor_cfg.action_frame != "absolute":
        raise NotImplementedError(
            f"{run_dir}: model.actor.action_frame {actor_cfg.action_frame!r}: the residual reads the base's "
            "action normalizer as the env action's, which only holds for action_frame: absolute"
        )
    state = torch.load(path, map_location=device, weights_only=False)
    policy = DiffusionPolicy(actor_cfg, state["obs_dim"], state["action_dim"]).to(device)
    policy.load_state_dict(state["model_state_dict"])
    policy.checkpoint_step = int(state.get("global_step", -1))
    policy.checkpoint_success_rate = float(state.get("best_success_rate", float("nan")))
    return policy.eval()


def override_base(policy, rc: Any, source: str = "the base"):
    """Apply ``residual.base_action_horizon`` / ``base_inference_steps`` to a loaded base (null keeps
    the base run's own), then check the chunking: the base's executed ``action_horizon`` must be a
    multiple of ``residual.steps_per_chunk``. One call for the env wrapper and the demo conversion."""
    if rc.base_action_horizon is not None:
        if int(rc.base_action_horizon) > int(policy.horizon):
            raise ValueError(
                f"residual.base_action_horizon ({rc.base_action_horizon}) exceeds {source}'s horizon "
                f"({policy.horizon}): it executes predicted actions"
            )
        policy.action_horizon = int(rc.base_action_horizon)
    if getattr(rc, "base_warm_start", None) is not None:
        policy.cfg.warm_start = bool(rc.base_warm_start)
        print(f"[residual] base warm start {'on' if policy.cfg.warm_start else 'off'} (residual.base_warm_start)", flush=True)
    if rc.base_inference_steps is not None:
        limit = int(policy.cfg.num_diffusion_iters)
        if int(rc.base_inference_steps) > limit:
            raise ValueError(
                f"residual.base_inference_steps ({rc.base_inference_steps}) exceeds {source}'s "
                f"num_diffusion_iters ({limit})"
            )
        policy.cfg.inference_steps = int(rc.base_inference_steps)
    k = int(rc.steps_per_chunk)
    if int(policy.action_horizon) % k:
        raise ValueError(
            f"residual.steps_per_chunk ({k}) must divide {source}'s executed action_horizon "
            f"({policy.action_horizon}; residual.base_action_horizon overrides it): a residual chunk never "
            "straddles a base re-plan"
        )
    return policy


def residual_scale(policy, action_scale, fraction: float = 0.1) -> torch.Tensor:
    """``residual.action_scale``, or ``fraction`` of the base's normalized action range per
    dim (``residual.action_scale_fraction``)."""
    if action_scale is not None:
        return torch.tensor(list(action_scale), dtype=torch.float32, device=policy.device)
    normalizer = policy.normalizer
    return float(fraction) * (normalizer.action_max - normalizer.action_min) / 2


def demo_action_range(base_dir: Any) -> Dict[str, torch.Tensor]:
    """The action's min and max per column over every frame of the base's demo dataset (its resolved config's
    ``bc.dataset``, ``bc.obs_key``, ``bc.action_key``): ``action_half_range_floor``'s half range."""
    from omegaconf import OmegaConf

    from robonuke_rl_core.evaluation import resolved_config_path

    from my_project.bc.dataset import load_lerobot

    bc = OmegaConf.load(resolved_config_path(base_dir)).bc
    demos = load_lerobot(str(bc.dataset), str(bc.obs_key), str(bc.action_key))
    return {"action_min": demos.action.min(0).values, "action_max": demos.action.max(0).values}


def residual_scale_and_clip(rc: Any, policy, base_dir: Any):
    """``(scale, clip)`` of the residual's action, as the config sets them: the per-dim scale (:func:`residual_scale`, or
    with ``action_half_range_floor`` ``action_scale_fraction x max(the demos' half range, floor)``) and, for
    ``action_space: per_dim`` with ``executed_clip``, the executed action's per-dim bounds ``center +- executed_clip x
    half range`` (ResFiT clamps the combined action to its widened range), else None. The residual's inputs are never
    scaled here: the learner's ``state_preprocessor`` normalizes them."""
    scale = residual_scale(policy, rc.action_scale, rc.action_scale_fraction)
    center = (policy.normalizer.action_max + policy.normalizer.action_min) / 2
    half = (policy.normalizer.action_max - policy.normalizer.action_min) / 2
    floor = getattr(rc, "action_half_range_floor", None)
    if floor is not None:
        stats = demo_action_range(base_dir)
        device = scale.device
        half = torch.clamp((stats["action_max"] - stats["action_min"]) / 2, min=float(floor)).to(device)
        center = ((stats["action_max"] + stats["action_min"]) / 2).to(device)
        scale = float(rc.action_scale_fraction) * half
    clip = None
    if rc.action_space == "per_dim" and getattr(rc, "executed_clip", None) is not None:
        clip = (center - float(rc.executed_clip) * half, center + float(rc.executed_clip) * half)
    return scale, clip


def phase_tables(path: Any, names, routes, width: int, device: Any) -> Dict[str, torch.Tensor]:
    """``residual.phase_scales``' per-label action ``scale`` ``(L, width)`` from ``residual.phase_stats`` for every
    ``std_label_names`` label ``L``; a label without a residual gets scale 0 (no correction)."""
    stats = torch.load(path, map_location="cpu", weights_only=False)
    if list(stats.get("names", [])) != list(names):
        raise ValueError(f"residual.phase_stats {path} is for labels {stats.get('names')}, the learner's are {list(names)}")
    missing = [r for r in routes if r not in stats["phases"]]
    if missing:
        raise ValueError(f"residual.phase_stats {path} has no stats for routed phases {missing}")
    scale = torch.zeros((len(names), int(width)))
    for route in routes:
        scale[list(names).index(route)] = torch.as_tensor(stats["phases"][route]["scale"])
    return {"scale": scale.to(device), "source": stats.get("meta", {})}


def full_action_transform(base_action_slice: slice, scale: torch.Tensor, steps_per_chunk: int = 1):
    """``(critic_state, r) -> critic action`` for ``residual.critic_action: full``: the executed pose action,
    ``a_base + scale * r``, ``a_base`` raw at ``base_action_slice`` of the wrapper's critic state (:func:`critic_state`;
    the learner hands the transform the critic's own raw input) -- never the agent's observation. Action dims past the
    pose (``hybrid``'s blocks) pass through. With ``residual.steps_per_chunk: k`` the slice holds the chunk's ``k`` base
    actions and ``r`` its ``k`` rows, both step-major: the chunk's ``k`` executed actions concatenated."""
    k = int(steps_per_chunk)
    width = int(scale.numel()) * k
    scale = scale.repeat(k)  # the same per-dim scale on every row

    def transform(observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        pose = observations[..., base_action_slice] + scale * actions[..., :width]
        return torch.cat((pose, actions[..., width:]), dim=-1)

    return transform


class SideInputTransform:
    """``(critic_inputs, r, side) -> critic action`` for ``residual.critic_action: full`` with
    ``residual.critic_state: observation`` on the per-dim space: the critics observe the observation and nothing else
    (Policy Decorator's critic), and the executed action ``a_base + scale * r`` (per row of a chunk) is rebuilt from the
    raw base action(s) the env publishes as the side input :data:`SIDE_INPUT` (``side_inputs``, stored per transition,
    never a critic input). ``side_scale`` (``residual.phase_scales``): each transition's own scale, the side input's
    second ``base_width`` columns."""

    def __init__(self, steps_per_chunk: int, base_width: int, scale=None, side_scale: bool = False) -> None:
        self.side_scale = bool(side_scale)
        if self.side_scale and (scale is not None or int(steps_per_chunk) != 1):
            raise ValueError("side_scale is the per-dim space's per-phase scale: no fixed scale, one step")
        self.k = int(steps_per_chunk)
        self.base_width = int(base_width)
        self.scale = None if scale is None else scale.repeat(self.k)
        self.side_inputs = {SIDE_INPUT: self.k * self.base_width + (self.base_width if self.side_scale else 0)}

    def __call__(self, critic_inputs: torch.Tensor, actions: torch.Tensor, side) -> torch.Tensor:
        base = side[SIDE_INPUT]
        if self.side_scale:
            width = self.base_width
            return torch.cat((base[..., :width] + base[..., width : 2 * width] * actions[..., :width], actions[..., width:]), dim=-1)
        width = self.k * self.base_width
        return torch.cat((base + self.scale * actions[..., :width], actions[..., width:]), dim=-1)


class PoseSideInput:
    """``(critic_inputs, r, side) -> critic action`` for ``residual.action_space: pose``
    (:meth:`~my_project.pose_residual.PoseResidual.critic_action`): the resulting target relative to the EE and the
    gripper (7), rebuilt from the side input :data:`SIDE_INPUT`."""

    def __init__(self, pose) -> None:
        from my_project.pose_residual import CRITIC_ACTION_WIDTH

        self.pose = pose
        self.side_inputs = {SIDE_INPUT: pose.side_width}
        self.critic_action_width = CRITIC_ACTION_WIDTH

    def __call__(self, critic_inputs: torch.Tensor, actions: torch.Tensor, side) -> torch.Tensor:
        return self.pose.critic_action(side[SIDE_INPUT], actions)


def find_residual(env: Any) -> "ResidualPolicyWrapper":
    """The :class:`ResidualPolicyWrapper` in ``env``'s wrapper chain (skrl's ``_env`` or gym's ``env``)."""
    node, seen = env, set()
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        if isinstance(node, ResidualPolicyWrapper):
            return node
        node = getattr(node, "_env", None) or getattr(node, "env", None)
    raise RuntimeError("the env has no ResidualPolicyWrapper: is residual.enabled set in the run's config?")


def residual_critic_action(cfg: Any, env: Any):
    """The package's ``register_critic_action`` builder (scripts/train.py's ``setup``): with
    ``residual.enabled`` and ``residual.critic_action: full``, the Q critics take the executed
    action; otherwise None, and they take ``r``."""
    rc = getattr(cfg, "residual", None)
    if rc is None or not rc.enabled or rc.critic_action != "full":
        return None
    wrapper = find_residual(env)
    if wrapper.pose is not None:
        seen = "the observation part and the current target" if wrapper.pose.critic_current_width else "the observation part"
        print(f"[residual] action_space: pose: the critics observe {seen} and take the resulting target relative to the "
              "EE and the gripper, rebuilt from the side input", flush=True)
        return PoseSideInput(wrapper.pose)
    k, base_width = wrapper.steps_per_chunk, wrapper.base_action.shape[-1]
    if wrapper.critic_mode == "observation":
        print(
            f"[residual] critics observe the observation alone (residual.critic_state: observation) and take a_base + scale * r, "
            f"the base action(s) a side input ({k} x {base_width})",
            flush=True,
        )
        if getattr(wrapper, "phase_tables", None) is not None:
            print("[residual] residual.phase_scales: each transition's scale from r to the executed action is its phase's, "
                  "carried in the side input after the base action", flush=True)
            return SideInputTransform(k, base_width, side_scale=True)
        return SideInputTransform(k, base_width, scale=wrapper.scale)
    print(
        f"[residual] critics take the executed action: a_base + "
        f"{[round(g, 5) for g in wrapper.scale.tolist()]} * r"
        + (f", every row of the {wrapper.steps_per_chunk}-step chunk" if wrapper.steps_per_chunk > 1 else ""),
        flush=True,
    )
    return full_action_transform(wrapper.critic_base_slice, wrapper.scale, wrapper.steps_per_chunk)


def residual_exploration_scale(cfg: Any, env: Any):
    """The package's ``register_exploration_scale`` builder (every training script's ``setup``): with
    ``residual.enabled``, ``() -> (num_envs, A)``, the base's motion in r units at the current step (the action space's
    ``motion_scale``), DDPG's ``exploration_std_mode: adaptive`` factor; otherwise None."""
    rc = getattr(cfg, "residual", None)
    if rc is None or not rc.enabled:
        return None
    wrapper = find_residual(env)

    def scale() -> torch.Tensor:
        step = wrapper.space_step
        if step is None:
            raise RuntimeError("the residual's exploration scale was read before the env built its first observation")
        if wrapper.steps_per_chunk > 1:
            raise NotImplementedError("exploration_std_mode: adaptive with residual.steps_per_chunk > 1: the chunk's rows "
                                      "have no per-row motion scale")
        return step["motion_scale"]

    return scale


def observation_keep(width: int, drop, device=None) -> Optional[torch.Tensor]:
    """The base observation's columns the residual sees (``residual.observation_drop``: ``[start, stop)`` pairs
    removed), as an index tensor; None when nothing is dropped."""
    if not drop:
        return None
    gone = set()
    for start, stop in drop:
        if int(stop) > width:
            raise ValueError(f"residual.observation_drop [{start}, {stop}) runs past the {width}-wide observation")
        gone.update(range(int(start), int(stop)))
    return torch.tensor([c for c in range(width) if c not in gone], dtype=torch.long, device=device)


def residual_observation(obs: torch.Tensor, base_action: torch.Tensor, observe_base_action: bool = True,
                         keep: Optional[torch.Tensor] = None):
    """The agent's observation from the base's raw observation and its action, raw: ``([obs, a_base], a_base)`` -- the
    second part is what the critic state gets appended. The learner's ``state_preprocessor`` normalizes it.
    ``residual.observe_base_action: false`` leaves the base action out: the observation is the observation part alone
    and the second element is empty, ``(N, 0)``. ``base_action`` is one action per env ``(N, A)`` or, with
    ``residual.steps_per_chunk: k``, the chunk's ``k`` ``(N, k, A)``, laid out step-major (``N, k x A``). ``keep``
    (:func:`observation_keep`): the observation columns the agent sees."""
    part = obs if keep is None else obs[..., keep]
    if not observe_base_action:
        return part, part[..., :0]
    base = base_action.reshape(obs.shape[0], -1)
    return torch.cat((part, base), dim=-1), base


def critic_state(obs: torch.Tensor, base_action: torch.Tensor, env_critic=None, keep: Optional[torch.Tensor] = None) -> torch.Tensor:
    """The residual's critic state before its appendices, raw: the env's own critic state (``env_critic``, Forge's
    ``state_order``) or, for an env without one, the base's observation (columns ``keep``), then the base action (one
    ``(N, A)`` or the chunk's ``k``, ``(N, k, A)``), step-major. The EE offset (``accumulate``) and the history go on
    after it."""
    head = env_critic if env_critic is not None else obs
    if keep is not None and env_critic is None:
        head = head[..., keep]
    return torch.cat((head, base_action.reshape(obs.shape[0], -1)), dim=-1)


class ResidualPolicyWrapper(gym.Wrapper):
    def __init__(self, env: Any, cfg: Any) -> None:
        super().__init__(env)
        from my_project.envhooks import hooks_for
        from my_project.families import action_layout
        from my_project.models.diffusion_policy import ChunkedBase

        hooks = hooks_for(env.unwrapped)
        rc = cfg.residual
        source = base_source(cfg)
        policy = load_base_policy(base_run_dir(cfg), rc.base_checkpoint, hooks.device)
        obs_width = hooks.obs_width()
        #: the env's action past the pose (``hybrid``'s selection and force blocks)
        self.extra_dims = cfg.hybrid.extra_dims_for(cfg) if cfg.hybrid.enabled else 0
        act_width = hooks.action_width() - self.extra_dims
        if (obs_width, act_width) != (policy.obs_dim, policy.action_dim):
            raise ValueError(
                f"the base policy ({source}) was trained on obs {policy.obs_dim} / action "
                f"{policy.action_dim}; this env has obs {obs_width} / action {act_width} "
                f"(+{self.extra_dims} hybrid dims)"
            )
        override_base(policy, rc, source)
        self.base = ChunkedBase(policy, hooks.num_envs)
        #: r scales by this per dim (residual.action_scale..., action_half_range_floor). The residual's inputs go out raw:
        #: the learner's state_preprocessor normalizes them
        self.scale, self.abs_clip = residual_scale_and_clip(rc, policy, base_run_dir(cfg))
        from my_project.cfg import q_section

        section = q_section(cfg)
        #: the learner's phase labels (std_label_names), the phase of each env's observation read from the env's tracker
        self.phase_names = list(getattr(section, "std_label_names", None) or [])
        #: residual.phase_scales: per std_label_names label its action scale; each step's rows are the envs' phases'
        self.phase_tables = None
        if getattr(rc, "phase_scales", False):
            self.phase_tables = phase_tables(rc.phase_stats, self.phase_names, list(section.route_labels), policy.action_dim, hooks.device)
            self.phase_scale = torch.zeros((hooks.num_envs, policy.action_dim), device=hooks.device)
        #: residual.observe_base_action: whether the agent's observation holds the base action
        self.observe_base = bool(rc.observe_base_action)
        #: residual.observation_drop: the base observation's columns the agent and the critics see (None: all)
        self.keep = observation_keep(obs_width, list(rc.observation_drop), hooks.device)
        kept_width = obs_width if self.keep is None else int(self.keep.numel())
        dropped = obs_width - kept_width
        #: residual.steps_per_chunk: env steps per agent action (override_base checked the base's horizon)
        self.steps_per_chunk = int(rc.steps_per_chunk)
        if self.steps_per_chunk > 1 and self.extra_dims:
            raise NotImplementedError("residual.steps_per_chunk > 1 with hybrid.enabled is not supported")
        #: where a_base sits in the agent's observation (residual_observation puts it right after the base's own; the
        #: history goes on after it; None when not observed)
        self.base_action_slice = (
            slice(kept_width, kept_width + self.steps_per_chunk * policy.action_dim) if self.observe_base else None
        )
        #: the critic state (:func:`critic_state`): the env's own critic state, or the base observation, then the chunk's
        #: base actions; ``critic_action: full`` reads them there
        self.env_critic = hooks.has_critic()
        critic_head = hooks.critic_width() if self.env_critic else kept_width
        self.critic_base_slice = slice(critic_head, critic_head + self.steps_per_chunk * policy.action_dim)
        self.base_action = torch.zeros((hooks.num_envs, policy.action_dim), device=hooks.device)
        self.layout = action_layout(cfg.task_name)
        self.obs_width = obs_width
        self.num_envs = hooks.num_envs
        #: sub-steps of the last agent step that belonged to each env's episode (all of them, or
        #: those up to and including the one it ended on): what an episode's length counts
        self.counted = torch.ones((hooks.num_envs,), dtype=torch.long, device=hooks.device)
        base_width = self.steps_per_chunk * policy.action_dim
        #: residual.critic_state: "residual" (the critic group below) or "observation" (none: the critics see the
        #: agent's observation; the base action(s) go out as the side input SIDE_INPUT)
        self.critic_mode = str(rc.critic_state)
        if self.critic_mode == "observation" and self.env_critic:
            raise NotImplementedError("residual.critic_state: observation on an env with a critic state of its own")
        #: residual.active_from_phase: each env's task phase at its current observation (infos' eval_state phase/id),
        #: and the first phase the residual acts in (None: always)
        from my_project.cfg import ResidualCfg

        self.active_from = None if rc.active_from_phase is None else ResidualCfg.PHASES.index(rc.active_from_phase)
        self.phase = torch.zeros((hooks.num_envs,), dtype=torch.long, device=hooks.device)
        #: the task phase of each env's current observation, whenever the env publishes it (infos eval_state phase/id,
        #: furniture.track_phases), for the per-phase correction sizes (:meth:`_phase_metrics`); None until it has
        self.obs_phase = None
        #: the phase-label consistency check (:meth:`_check_phase_label`), latched on the device and read every
        #: LABEL_CHECK_EVERY steps: the envs that ever mismatched, and the labels they mismatched with
        self._label_mismatch = None
        self._label_mismatch_pair = None
        self._label_checks = 0

        #: residual.action_space: pose -- r acts on the base's pose targets (my_project/pose_residual.py); per
        #: std_label_names label whether it has a residual (route_labels), so a phase without one sends the base
        self.pose = None
        #: the per-step residual width A: the agent's action is (steps_per_chunk x A)
        self.width = policy.action_dim
        extra_width = base_width if self.observe_base else 0
        critic_extra = base_width
        if rc.action_space == "pose":
            from my_project.pose_residual import pose_from

            r_range = tuple(float(v) for v in section.action_range) if hasattr(section, "action_range") else None
            self.pose = pose_from(rc, r_range, self.layout["gripper"])
            # the phases with a residual: the routed ones, or every phase with one residual for all (no route_labels);
            # without phase labels every step is the residual's
            routes = list(getattr(section, "route_labels", None) or []) or list(self.phase_names)
            self.routed_rows = torch.tensor([name in routes for name in self.phase_names], dtype=torch.bool, device=hooks.device)
            self.env_routed = torch.ones((hooks.num_envs,), dtype=torch.bool, device=hooks.device)
            self.width = self.pose.width
            # the agent's additions: the base's motion and gripper, then (target) the current target
            extra_width = self.pose.observed_width + self.pose.current_width
            hooks.set_agent_action_width(self.width)
        elif self.steps_per_chunk > 1:
            hooks.set_agent_action_width(self.steps_per_chunk * self.width)
        #: the normalized base action(s) the agent's last observation was made with: the critics' side input
        self.side = torch.zeros((hooks.num_envs, self.pose.side_width if self.pose is not None else base_width), device=hooks.device)
        #: the action space (my_project/action_spaces.py): one definition of a step's inputs, its sent action and its
        #: inverse, shared with the demo loader; its per-env state (the chain's current target, the chunk's EE pose)
        self.space = action_space_of(self, policy.action_dim)
        self.space_state = self.space.initial_state(hooks.num_envs, hooks.device)
        self.space_step = None

        # the agent's observation gains what the flags ask for; the critic state is the residual's own, or none
        if self.critic_mode == "observation":
            critic_width = kept_width + (self.pose.critic_current_width if self.pose is not None else 0)
            if extra_width or dropped or critic_width != obs_width:
                # the critics keep the observation without the agent's additions (Policy Decorator's critic input),
                # plus the current target with action_target: target (state)
                hooks.create_critic(critic_width)
                hooks.widen(extra_width - dropped, 0)
        elif self.env_critic:
            hooks.widen(extra_width - dropped, critic_extra)
        else:
            hooks.create_critic(kept_width + critic_extra)
            hooks.widen(extra_width - dropped, 0)
        self.observation_space = hooks.observation_space
        self.action_space = hooks.action_space
        self._hooks = hooks
        hooks.hook_observations(self._observations)
        pose = self.pose
        print(
            f"[residual] base {source}/{rc.base_checkpoint} (step {getattr(policy, 'checkpoint_step', -1)}, "
            f"best success rate {getattr(policy, 'checkpoint_success_rate', float('nan')):.3f}; executes "
            f"{policy.action_horizon} of {policy.horizon}, {policy.cfg.inference_steps} DDIM steps)"
            + (f"; per-dim residual, scale {self.scale.tolist()}" if pose is None else "")
            + (f"; {self.steps_per_chunk} env steps per agent action ({self.steps_per_chunk} x {self.width})"
               if self.steps_per_chunk > 1 else "")
            + "; raw inputs"
            + ("" if self.observe_base else "; base action not observed")
            + (f"; observation columns {list(rc.observation_drop)} dropped ({kept_width} of {obs_width} kept)" if dropped else "")
            + (f"; sent per-dim action clamped to the base's normalized +-{rc.executed_clip}" if pose is None and rc.executed_clip is not None else "")
            + ("; critics observe the observation alone" if self.critic_mode == "observation" else "")
            + (f"; pose residual: action frame {pose.action_frame} ({pose.action_rotation} rotation), {pose.param}, "
               f"{pose.combine}{' with a chained target (the EE at each chunk start)' if pose.target else ''}"
               + (f", {pose.scale_pos} m / {rc.step_scale_rot_deg} deg per unit r" if pose.combine == "add" else "")
               + f"; observation in the {pose.observation_frame} frame ({pose.observation_rotation} rotation); the gripper "
               f"the base's; residuals in {[n for n, on in zip(self.phase_names, self.routed_rows.tolist()) if on]}"
               if pose is not None else ""),
            flush=True,
        )

    def _phase_rows(self, n: int) -> None:
        """The envs' current phases' rows (residual.phase_scales' scale, the pose residual's routed flag), from the
        env's phase tracker, which already holds the phase of the observation being built (it updates with the
        terminations and resets with the envs, both before it). The label used is kept (``obs_label``): :meth:`step`
        checks it is the one the env then publishes for the learner (``eval_state/phase/id``) -- one source of truth."""
        tracker = getattr(self.env.unwrapped, "phases", None)
        if tracker is None:
            raise RuntimeError("the residual's per-phase rows read the env's phase tracker: set task.cfg.furniture.track_phases: true")
        phase = tracker.phase.reshape(-1).long().clamp(0, len(self.phase_names) - 1).to(self.scale.device)
        self.obs_label = phase.clone()
        if self.phase_tables is not None:
            self.phase_scale = self.phase_tables["scale"][phase]
        if self.pose is not None:
            self.env_routed = self.routed_rows[phase]

    def _observations(self, observations):
        obs = observations["policy"]
        if self.phase_tables is not None or (self.pose is not None and self.phase_names):
            self._phase_rows(obs.shape[0])
        fresh = self._hooks.fresh()
        # the envs whose base starts a new chunk this step (ChunkedBase.actions replans those): where the pose space's
        # chained target and chunk frame start over
        new_chunk = None
        if self.pose is not None:
            new_chunk = fresh.to(torch.bool).reshape(-1) | (self.base.index >= self.base.policy.action_horizon)
        self.base_action = self.base.actions(obs, fresh)
        # the observed base action: this step's, or the chunk's k (at a chunk boundary, the next k)
        observed = self.base_action if self.steps_per_chunk == 1 else self.base.block(self.steps_per_chunk)
        step, self.space_state = self.space.observe(
            obs, self.base_action, self.space_state, observed=observed,
            env_critic_obs=observations.get("critic") if self.env_critic else None, fresh=fresh, new_chunk=new_chunk,
            routed=self.env_routed if self.pose is not None else None,
            scale_rows=self.phase_scale if self.phase_tables is not None else None,
        )
        observations["policy"] = step["policy"]
        if step["critic"] is not None:
            observations["critic"] = step["critic"]
        self.side = step["side"]
        self.space_step = step
        return observations

    def step(self, action):
        action = action.to(self.scale.device)
        side = self.side
        active = None
        if self.active_from is not None:
            # the residual acts only from phase active_from on (phase ids 4-6: an episode's end, never)
            active = (self.phase >= self.active_from) & (self.phase < len(PHASE_NAMES))
            action = action * active.unsqueeze(-1).to(action.dtype)
        acted_in = self.obs_phase  # the phase of the observation this action was chosen for
        out = self._step_chunk(action) if self.steps_per_chunk > 1 else self._step_once(action)
        self._count_episode(out)
        self._phase_metrics(out[4], acted_in)
        self._check_phase_label(out[4])
        if self.active_from is not None:
            infos = out[4]
            if not isinstance(infos, dict):
                raise TypeError(f"residual.active_from_phase publishes residual_active in the env's infos dict; got {type(infos)}")
            infos["residual_active"] = active.to(torch.float32)
        if self.critic_mode == "observation":
            infos = out[4]
            if not isinstance(infos, dict):
                raise TypeError(f"residual.critic_state: observation publishes its side input in the env's infos dict; got {type(infos)}")
            infos[SIDE_INPUT], infos[f"next_{SIDE_INPUT}"] = side, self.side
        return out

    def reset(self, **kwargs):
        # a genuine reset starts every env's episode over: the agent-step count too
        if hasattr(self, "agent_steps"):
            self.agent_steps.zero_()
        if self.obs_phase is not None:
            self.obs_phase = torch.zeros_like(self.obs_phase)  # every episode starts in Corner
        return self.env.reset(**kwargs)

    def _count_episode(self, out) -> None:
        """The episode bookkeeping check: ``episode/agent_steps`` (the agent steps the ending episode got from this
        wrapper, what the learner counts) and, when the env publishes its own clock (``episode/isaac_steps``),
        ``episode/count_offset`` = Isaac's steps - steps_per_chunk x agent steps: within +-(steps_per_chunk - 1)
        when an episode ends mid-chunk (the rest of that chunk is the next episode's), 0 otherwise; never more."""
        _, _, terminated, truncated, infos = out
        if not hasattr(self, "agent_steps"):
            self.agent_steps = torch.zeros(self.num_envs, dtype=torch.long, device=terminated.device)
        self.agent_steps += 1
        ended = (terminated.reshape(-1) | truncated.reshape(-1)).to(torch.bool).to(self.agent_steps.device)
        if isinstance(infos, dict):
            metrics = infos.setdefault("episode_metrics_to_log", {})
            steps = self.agent_steps.to(torch.float32)
            metrics["episode/agent_steps"] = steps.clone()
            isaac = metrics.get("episode/isaac_steps")
            if isaac is not None:
                metrics["episode/count_offset"] = torch.as_tensor(isaac, device=steps.device).reshape(-1) - self.steps_per_chunk * steps
        self.agent_steps[ended] = 0

    def _step_chunk(self, action):
        """One agent step of ``steps_per_chunk`` env steps (the module docstring's **Chunks**)."""
        n, k, width = self.num_envs, self.steps_per_chunk, self.width
        if action.shape != (n, k * width):
            raise ValueError(
                f"residual.steps_per_chunk {k}: the agent's action is ({n}, {k} x {width}) = ({n}, {k * width}), "
                f"got {tuple(action.shape)}"
            )
        rows = action.reshape(n, k, width)
        alive = torch.ones((n,), dtype=torch.bool, device=action.device)
        counted = torch.zeros((n,), dtype=torch.long, device=action.device)
        reward_sum = terminated = truncated = None
        sums, closing_values, closing_state = {}, {}, {}
        #: the env's no-progress flag (eval_state phase/still), over the chunk: still only if every counted sub-step was
        still_all = None
        horizon = self.base.policy.action_horizon
        for j in range(k):
            closed_before = ~alive  # envs whose episode ended at an earlier sub-step of this chunk
            if j == k - 1 and bool(closed_before.any()):
                # their base re-plans at the chunk boundary (this sub-step's observation), so the
                # new episode's chunks line up
                self.base.index[closed_before] = horizon
            observations, reward, term, trunc, infos = self._step_once(rows[:, j] * alive.unsqueeze(-1).to(rows.dtype))
            term_f, trunc_f = term.reshape(n).to(torch.bool), trunc.reshape(n).to(torch.bool)
            if reward_sum is None:
                reward_sum = torch.zeros_like(reward)
                terminated = torch.zeros((n,), dtype=torch.bool, device=term_f.device)
                truncated = torch.zeros((n,), dtype=torch.bool, device=trunc_f.device)
            reward_sum = reward_sum + reward * alive.to(reward.dtype).reshape(reward.shape)
            terminated |= alive & term_f
            truncated |= alive & trunc_f
            counted += alive.to(counted.dtype)
            if isinstance(infos, dict):
                for name, value in infos.get("metrics_to_log", {}).items():
                    mask = alive.reshape((n,) + (1,) * (value.dim() - 1)).to(value.dtype)
                    sums[name] = value * mask if name not in sums else sums[name] + value * mask
                for name, value in infos.get("episode_metrics_to_log", {}).items():
                    # an env keeps the values of the sub-step its episode closed on; the others the latest
                    mask = closed_before.reshape((n,) + (1,) * (value.dim() - 1))
                    closing_values[name] = value if name not in closing_values else torch.where(mask, closing_values[name], value)
                # the env's per-step state likewise (its phase label, the no-progress flags): an env whose episode closed
                # in this chunk keeps the closing sub-step's, not the next episode's first
                for name, value in (infos.get("eval_state") or {}).items():
                    if not torch.is_tensor(value) or value.shape[:1] != (n,):
                        continue
                    mask = closed_before.reshape((n,) + (1,) * (value.dim() - 1))
                    closing_state[name] = value if name not in closing_state else torch.where(mask, closing_state[name], value)
                still = (infos.get("eval_state") or {}).get("phase/still")
                if still is not None:
                    # only the sub-steps of the chunk's own episode count (alive: not yet closed before this one)
                    flag = (torch.as_tensor(still, device=action.device).reshape(n) > 0.5) | ~alive
                    still_all = flag if still_all is None else still_all & flag
            alive &= ~(term_f | trunc_f)
        self.counted = counted
        if isinstance(infos, dict):
            if sums:
                infos["metrics_to_log"] = {
                    name: value / counted.reshape((n,) + (1,) * (value.dim() - 1)).to(value.dtype) for name, value in sums.items()
                }
            if closing_values:
                infos["episode_metrics_to_log"] = closing_values
            if closing_state:
                infos["eval_state"] = {**infos.get("eval_state", {}), **closing_state}
                if still_all is not None:
                    infos["eval_state"]["phase/still"] = still_all.to(torch.float32)
        return observations, reward_sum, terminated.reshape(term.shape).to(term.dtype), truncated.reshape(trunc.shape).to(trunc.dtype), infos

    def _step_once(self, action):
        """One env step: ``action`` is the agent's per-step action (``width`` plus the hybrid dims)."""
        width = self.base_action.shape[-1]
        base = self.base_action  # env.step's observation hook replaces it with the next step's
        step = self.space_step
        # the space's sent action (per_dim: base + scale x r, clamped by residual.executed_clip -- the critics take the
        # unclamped one; pose: the target the step's pose gives, the base's where the phase has no residual)
        sent, self.space_state = self.space.send(action, step, self.space_state)
        if self.extra_dims:
            sent = torch.cat((sent, action[:, width:]), dim=-1)
        observations, rewards, terminated, truncated, infos = self.env.step(sent)
        self._track_phase(infos)
        if isinstance(infos, dict):
            metrics = infos.setdefault("metrics_to_log", {})
            if self.pose is not None:
                metrics["residual/position_mm"] = (sent[:, 0:3] - base[:, 0:3]).norm(dim=-1) * 1000.0
                metrics["residual/rot_deg"] = rot6d_angle_deg(base[:, 3:9], sent[:, 3:9])
                if self.pose.combine == "multiply":
                    # the mean multiplier over the pose dims (1 where the phase has no residual)
                    r = action[:, : self.width]
                    metrics["residual/multiplier"] = torch.where(step["routed"].reshape(-1, 1), r, torch.ones_like(r)).mean(dim=-1)
                return observations, rewards, terminated, truncated, infos
            correction = step["correction"]
            # an action in controller units (robosuite's OSC_POSE inputs) is scaled to metres / radians first
            metrics["residual/position_mm"] = correction[:, self.layout["pos"]].norm(dim=-1) * (1000.0 * self.layout.get("pos_m_per_unit", 1.0))
            if "axis_angle" in self.layout:
                metrics["residual/rot_deg"] = (
                    correction[:, self.layout["axis_angle"]].norm(dim=-1) * (self.layout.get("rad_per_unit", 1.0) * 180.0 / torch.pi)
                )
            elif "yaw" in self.layout:
                metrics["residual/yaw_deg"] = correction[:, self.layout["yaw"]].abs() * (180.0 / torch.pi)
            elif "rot6d" in self.layout:  # absolute targets: the angle between the base's and the sent rotation
                r6 = self.layout["rot6d"]
                metrics["residual/rot_deg"] = rot6d_angle_deg(base[:, r6], sent[:, r6])
            else:  # a quaternion delta's size: the angle of the correction's rotation part
                q = correction[:, self.layout["quat_xyzw"]]
                metrics["residual/rot_deg"] = 2.0 * torch.atan2(q[:, 0:3].norm(dim=-1), q[:, 3].abs()) * (180.0 / torch.pi)
        return observations, rewards, terminated, truncated, infos

    def _phase_metrics(self, infos, acted_in) -> None:
        """With the env's task phases published (``furniture.track_phases``): this step's correction sizes per phase,
        ``residual/<size>_<phase>`` (``position_mm``, ``rot_deg``, ...), by the phase of the observation the action
        was chosen for -- each env's value in its phase's channel, NaN in the others, so every interval's mean is over
        that phase's steps alone (the logger skips NaN). Then the new observation's phase, kept for the next step: an
        episode-end code (4-6) is followed by the next episode's first observation, Corner."""
        state = infos.get("eval_state") if isinstance(infos, dict) else None
        phase = state.get("phase/id") if isinstance(state, dict) else None
        if phase is None:
            return
        if acted_in is not None:
            metrics = infos.setdefault("metrics_to_log", {})
            for name in [n for n in metrics if n.startswith("residual/") and n.split("/", 1)[1] in RESIDUAL_SIZES]:
                value = metrics[name]
                nan = torch.full_like(value, float("nan"))
                for k, label in enumerate(PHASE_NAMES):
                    metrics[f"{name}_{label}"] = torch.where(acted_in.to(value.device) == k, value, nan)
        now = torch.as_tensor(phase, device=self.phase.device).reshape(-1).round().long()
        self.obs_phase = torch.where(now >= len(PHASE_NAMES), torch.zeros_like(now), now)

    def _check_phase_label(self, infos) -> None:
        """One source of truth for the phase: the label the observation was built with (the tracker's, :meth:`_phase_rows`)
        must be the one the env publishes with it for the learner's routing (``eval_state/phase/id``; an episode's end
        code there is followed by the next episode's first observation, Corner). Refused loudly when they differ."""
        label = getattr(self, "obs_label", None)
        state = infos.get("eval_state") if isinstance(infos, dict) else None
        published = state.get("phase/id") if isinstance(state, dict) else None
        if label is None or published is None or self.steps_per_chunk != 1:
            return
        now = torch.as_tensor(published, device=label.device).reshape(-1).round().long()
        expected = torch.where(now >= len(PHASE_NAMES), torch.zeros_like(now), now)
        # the check stays on the device (torch.equal would synchronize every step); a mismatch is latched and
        # raised, with the offending envs, at the next interval check (_phase_label_check_due)
        mismatch = expected != label
        if self._label_mismatch is None:
            self._label_mismatch = mismatch
            self._label_mismatch_pair = (expected, label, now)
        else:
            self._label_mismatch = self._label_mismatch | mismatch
            self._label_mismatch_pair = (torch.where(mismatch, expected, self._label_mismatch_pair[0]),
                                         torch.where(mismatch, label, self._label_mismatch_pair[1]),
                                         torch.where(mismatch, now, self._label_mismatch_pair[2]))
        self._label_checks += 1
        if self._label_checks % LABEL_CHECK_EVERY == 0:
            self._raise_label_mismatch()

    def _raise_label_mismatch(self) -> None:
        """Read the latched label check (one host synchronization) and fail loudly if any env ever mismatched."""
        if self._label_mismatch is None:
            return
        if bool(self._label_mismatch.any()):
            expected, label, now = self._label_mismatch_pair
            bad = torch.nonzero(self._label_mismatch).reshape(-1)[:8].tolist()
            raise RuntimeError(f"the residual built envs {bad}'s observations with phase labels {label[bad].tolist()} but the env "
                               f"published {now[bad].tolist()} for the learner: the two must be one source "
                               f"(found within the last {LABEL_CHECK_EVERY} steps)")

    def _track_phase(self, infos) -> None:
        """``active_from_phase``: each env's phase at its new observation, from the env's eval_state phase/id."""
        if self.active_from is None:
            return
        state = infos.get("eval_state") if isinstance(infos, dict) else None
        phase = state.get("phase/id") if isinstance(state, dict) else None
        if phase is None:
            raise RuntimeError(
                "residual.active_from_phase reads the env's task phase (infos eval_state phase/id): set "
                "task.cfg.furniture.track_phases: true"
            )
        self.phase = torch.as_tensor(phase, device=self.phase.device).reshape(-1).round().long()

def action_space_of(source: Any, action_dim: int):
    """The action space (``my_project/action_spaces.py``) of a wrapper -- or of anything holding its attributes
    (``pose``, ``scale``, ``abs_clip``, ``phase_tables``, ``keep``, ``observe_base``, ``critic_mode``, ``env_critic``,
    ``steps_per_chunk``)."""
    from my_project.action_spaces import PerDimSpace, PoseSpace

    common = dict(keep=source.keep, observe_base=source.observe_base, critic_mode=source.critic_mode,
                  env_critic=source.env_critic, steps_per_chunk=source.steps_per_chunk)
    if source.pose is not None:
        return PoseSpace(source.pose, action_dim, **common)
    return PerDimSpace(source.scale, clip=source.abs_clip, phase_scales=source.phase_tables is not None, **common)


def _rot6d_to_mat(d6: torch.Tensor) -> torch.Tensor:
    """``(N, 6)`` first two rows -> ``(N, 3, 3)`` by Gram-Schmidt (the action term's own decode)."""
    b1 = torch.nn.functional.normalize(d6[:, 0:3], dim=-1)
    a2 = d6[:, 3:6]
    b2 = torch.nn.functional.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1)
    return torch.stack((b1, b2, torch.cross(b1, b2, dim=-1)), dim=-2)


def rot6d_angle_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Angle (deg) between the rotations two ``(N, 6)`` rot6d vectors decode to."""
    R = _rot6d_to_mat(a).transpose(-1, -2) @ _rot6d_to_mat(b)
    cos = ((R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.arccos(cos))
