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

``residual.action_frame: multiply`` (``my_project/residual_multiply.py``): the agent's ``r`` multiplies the base's target
taken relative to the EE, per pose dim, and the agent observes that relative target and the observation relative to the
EE (each routed phase standardized over its own demo frames, ``residual.phase_scales``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import gymnasium as gym
import torch

__all__ = [
    "load_base_policy", "base_run_dir", "base_source", "override_base", "ResidualPolicyWrapper", "residual_scale",
    "residual_observation", "critic_state", "find_residual", "full_action_transform", "target_residual", "residual_scale_and_clip",
    "residual_critic_action",
    "SideInputTransform", "MultiplySideInput", "SIDE_INPUT",
]

#: ``residual.critic_state: observation``: the per-transition side input (the learner's ``side_inputs``) the
#: critics' executed action is rebuilt from -- this agent step's base action(s), normalized, step-major; the
#: env publishes it and its ``next_`` value in ``infos`` every agent step
SIDE_INPUT = "residual_base"

#: one_leg's task phases in phase/id order (furniture_lab assembly/phases.py); ids from len(PHASE_NAMES) on are an
#: episode's end codes
PHASE_NAMES = ("corner", "grasp", "insert", "screw")
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
    ``action_frame: absolute`` with ``executed_clip``, the executed action's per-dim bounds ``center +- executed_clip x
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
    if rc.action_frame == "absolute" and getattr(rc, "executed_clip", None) is not None:
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


def target_residual(rc: Any, scale: torch.Tensor, normalizer, gripper: int):
    """``residual.action_frame: target``'s map (:class:`~my_project.residual_target.TargetResidual`) as the config
    gives it: ``scale`` the absolute frame's per-dim scale (:func:`residual_scale`; the gripper's unless
    ``ee_grip_scale`` sets it, the position's with ``target_scale_from_base``), ``normalizer`` the base's (``executed_clip``'s bounds), ``gripper``
    the gripper's action dim."""
    from my_project.residual_target import TargetResidual

    pos = scale[0:3].clone() if rc.target_scale_from_base else rc.ee_pos_scale
    if getattr(rc, "target_pos_scale", None) is not None:
        pos = torch.tensor([float(v) for v in rc.target_pos_scale], dtype=torch.float32, device=scale.device)
    rot = rc.ee_rot_scale_deg
    if getattr(rc, "target_rot_scale_deg", None) is not None:
        rot = torch.tensor([float(v) for v in rc.target_rot_scale_deg], dtype=torch.float32, device=scale.device)
    clip = None
    if rc.executed_clip is not None:
        center = (normalizer.action_max + normalizer.action_min) / 2
        half = (normalizer.action_max - normalizer.action_min) / 2 * float(rc.executed_clip)
        clip = (center - half, center + half)
    grip = float(rc.ee_grip_scale) if getattr(rc, "ee_grip_scale", None) is not None else float(scale[gripper])
    return TargetResidual(pos, rot, grip, axes=str(rc.target_axes),
                          gripper=bool(rc.residual_gripper), rotation=str(rc.target_rotation), clip=clip)


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
    ``residual.critic_state: observation``: the critics observe the observation and nothing else (Policy Decorator's
    critic), and the executed action is rebuilt from the raw base action(s) the env publishes as the side input
    :data:`SIDE_INPUT` (``side_inputs``, stored per transition, never a critic input). ``absolute``: ``a_base + scale *
    r`` per row, the agent's width; ``side_scale`` (``residual.phase_scales``): each transition's own scale, the side
    input's second ``base_width`` columns. ``target``: the chunk's executed targets ``TargetResidual.step(a_base_j,
    r_j)``, ``k x 10`` wide (``critic_action_width``)."""

    def __init__(self, steps_per_chunk: int, base_width: int, scale=None, target=None, ee_rel=None,
                 accumulate_reach: Optional[int] = None, side_scale: bool = False) -> None:
        self.side_scale = bool(side_scale)
        if self.side_scale and (target is not None or ee_rel is not None or scale is not None or int(steps_per_chunk) != 1):
            raise ValueError("side_scale is the absolute frame's per-phase scale: no target, ee_rel or fixed scale, one step")
        self.k = int(steps_per_chunk)
        self.base_width = int(base_width)
        self.scale = None if scale is None else scale.repeat(self.k)
        self.target = target
        #: ``residual.ee_relative_actions``: the executed targets in the EE-relative command form, against the raw EE
        #: pose the side input carries after the base actions (``k x 7``, the agent's own width)
        self.ee_rel = ee_rel
        if ee_rel is not None and target is None:
            raise ValueError("ee_relative_actions rebuilds the target-frame residual's executed targets: it needs target")
        #: residual.target_accumulate: the side input carries the offset before this step (its observation, 6) after the
        #: base action, over this many steps of reach
        self.accumulate_reach = accumulate_reach
        if accumulate_reach is not None and (target is None or ee_rel is not None or self.k != 1):
            raise ValueError("the accumulating target residual's critic action needs target, no ee_relative_actions, one step")
        self.side_inputs = {SIDE_INPUT: self.k * self.base_width + (9 if ee_rel is not None else 0) + (6 if accumulate_reach else 0)
                            + (self.base_width if self.side_scale else 0)}
        if ee_rel is not None:
            self.critic_action_width = self.k * 7
        elif target is not None:
            self.critic_action_width = self.k * self.base_width

    def __call__(self, critic_inputs: torch.Tensor, actions: torch.Tensor, side) -> torch.Tensor:
        base = side[SIDE_INPUT]
        if self.side_scale:
            width = self.base_width
            return torch.cat((base[..., :width] + base[..., width : 2 * width] * actions[..., :width], actions[..., width:]), dim=-1)
        if self.target is None:
            width = self.k * self.base_width
            return torch.cat((base + self.scale * actions[..., :width], actions[..., width:]), dim=-1)
        n = actions.shape[0]
        width = self.k * self.base_width
        chunk = base[..., :width].reshape(n, self.k, self.base_width)
        if self.accumulate_reach is not None:
            off_pos, off_rot = self.target.from_offset_obs(base[..., width:width + 6], self.accumulate_reach)
            sent, _, _ = self.target.accumulate(chunk[:, 0], actions, off_pos, off_rot)
            return sent.reshape(n, -1)
        sent = self.target.step(chunk, actions.reshape(n, self.k, -1))
        if self.ee_rel is not None:
            return self.ee_rel(sent, base[..., width:])
        return sent.reshape(n, -1)


class MultiplySideInput:
    """``(critic_inputs, r, side) -> critic action`` for ``residual.action_frame: multiply``
    (:meth:`~my_project.residual_multiply.MultiplyResidual.critic_action`): the resulting target relative to the EE and
    the gripper (7), rebuilt from the side input :data:`SIDE_INPUT`."""

    def __init__(self, multiply) -> None:
        from my_project.residual_multiply import CRITIC_ACTION_WIDTH

        self.multiply = multiply
        self.side_inputs = {SIDE_INPUT: multiply.side_width}
        self.critic_action_width = CRITIC_ACTION_WIDTH

    def __call__(self, critic_inputs: torch.Tensor, actions: torch.Tensor, side) -> torch.Tensor:
        return self.multiply.critic_action(side[SIDE_INPUT], actions)


def target_full_transform(base_slice: slice, steps_per_chunk: int, base_width: int, target):
    """``(critic_state, r) -> critic action`` for ``critic_action: full`` with ``action_frame: target`` and the
    residual's critic state: :class:`SideInputTransform`'s target map, the base action(s) read at
    ``base_slice`` of the critic state."""
    inner = SideInputTransform(steps_per_chunk, base_width, target=target)

    def transform(critic: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return inner(critic, actions, {SIDE_INPUT: critic[..., base_slice]})

    transform.critic_action_width = inner.critic_action_width
    return transform


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
    ``residual.enabled`` and ``residual.critic_action: full``, the SAC critics take the executed
    action (:func:`full_action_transform`); otherwise None, and they take ``r``."""
    rc = getattr(cfg, "residual", None)
    if rc is None or not rc.enabled or rc.critic_action != "full":
        return None
    wrapper = find_residual(env)
    critic_mode, target = getattr(wrapper, "critic_mode", "residual"), getattr(wrapper, "target", None)
    if critic_mode == "observation" or target is not None:
        k, base_width = wrapper.steps_per_chunk, wrapper.base_action.shape[-1]
    if critic_mode == "observation":
        if getattr(wrapper, "multiply", None) is not None:
            print("[residual] action_frame: multiply: the critics observe the observation part and take the resulting target "
                  "relative to the EE and the gripper, rebuilt from the side input", flush=True)
            return MultiplySideInput(wrapper.multiply)
        ee_rel = getattr(wrapper, "ee_rel", None)
        print(
            "[residual] critics observe the observation alone (residual.critic_state: observation) and take "
            + ("the executed targets of the target-frame residual" if target is not None else "a_base + scale * r")
            + (f" in the EE-relative command form ({k} x 7)" if ee_rel is not None else "")
            + f", the base action(s) a side input ({k} x {base_width}{' + the EE pose' if ee_rel is not None else ''})",
            flush=True,
        )
        if getattr(wrapper, "phase_tables", None) is not None:
            print("[residual] residual.phase_scales: each transition's scale from r to the executed action is its phase's, "
                  "carried in the side input after the base action", flush=True)
            return SideInputTransform(k, base_width, side_scale=True)
        return SideInputTransform(
            k, base_width,
            scale=None if target is not None else wrapper.scale, target=target,
            ee_rel=getattr(wrapper, "ee_rel", None),
            accumulate_reach=wrapper.t_reach if getattr(wrapper, "t_acc", False) else None,
        )
    if target is not None:
        print("[residual] critics take the executed targets of the target-frame residual, a_base in their state", flush=True)
        return target_full_transform(wrapper.critic_base_slice, k, base_width, target)
    if wrapper.ee is not None:
        if wrapper.steps_per_chunk > 1:
            raise NotImplementedError(
                f"residual.critic_action: full with action_frame: ee and residual.steps_per_chunk {wrapper.steps_per_chunk}: "
                "the EE frame the correction is expressed in turns between the chunk's sub-steps, so the offsets "
                "executed after the first sub-step are not determined by the chunk's observation"
            )
        print(
            f"[residual] critics take the executed correction: the EE-frame offset after the step "
            f"({rc.ee_pos_scale} m, {rc.ee_rot_scale_deg} deg per unit{', accumulated over the chunk' if rc.accumulate else ''}), "
            "with a_base in their state",
            flush=True,
        )
        return ee_full_action_transform(wrapper)
    print(
        f"[residual] critics take the executed action: a_base + "
        f"{[round(g, 5) for g in wrapper.scale.tolist()]} * r"
        + (f", every row of the {wrapper.steps_per_chunk}-step chunk" if wrapper.steps_per_chunk > 1 else ""),
        flush=True,
    )
    return full_action_transform(wrapper.critic_base_slice, wrapper.scale, wrapper.steps_per_chunk)


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
        self.critic_offset_slice = None
        self.base_action = torch.zeros((hooks.num_envs, policy.action_dim), device=hooks.device)
        self.layout = action_layout(cfg.task_name)
        self.obs_width = obs_width
        self.num_envs = hooks.num_envs
        #: sub-steps of the last agent step that belonged to each env's episode (all of them, or
        #: those up to and including the one it ended on): what an episode's length counts
        self.counted = torch.ones((hooks.num_envs,), dtype=torch.long, device=hooks.device)
        base_width = self.steps_per_chunk * policy.action_dim
        #: residual.ee_relative_actions: the observed base action and the critics' action in Policy Decorator's
        #: EE-relative command form (my_project/residual_target.py EERelative), 7 per step
        self.ee_rel = None
        if rc.ee_relative_actions:
            from my_project.residual_target import EERelative

            self.ee_rel = EERelative(self.layout["gripper"])
        #: residual.action_frame: multiply -- r multiplies the base's motion (my_project/residual_multiply.py); per
        #: std_label_names label whether it has a residual (route_labels), and the chain's state per env
        self.multiply = None
        if rc.action_frame == "multiply":
            from my_project.residual_multiply import multiply_from

            self.multiply = multiply_from(rc, self.layout["gripper"])
            routes = list(section.route_labels)
            self.m_routed = torch.tensor([name in routes for name in self.phase_names], dtype=torch.bool, device=hooks.device)
            n = hooks.num_envs
            self.m_env_routed = torch.zeros((n,), dtype=torch.bool, device=hooks.device)
            self.m_increment = self.multiply.reference == "increment"
            #: increment: the current target (the last sent one) and the base's last target, this step's planned step and
            #: the pose it is chained onto
            self.m_prev_sent = torch.zeros((n, policy.action_dim), device=hooks.device)
            self.m_prev_base = torch.zeros((n, policy.action_dim), device=hooks.device)
            self.m_d = torch.zeros((n, 6), device=hooks.device)
            self.m_ref = torch.zeros((n, 9), device=hooks.device)
        observed_width = self.steps_per_chunk * 7 if self.ee_rel is not None else base_width
        if self.multiply is not None:
            from my_project.residual_multiply import CURRENT_WIDTH, OBSERVED_WIDTH

            # the base's motion, then (increment) the current target relative to the EE
            observed_width = OBSERVED_WIDTH + (CURRENT_WIDTH if self.m_increment else 0)
        extra_width = observed_width if self.observe_base else 0
        critic_extra = base_width

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
        #: the normalized base action(s) the agent's last observation was made with (the side input), then with
        #: ee_relative_actions the raw EE pose (9) the chunk's targets are taken against
        self.side = torch.zeros((hooks.num_envs, base_width + (9 if self.ee_rel is not None else 0)), device=hooks.device)
        if self.multiply is not None:
            self.side = torch.zeros((hooks.num_envs, self.multiply.side_width), device=hooks.device)
        # residual.action_frame: target -- Policy Decorator's relative pose on the base's target (my_project/residual_target.py)
        self.target = None
        if rc.action_frame == "target":
            from my_project.residual_target import TargetResidual

            self.target = target_residual(rc, self.scale, policy.normalizer, self.layout["gripper"])
        #: residual.target_accumulate: each env's offset on the base's targets (m, rotation), reset when the base replans;
        #: it is observed (after the base action) by the agent and the critics, and the critics' side input carries it
        self.t_acc = bool(getattr(rc, "target_accumulate", False)) and self.target is not None
        self.t_reach = int(policy.action_horizon)
        if self.t_acc:
            self.t_off_pos, self.t_off_rot = self.target.identity(hooks.num_envs, hooks.device)
            extra_width += 6
            self.side = torch.zeros((hooks.num_envs, base_width + 6), device=hooks.device)
        # residual.action_frame: ee -- the agent's 6-d EE-frame correction (my_project/residual_ee.py)
        self.ee = None
        self.offset_slice = None
        #: the per-step residual width A: the agent's action is (steps_per_chunk x A)
        self.width = policy.action_dim
        if rc.action_frame == "ee":
            from my_project.residual_ee import OFFSET_WIDTH, WIDTH, EEResidual

            self.ee = EEResidual(rc.ee_pos_scale, rc.ee_rot_scale_deg, rc.accumulate, policy.action_horizon)
            n = hooks.num_envs
            self.R_ee = torch.eye(3, device=hooks.device).expand(n, 3, 3).clone()
            self.offset_pos, self.offset_rot = self.ee.identity(n, hooks.device)
            if rc.accumulate:  # the offset follows the base action in the observation and the critic state
                start = kept_width + extra_width
                self.offset_slice = slice(start, start + OFFSET_WIDTH)
                self.critic_offset_slice = slice(self.critic_base_slice.stop, self.critic_base_slice.stop + OFFSET_WIDTH)
                extra_width += OFFSET_WIDTH
                critic_extra += OFFSET_WIDTH
            self.width = WIDTH
        if self.target is not None:
            self.width = self.target.width
        if self.multiply is not None:
            from my_project.residual_multiply import WIDTH as MULTIPLY_WIDTH  # noqa: N811

            self.width = MULTIPLY_WIDTH
        if self.ee is not None or self.target is not None or self.multiply is not None or self.steps_per_chunk > 1:
            hooks.set_agent_action_width(self.steps_per_chunk * self.width)

        # the agent's observation gains what the flags ask for; the critic state is the residual's own, or none
        if self.critic_mode == "observation":
            if extra_width or dropped:
                # the critics keep the observation without the agent's additions (Policy Decorator's critic input),
                # plus the accumulated offset (state) with target_accumulate
                hooks.create_critic(kept_width + (6 if self.t_acc else 0))
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
        print(
            f"[residual] base {source}/{rc.base_checkpoint} (step {getattr(policy, 'checkpoint_step', -1)}, "
            f"best success rate {getattr(policy, 'checkpoint_success_rate', float('nan')):.3f}; executes "
            f"{policy.action_horizon} of {policy.horizon}, {policy.cfg.inference_steps} DDIM steps); "
            f"scale {self.scale.tolist()}"
            + (f"; {self.steps_per_chunk} env steps per agent action ({self.steps_per_chunk} x {self.width})"
               if self.steps_per_chunk > 1 else "")
            + "; raw inputs (the learner's state_preprocessor normalizes them)"
            + ("" if self.observe_base else "; base action not observed")
            + (f"; observation columns {list(rc.observation_drop)} dropped ({kept_width} of {obs_width} kept)" if dropped else "")
            + ("; gripper left to the base" if self.target is not None and not rc.residual_gripper else "")
            + (f"; target-frame residual ({rc.target_axes} axes, {rc.target_rotation}), "
               f"{[round(v, 5) for v in self.target.pos_scale.tolist()] if isinstance(self.target.pos_scale, torch.Tensor) else self.target.pos_scale} m / "
               f"{[round(v, 3) for v in torch.rad2deg(self.target.rot_scale).tolist()] if isinstance(self.target.rot_scale, torch.Tensor) else rc.ee_rot_scale_deg} deg / {self.target.grip_scale:.5f} gripper per unit"
               + (f", accumulated into an offset that resets when the base replans (every {self.t_reach} steps)" if self.t_acc else "")
               + (f", sent position and gripper clamped to the base's normalized +-{rc.executed_clip}" if rc.executed_clip is not None else "")
               if self.target is not None else "")
            + ("; critics observe the observation alone" if self.critic_mode == "observation" else "")
            + ("; base action and critic action EE-relative (m, rad)" if self.ee_rel is not None else "")
            + (f"; multiply residual: the base's {'planned step (robot-base frame, chained on the current target, the EE at a chunk start)' if self.m_increment else 'EE-relative pose target'} times m in [{rc.multiply_min}, {rc.multiply_max}] per dim "
               f"(m = 1 at r = {self.multiply.neutral:.4f}), the gripper the base's, the observation EE-relative, the critics' action "
               f"the resulting target relative to the EE, residuals in {[n for n, on in zip(self.phase_names, self.m_routed.tolist()) if on]}"
               if self.multiply is not None else ""),
            flush=True,
        )

    def _phase_rows(self, n: int) -> None:
        """The envs' current phases' rows (residual.phase_scales' scale, the multiply residual's routed flag), from the
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
        if self.multiply is not None:
            self.m_env_routed = self.m_routed[phase]

    def _observations(self, observations):
        obs = observations["policy"]
        if self.phase_tables is not None or self.multiply is not None:
            self._phase_rows(obs.shape[0])
        fresh = self._hooks.fresh()
        # residual.accumulate: the envs whose base starts a new chunk this step (ChunkedBase.actions replans those)
        new_chunk = None
        if (self.ee is not None and self.ee.accumulate) or self.t_acc or (self.multiply is not None and self.m_increment):
            new_chunk = fresh.to(torch.bool).reshape(-1) | (self.base.index >= self.base.policy.action_horizon)
        self.base_action = self.base.actions(obs, fresh)
        if self.multiply is not None:
            return self._multiply_observations(observations, obs, new_chunk, fresh)
        # the observed base action: this step's, or the chunk's k (at a chunk boundary, the next k)
        observed = self.base_action if self.steps_per_chunk == 1 else self.base.block(self.steps_per_chunk)
        if self.ee_rel is not None:
            part = obs if self.keep is None else obs[..., self.keep]
            block = observed.reshape(obs.shape[0], self.steps_per_chunk, -1)
            observations["policy"] = torch.cat((part, self.ee_rel(block, obs[:, 0:9])), dim=-1) if self.observe_base else part
            if self.critic_mode == "observation" and self.observe_base:
                observations["critic"] = part
        else:
            observations["policy"], _ = residual_observation(obs, observed, self.observe_base, self.keep)
            if self.critic_mode == "observation" and (self.observe_base or self.keep is not None):
                observations["critic"] = obs if self.keep is None else obs[..., self.keep]
        self.side = observed.reshape(obs.shape[0], -1)
        if self.phase_tables is not None:
            # the critics' scale from r to the executed action: this transition's phase's (residual_critic_action)
            self.side = torch.cat((self.side, self.phase_scale), dim=-1)
        if self.ee_rel is not None:
            self.side = torch.cat((self.side, obs[:, 0:9]), dim=-1)
        if self.t_acc:
            # target_accumulate: the offset goes back to zero when the base replans; it is state, so the agent, the
            # critics and the side input all carry it
            zero_p, eye = self.target.identity(1, obs.device)
            self.t_off_pos = torch.where(new_chunk[:, None], zero_p, self.t_off_pos)
            self.t_off_rot = torch.where(new_chunk[:, None, None], eye, self.t_off_rot)
            off = self.target.offset_obs(self.t_off_pos, self.t_off_rot, self.t_reach)
            observations["policy"] = torch.cat((observations["policy"], off), dim=-1)
            part = obs if self.keep is None else obs[..., self.keep]
            observations["critic"] = torch.cat((part, off), dim=-1)
            self.side = torch.cat((self.side, off), dim=-1)
        critic = (
            None if self.critic_mode == "observation"
            else critic_state(obs, observed, observations.get("critic") if self.env_critic else None, self.keep)
        )
        if self.ee is not None:
            from my_project.residual_ee import ee_rotation

            self.R_ee = ee_rotation(obs)
            if self.ee.accumulate:  # the offset goes back to zero when the base re-plans
                zero_o, eye = self.ee.identity(1, obs.device)
                self.offset_pos = torch.where(new_chunk[:, None], zero_o, self.offset_pos)
                self.offset_rot = torch.where(new_chunk[:, None, None], eye, self.offset_rot)
                offset = self.ee.offset_obs(self.R_ee, self.offset_pos, self.offset_rot)
                observations["policy"] = torch.cat((observations["policy"], offset), dim=-1)
                critic = torch.cat((critic, offset), dim=-1)
        if critic is not None:
            observations["critic"] = critic
        return observations

    def _multiply_observations(self, observations, obs, new_chunk=None, fresh=None):
        """action_frame: multiply: one step's inputs (:meth:`MultiplyResidual.step_inputs`, the one definition the demo
        loader uses too): the agent observes the EE-relative observation part, the base's motion and (increment) the
        current target relative to the EE; the critics the observation part; the side input rebuilds the resulting
        target. increment: a new episode's current target is the EE (it has sent nothing yet), and every chunk starts
        from the EE (the base just replanned from the real state)."""
        if self.m_increment and fresh is not None:
            new = fresh.to(torch.bool).reshape(-1, 1)
            at_ee = torch.cat((obs[:, 0:9], self.base_action[:, 9:]), dim=-1)
            self.m_prev_sent = torch.where(new, at_ee, self.m_prev_sent)
            self.m_prev_base = torch.where(new, self.base_action, self.m_prev_base)
        inputs = self.multiply.step_inputs(obs, self.base_action, self.keep, self.m_env_routed, prev_sent=self.m_prev_sent,
                                           prev_base=self.m_prev_base, chunk_start=new_chunk)
        if not self.observe_base:
            raise NotImplementedError("action_frame: multiply observes the base's motion: set residual.observe_base_action: true")
        observations["policy"], observations["critic"] = inputs["policy"], inputs["critic"]
        self.side, self.m_d, self.m_ref = inputs["side"], inputs["d"], inputs["ref"]
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
        if self.ee is not None:
            return self._step_ee(action)
        if self.target is not None:
            return self._step_target(action)
        if self.multiply is not None:
            return self._step_multiply(action)
        width = self.base_action.shape[-1]
        # residual.phase_scales: each env's phase's scale (the observation's, set by the observation hook)
        correction = (self.phase_scale if self.phase_tables is not None else self.scale) * action[:, :width]
        base = self.base_action  # env.step's observation hook replaces it with the next step's
        sent = base + correction
        if self.abs_clip is not None:
            # residual.executed_clip: the executed action within center +- executed_clip x the scale's half range
            # (ResFiT's clamp of the combined action to its widened range); the critics take the unclamped one
            sent = torch.maximum(torch.minimum(sent, self.abs_clip[1]), self.abs_clip[0])
        if self.extra_dims:
            sent = torch.cat((sent, action[:, width:]), dim=-1)
        observations, rewards, terminated, truncated, infos = self.env.step(sent)
        self._track_phase(infos)
        if isinstance(infos, dict):
            metrics = infos.setdefault("metrics_to_log", {})
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

    def _step_target(self, action):
        base = self.base_action  # env.step's observation hook replaces it with the next step's
        if self.t_acc:
            sent, self.t_off_pos, self.t_off_rot = self.target.accumulate(base, action[:, : self.width], self.t_off_pos, self.t_off_rot)
        else:
            sent = self.target.step(base, action[:, : self.width])
        observations, rewards, terminated, truncated, infos = self.env.step(sent)
        self._track_phase(infos)
        if isinstance(infos, dict):
            metrics = infos.setdefault("metrics_to_log", {})
            metrics["residual/position_mm"] = (sent[:, 0:3] - base[:, 0:3]).norm(dim=-1) * 1000.0
            metrics["residual/rot_deg"] = rot6d_angle_deg(base[:, 3:9], sent[:, 3:9])
        return observations, rewards, terminated, truncated, infos

    def _step_multiply(self, action):
        # env.step's observation hook replaces the base action, the EE pose and the routed rows with the next step's
        base, routed = self.base_action, self.m_env_routed
        r = action[:, : self.width]
        sent = self.multiply.sent(base, self.m_d, self.m_ref, r, routed)
        if self.m_increment:  # the chain's state for the next step's observation (built inside env.step)
            self.m_prev_sent, self.m_prev_base = sent, base
        observations, rewards, terminated, truncated, infos = self.env.step(sent)
        if isinstance(infos, dict):
            metrics = infos.setdefault("metrics_to_log", {})
            metrics["residual/position_mm"] = (sent[:, 0:3] - base[:, 0:3]).norm(dim=-1) * 1000.0
            metrics["residual/rot_deg"] = rot6d_angle_deg(base[:, 3:9], sent[:, 3:9])
            # the mean multiplier over the pose dims (1 where the phase has no residual)
            m = torch.where(routed.reshape(-1, 1), self.multiply.multiplier(r), torch.ones_like(r))
            metrics["residual/multiplier"] = m.mean(dim=-1)
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
        if not torch.equal(expected, label):
            bad = torch.nonzero(expected != label).reshape(-1)[:8].tolist()
            raise RuntimeError(f"the residual built envs {bad}'s observations with phase labels {label[bad].tolist()} but the env "
                               f"published {now[bad].tolist()} for the learner: the two must be one source")

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

    def _step_ee(self, action):
        base = self.base_action  # env.step's observation hook replaces it (and R_ee, the offset)
        sent, self.offset_pos, self.offset_rot = self.ee.step(base, self.R_ee, self.offset_pos, self.offset_rot, action[:, :6])
        observations, rewards, terminated, truncated, infos = self.env.step(sent)
        self._track_phase(infos)
        if isinstance(infos, dict):
            metrics = infos.setdefault("metrics_to_log", {})
            metrics["residual/position_mm"] = (sent[:, 0:3] - base[:, 0:3]).norm(dim=-1) * 1000.0
            metrics["residual/rot_deg"] = rot6d_angle_deg(base[:, 3:9], sent[:, 3:9])
        return observations, rewards, terminated, truncated, infos


def ee_full_action_transform(wrapper: "ResidualPolicyWrapper"):
    """``(observations, r) -> critic action`` for ``residual.critic_action: full`` with ``action_frame: ee``: the
    executed correction, the offset after the step (:meth:`EEResidual.offset_obs`, 6 like ``r``), rebuilt from
    the critic state -- the EE orientation (its normalized base observation) and the offset before the step
    (``accumulate``) -- through the env's own map (:meth:`EEResidual.step`). With ``normalize(a_base)`` in the
    critic state it determines the executed target."""
    from my_project.residual_ee import ee_rotation

    ee, base_slice, offset_slice = wrapper.ee, wrapper.critic_base_slice, wrapper.critic_offset_slice
    obs_width = wrapper.obs_width
    if wrapper.env_critic:
        raise NotImplementedError(
            "residual.critic_action: full with action_frame: ee reads the EE pose from the critic state's normalized "
            "base observation, which an env with its own critic state does not put there"
        )

    def transform(critic: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        base = critic[..., base_slice]
        R_ee = ee_rotation(critic[..., :obs_width])
        if offset_slice is not None:
            o, O = ee.from_offset_obs(R_ee, critic[..., offset_slice])
        else:
            o, O = ee.identity(critic.shape[0], critic.device)
        _, o, O = ee.step(base, R_ee, o, O, actions[..., :6])
        return ee.offset_obs(R_ee, o, O)

    return transform


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
