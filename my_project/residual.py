"""Residual RL on a frozen diffusion policy: the env runs the base, the agent learns a correction.

:class:`ResidualPolicyWrapper` sits outermost (``register_env_wrapper`` in every script's
``setup``), on top of the world-pose action. Every step:

1. The frozen base (:class:`~my_project.models.diffusion_policy.ChunkedBase`) gives this
   step's base action ``a_base = [x, y, z, yaw]`` from the current observation. Each env keeps
   its own chunk: a new one is predicted when ``action_horizon`` actions are used up or the
   env's episode just began. ``residual.base_action_horizon`` / ``base_inference_steps``
   override the base run's executed horizon and DDIM steps (:func:`override_base`).
2. The agent observes ``[normalize(obs), normalize(a_base)]``: the base's observation and the
   base action, with the **base policy's** min-max normalizer (as robust-rearrangement's
   residual does), its near-constant observation columns widened (:func:`residual_observation`).
   ``residual.normalize_observation: false`` leaves the agent's observation raw;
   ``residual.observe_base_action: false`` leaves the base action out of it. The **critic state**
   is the residual's own, whatever those flags say (:func:`critic_state`): the env's critic state
   or, without one, the normalized base observation, then the base action(s), normalized; the
   wrapper creates a critic group for an env that has none. ``critic_action: full`` reads it.
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
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import torch

__all__ = [
    "load_base_policy", "base_run_dir", "base_source", "override_base", "ResidualPolicyWrapper", "residual_scale",
    "residual_observation", "critic_state", "find_residual", "full_action_transform", "residual_critic_action",
]


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


def full_action_transform(base_action_slice: slice, gain: torch.Tensor, steps_per_chunk: int = 1):
    """``(critic_state, r) -> critic action`` for ``residual.critic_action: full``: the executed pose
    action normalized as the base's, ``normalize(a_base) + gain * r`` with ``gain = scale /
    half_range`` -- exactly ``normalize(a_base + scale * r)``, the base's normalizer being affine.
    ``critic_state`` is the wrapper's critic state (:func:`critic_state`; the learner hands the
    transform the critic's own input), ``normalize(a_base)`` at ``base_action_slice`` -- never the
    agent's observation, so what the agent observes does not matter. Action dims past the pose
    (``hybrid``'s blocks) pass through.

    With ``residual.steps_per_chunk: k`` the slice holds the chunk's ``k`` base actions and ``r`` its
    ``k`` rows, both step-major: the result is the chunk's ``k`` executed actions, ``normalize(a_base_j)
    + gain * r_j`` concatenated (``k x A``, the agent's action width)."""
    k = int(steps_per_chunk)
    width = int(gain.numel()) * k
    gain = gain.repeat(k)  # the same per-dim gain on every row

    def transform(observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        pose = observations[..., base_action_slice] + gain * actions[..., :width]
        return torch.cat((pose, actions[..., width:]), dim=-1)

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
            "with normalize(a_base) in their state",
            flush=True,
        )
        return ee_full_action_transform(wrapper)
    print(
        f"[residual] critics take the executed action: normalize(a_base) + "
        f"{[round(g, 5) for g in wrapper.critic_action_gain.tolist()]} * r"
        + (f", every row of the {wrapper.steps_per_chunk}-step chunk" if wrapper.steps_per_chunk > 1 else ""),
        flush=True,
    )
    return full_action_transform(wrapper.critic_base_slice, wrapper.critic_action_gain, wrapper.steps_per_chunk)


def residual_observation(
    normalizer, obs: torch.Tensor, base_action: torch.Tensor, normalize_obs: bool = True, observe_base_action: bool = True
):
    """The agent's observation from the base's raw observation and its action:
    ``([normalize(obs), normalize(a_base)], normalize(a_base))`` — the second part is what the
    critic state gets appended. One definition for the env wrapper and the offline demo
    conversion (``my_project/residual_demos.py``).

    The observation part reads the base's stats with every column narrower than
    ``MinMaxNormalizer.NARROW_RANGE`` widened (``normalize(..., widen=True)``): a base fit before
    that rule stretched float noise in those columns over [-1, 1], and the agent should see them
    as the constants they are. The base itself still reads its stats as it was trained on them.
    The base action keeps the base's stats exactly (``critic_action_gain`` is built from them).

    ``residual.normalize_observation: false`` (``normalize_obs``) keeps the observation part raw;
    ``residual.observe_base_action: false`` leaves the base action out: the observation is the
    observation part alone and the second element is empty, ``(N, 0)``. ``base_action`` is one
    action per env ``(N, A)`` or, with ``residual.steps_per_chunk: k``, the chunk's ``k`` ``(N, k, A)``:
    each row normalized, laid out step-major (``N, k x A``)."""
    part = normalizer.normalize(obs, "obs", widen=True) if normalize_obs else obs
    if not observe_base_action:
        return part, part[..., :0]
    base_n = normalizer.normalize(base_action, "action").reshape(obs.shape[0], -1)
    return torch.cat((part, base_n), dim=-1), base_n


def critic_state(normalizer, obs: torch.Tensor, base_action: torch.Tensor, env_critic=None) -> torch.Tensor:
    """The residual's critic state before its appendices: the env's own critic state (``env_critic``,
    Forge's ``state_order``) or, for an env without one, the base's observation normalized as the agent's
    default (widened), then the base action (one ``(N, A)`` or the chunk's ``k``, ``(N, k, A)``),
    normalized, step-major. Independent of ``residual.normalize_observation`` / ``observe_base_action``:
    those shape the agent's observation only. One definition for the env wrapper and the demo
    conversion; the EE offset (``accumulate``) and the history go on after it."""
    head = env_critic if env_critic is not None else normalizer.normalize(obs, "obs", widen=True)
    return torch.cat((head, normalizer.normalize(base_action, "action").reshape(obs.shape[0], -1)), dim=-1)


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
        self.normalizer = policy.normalizer
        self.scale = residual_scale(policy, rc.action_scale, rc.action_scale_fraction)
        #: residual.normalize_observation / observe_base_action: what the agent's observation holds
        self.normalize_obs = bool(rc.normalize_observation)
        self.observe_base = bool(rc.observe_base_action)
        #: residual.steps_per_chunk: env steps per agent action (override_base checked the base's horizon)
        self.steps_per_chunk = int(rc.steps_per_chunk)
        if self.steps_per_chunk > 1 and self.extra_dims:
            raise NotImplementedError("residual.steps_per_chunk > 1 with hybrid.enabled is not supported")
        #: where normalize(a_base) sits in the agent's observation (residual_observation puts it
        #: right after the base's own; the history goes on after it; None when not observed), and
        #: the gain from r to the normalized executed action: residual.critic_action: full
        self.base_action_slice = (
            slice(obs_width, obs_width + self.steps_per_chunk * policy.action_dim) if self.observe_base else None
        )
        #: the critic state (:func:`critic_state`): the env's own critic state, or the normalized base
        #: observation, then the chunk's base actions; ``critic_action: full`` reads them there
        self.env_critic = hooks.has_critic()
        critic_head = hooks.critic_width() if self.env_critic else obs_width
        self.critic_base_slice = slice(critic_head, critic_head + self.steps_per_chunk * policy.action_dim)
        self.critic_offset_slice = None
        self.critic_action_gain = self.scale / ((policy.normalizer.action_max - policy.normalizer.action_min) / 2)
        self.base_action = torch.zeros((hooks.num_envs, policy.action_dim), device=hooks.device)
        self.layout = action_layout(cfg.task_name)
        self.obs_width = obs_width
        self.num_envs = hooks.num_envs
        #: sub-steps of the last agent step that belonged to each env's episode (all of them, or
        #: those up to and including the one it ended on): what an episode's length counts
        self.counted = torch.ones((hooks.num_envs,), dtype=torch.long, device=hooks.device)
        base_width = self.steps_per_chunk * policy.action_dim
        extra_width = base_width if self.observe_base else 0
        critic_extra = base_width

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
                start = obs_width + extra_width
                self.offset_slice = slice(start, start + OFFSET_WIDTH)
                self.critic_offset_slice = slice(self.critic_base_slice.stop, self.critic_base_slice.stop + OFFSET_WIDTH)
                extra_width += OFFSET_WIDTH
                critic_extra += OFFSET_WIDTH
            self.width = WIDTH
        if self.ee is not None or self.steps_per_chunk > 1:
            hooks.set_agent_action_width(self.steps_per_chunk * self.width)

        # the agent's observation gains what the flags ask for; the critic state is always the residual's own
        if self.env_critic:
            hooks.widen(extra_width, critic_extra)
        else:
            hooks.create_critic(obs_width + critic_extra)
            hooks.widen(extra_width, 0)
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
            + ("" if self.normalize_obs else "; raw observation")
            + ("" if self.observe_base else "; base action not observed"),
            flush=True,
        )

    def _observations(self, observations):
        obs = observations["policy"]
        fresh = self._hooks.fresh()
        # residual.accumulate: the envs whose base starts a new chunk this step (ChunkedBase.actions replans those)
        new_chunk = None
        if self.ee is not None and self.ee.accumulate:
            new_chunk = fresh.to(torch.bool).reshape(-1) | (self.base.index >= self.base.policy.action_horizon)
        self.base_action = self.base.actions(obs, fresh)
        # the observed base action: this step's, or the chunk's k (at a chunk boundary, the next k)
        observed = self.base_action if self.steps_per_chunk == 1 else self.base.block(self.steps_per_chunk)
        observations["policy"], _ = residual_observation(
            self.normalizer, obs, observed, self.normalize_obs, self.observe_base
        )
        critic = critic_state(self.normalizer, obs, observed, observations.get("critic") if self.env_critic else None)
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
        observations["critic"] = critic
        return observations

    def step(self, action):
        action = action.to(self.scale.device)
        if self.steps_per_chunk > 1:
            return self._step_chunk(action)
        return self._step_once(action)

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
        sums, closing_values = {}, {}
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
            alive &= ~(term_f | trunc_f)
        self.counted = counted
        if isinstance(infos, dict):
            if sums:
                infos["metrics_to_log"] = {
                    name: value / counted.reshape((n,) + (1,) * (value.dim() - 1)).to(value.dtype) for name, value in sums.items()
                }
            if closing_values:
                infos["episode_metrics_to_log"] = closing_values
        return observations, reward_sum, terminated.reshape(term.shape).to(term.dtype), truncated.reshape(trunc.shape).to(trunc.dtype), infos

    def _step_once(self, action):
        """One env step: ``action`` is the agent's per-step action (``width`` plus the hybrid dims)."""
        if self.ee is not None:
            return self._step_ee(action)
        width = self.base_action.shape[-1]
        correction = self.scale * action[:, :width]
        base = self.base_action  # env.step's observation hook replaces it with the next step's
        sent = base + correction
        if self.extra_dims:
            sent = torch.cat((sent, action[:, width:]), dim=-1)
        observations, rewards, terminated, truncated, infos = self.env.step(sent)
        if isinstance(infos, dict):
            metrics = infos.setdefault("metrics_to_log", {})
            metrics["residual/position_mm"] = correction[:, self.layout["pos"]].norm(dim=-1) * 1000.0
            if "yaw" in self.layout:
                metrics["residual/yaw_deg"] = correction[:, self.layout["yaw"]].abs() * (180.0 / torch.pi)
            elif "rot6d" in self.layout:  # absolute targets: the angle between the base's and the sent rotation
                r6 = self.layout["rot6d"]
                metrics["residual/rot_deg"] = rot6d_angle_deg(base[:, r6], sent[:, r6])
            else:  # a quaternion delta's size: the angle of the correction's rotation part
                q = correction[:, self.layout["quat_xyzw"]]
                metrics["residual/rot_deg"] = 2.0 * torch.atan2(q[:, 0:3].norm(dim=-1), q[:, 3].abs()) * (180.0 / torch.pi)
        return observations, rewards, terminated, truncated, infos

    def _step_ee(self, action):
        base = self.base_action  # env.step's observation hook replaces it (and R_ee, the offset)
        sent, self.offset_pos, self.offset_rot = self.ee.step(base, self.R_ee, self.offset_pos, self.offset_rot, action[:, :6])
        observations, rewards, terminated, truncated, infos = self.env.step(sent)
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

    ee, normalizer, base_slice, offset_slice = wrapper.ee, wrapper.normalizer, wrapper.critic_base_slice, wrapper.critic_offset_slice
    obs_width = wrapper.obs_width
    if wrapper.env_critic:
        raise NotImplementedError(
            "residual.critic_action: full with action_frame: ee reads the EE pose from the critic state's normalized "
            "base observation, which an env with its own critic state does not put there"
        )

    def transform(critic: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        base = normalizer.denormalize(critic[..., base_slice], "action")
        R_ee = ee_rotation(normalizer.denormalize(critic[..., :obs_width], "obs"))
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
