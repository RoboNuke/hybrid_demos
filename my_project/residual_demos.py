"""Convert recorded demos into SAC demo transitions in a residual agent's own format, offline.

    python scripts/make_demo_transitions.py --config configs/experiments/resip_replica/residual_resip_replica.yaml

No env is built and nothing is simulated: every quantity comes from the recorded frames
(``my_project/demos/record.py``), so the observations are exactly the ones the base policy was
trained on. For each demo of ``T`` frames, transitions ``t -> t+1`` for ``t < T - 1`` (the last
frame's next state was never recorded):

* **observations** / **next_observations**: what the agent would have seen —
  :func:`~my_project.residual.residual_observation` of the recorded policy observation
  (``observation.state``) and the base action, then the :class:`~my_project.wrappers.HistoryBuffer`
  when ``history.enabled``. The base action comes from the frozen base itself
  (:class:`~my_project.models.diffusion_policy.ChunkedBase`, its own chunk clock from the
  demo's first frame), run on the recorded observations: every demo in one batch.
* **states** / **next_states**: the recorded critic state (``observation.critic``) with the
  normalized base action (and the history) appended, as the env wrappers append them.
* **actions**: the residual that reproduces the demo, ``(a_demo - a_base) / scale``, exactly:
  never clipped (a clipped action did not produce the stored next state). Where it lies beyond
  the squashed policy's [-1, 1] in any dim, ``residual.demo_actions_beyond_scale`` stores the
  transition as it is (the critic takes the action as an input) or drops it; ``meta`` reports
  how often, per dim.
* **rewards**: Forge's peg-insert reward of step ``t``, recomputed from the recorded true state
  after it (frame ``t + 1``) by :func:`forge_rewards`, which calls Factory's own reward and
  success code.
* **terminated** / **truncated**: false (a demo ends on success, inside the episode limit, which
  is checked).

``--check_in_sim N`` replays the first ``N`` demos of the session through the experiment's
training env (the demo's own actions) and compares the env's per-step reward, term by term,
with the offline one: the reconstruction's test against the real thing. A restored start
replays exactly only when the reset before it recreates the recorded one (``restore`` cannot
reset PhysX's internal state), i.e. under the session's seed, with plain resets, in recording
order, and with the RNG drawn exactly as the recorder drew it; the check builds its env first
for that, and skips any demo whose peg leaves the recording (the residual's base draws from
the same RNG, so in practice the first demo is the exact one). The one approximated term is
Forge's asset action penalty (scale 0.001), which uses the fingertip at the last physics
substep, here the next frame's. Measured on the search demos: 1.4e-5 at most per step.

**FurnitureBench demos** (``Isaac-Furniture-*-FB-v0``, imported by
``scripts/import_furniture_demos.py``) go through the same conversion with the family's
columns: the recorded next state is in the row (``observation.next_state``), so every one of
the ``T`` actions makes a transition; the reward is the recorded one (1 at the assembly step)
times the env's ``step_dt``, because Isaac Lab's reward manager scales every term by it (so it
is what the env pays online, 0.1 at 10 Hz); the critic state is the policy observation (the env has no
critic group); the last transition is ``terminated`` when the demo succeeded (the env ends an
episode on assembly). On the absolute interface (``*-FBAbs-*``) the columns are the
importer's ``observation.state_abs`` / ``action.abs`` / ``observation.next_state_abs``. Imported demos carry no recording session, so there is nothing to compare
the experiment's task config with, and ``--check_in_sim`` is Forge-only.

``meta["fingerprint"]`` records what the transitions depend on (the residual, the history,
the action interface, the task config); :func:`check_demo_transitions` refuses a file whose
fingerprint differs from the run's, at env build (``my_project/wrappers.py``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

#: task.cfg leaves that may differ between the recording and the experiment: they change no
#: observation or reward of a recorded step (the episode length only has to fit the demo,
#: which is checked). The three spaces are derived: the env recomputes them from obs_order /
#: state_order (which are compared) and the project wrappers widen them, so a dumped config
#: holds whatever the wrappers of that script made of them
TASK_DIFF_ALLOWED = (
    "episode_length_s",
    "seed",
    "sim.render_interval",
    "viewer.",
    "scene.num_envs",
    "observation_space",
    "state_space",
    "action_space",
    # PhysX GPU buffer capacities: how much memory is reserved, not how the sim behaves (an
    # overflow is a PhysX error, never a silent change)
    "sim.physx.gpu_",
)

#: per-episode randomization ranges a recording may hold narrower than the experiment (every
#: value no larger in size): each recorded episode is then one the experiment could also have
#: drawn, so its transitions are valid experiment data. The nut demos are recorded with the
#: bolt estimate exact and the nut centred in the gripper; the experiment randomizes both.
TASK_NARROWER_ALLOWED = (
    "obs_rand.fixed_asset_pos",
    "task.held_asset_pos_noise",
)

#: recorded columns the conversion reads, per env family, with their widths
FURNITURE_COLUMNS = {
    "observation.state": None,  # width from the base policy
    "action": 8,
    "observation.next_state": None,
    "reward": 1,
}
#: the absolute rot6d interface (``*-FBAbs-*``): observation 61 (70 for the ResiP task: + the obstacle pose), action 10
FURNITURE_ABS_COLUMNS = {
    "observation.state_abs": None,
    "action.abs": 10,
    "observation.next_state_abs": None,
    "reward": 1,
}
#: column -> the frame key the conversion uses
FRAME_KEYS = {
    "action": "action",
    "observation.state_abs": "state",
    "action.abs": "action",
    "observation.next_state_abs": "next_state",
}
COLUMNS = {
    "observation.state": None,  # the policy observation; width from the base policy
    "observation.critic": None,  # the critic state; width from the env cfg's state_order
    "action": 4,
    "action.applied": 7,
    "observation.wrench": 6,
    "observation.force_threshold": 1,
    "observation.noisy_hole_pos": 3,
    "observation.fingertip_pos": 3,
    "observation.fingertip_quat": 4,
    "observation.peg_pos": 3,
    "observation.peg_quat": 4,
}
FORGE_COLUMNS = COLUMNS


# ---------------------------------------------------------------------- the fingerprint
def _leaves(data: Any, prefix: str = "") -> Dict[str, Any]:
    if isinstance(data, dict):
        out = {}
        for key, value in data.items():
            out.update(_leaves(value, f"{prefix}.{key}" if prefix else str(key)))
        return out
    return {prefix: data}


def _allowed(path: str) -> bool:
    return any(path == a or (a.endswith((".", "_")) and path.startswith(a)) for a in TASK_DIFF_ALLOWED)


def demo_fingerprint(cfg: Any) -> Dict[str, Any]:
    """What a demo-transition file depends on, from a loaded config, as plain data."""
    from omegaconf import OmegaConf

    root = OmegaConf.to_container(cfg.root, resolve=True)
    task = {k: v for k, v in _leaves(root["task"]["cfg"]).items() if not _allowed(k)}
    return {
        "task_name": cfg.task_name,
        "task_cfg": task,
        "action": root["action"],
        "residual": root["residual"],
        "history": root["history"],
        "hybrid": root["hybrid"],
    }


def check_demo_transitions(cfg: Any) -> None:
    """Refuse ``sac.demo_transitions`` when it was made for another setup. A missing file is
    left to the learner, which needs it only to train (eval and debug never read it).

    A config rebuilt from a run (``cfg.source_run``: eval, debug, scripts/watch_residual_eval.py)
    is not checked: the run checked its file when it trained, nothing it builds reads the file, and
    its ``task.cfg`` is the live env's, holding what the env derived at ``gym.make`` (the gripper's
    and the IK's action terms), which the fingerprint -- taken from the config before ``gym.make`` --
    never holds. Training never starts from a run."""
    path = getattr(getattr(cfg, "sac", None), "demo_transitions", None)
    if not path or cfg.trainer.learner != "sac" or not Path(path).is_file() or cfg.source_run is not None:
        return
    import torch

    meta = torch.load(path, map_location="cpu", weights_only=False).get("meta", {})
    made = meta.get("fingerprint")
    if made is None:
        raise ValueError(f"sac.demo_transitions ({path}) has no meta.fingerprint; regenerate it with scripts/make_demo_transitions.py")
    want = demo_fingerprint(cfg)
    if made != want:
        made_l, want_l = _leaves(made), _leaves(want)
        defaults = _section_defaults()
        diff = sorted(
            k for k in set(made_l) | set(want_l)
            if made_l.get(k, "<unset>") != want_l.get(k, "<unset>")
            # a file made before a field was allowed to differ still holds it
            and not (k.startswith("task_cfg.") and _allowed(k[len("task_cfg."):]))
            # a field added after the file was made, at its default: what the file was made with
            and not (k not in made_l and k in defaults and want_l.get(k) == defaults[k])
        )
        if not diff:
            return
        shown = "; ".join(f"{k}: file {made_l.get(k, '<unset>')!r}, run {want_l.get(k, '<unset>')!r}" for k in diff[:8])
        raise ValueError(
            f"sac.demo_transitions ({path}) was made for a different setup ({len(diff)} difference(s): "
            f"{shown}). Regenerate it: python scripts/make_demo_transitions.py --config <this experiment>"
        )


def _section_defaults() -> Dict[str, Any]:
    """The project sections' field defaults, keyed like the fingerprint's leaves."""
    import dataclasses

    from my_project.cfg import ActionCfg, HistoryCfg, HybridCfg, ResidualCfg

    out = {}
    for section, cls in (("action", ActionCfg), ("residual", ResidualCfg), ("history", HistoryCfg), ("hybrid", HybridCfg)):
        for f in dataclasses.fields(cls):
            if f.default is not dataclasses.MISSING:
                out[f"{section}.{f.name}"] = f.default
            elif f.default_factory is not dataclasses.MISSING:
                out[f"{section}.{f.name}"] = f.default_factory()
    return out


# ---------------------------------------------------------------------- the reward
def forge_rewards(env_cfg: Any, *, held_pos, held_quat, fixed_pos, fixed_quat, actions, prev_actions,
                  hole_estimate, fingertip_pos, fingertip_quat, force, force_threshold) -> Dict[str, Any]:
    """Forge's reward for a batch of steps, from the true state after each step.

    Every argument is ``(rows, width)``: the state after the step (peg and hole base poses,
    fingertip pose, smoothed wrench's force), Forge's smoothed action of the step and of the
    step before, the per-episode hole estimate and contact threshold. Returns ``{"total",
    <term>...}`` per row, terms as Forge logs them (unscaled). Factory's terms and success
    checks are Factory's own methods, called on a stand-in for the env; Forge's three terms are
    its ``_get_rewards`` lines. Two differences from the live env, both exact for one env:
    ``action_penalty_ee`` is a per-row norm (Factory's is a norm over the whole batch), and the
    success-prediction term is 0, which ``delay_until_ratio > 1`` guarantees (it is refused
    otherwise: its scale switches on from a batch statistic).
    """
    import numpy as np
    import torch
    import isaacsim.core.utils.torch as torch_utils
    from isaaclab_tasks.direct.factory import factory_utils
    from isaaclab_tasks.direct.factory.factory_env import FactoryEnv

    task, ctrl = env_cfg.task, env_cfg.ctrl
    if task.delay_until_ratio <= 1.0:
        raise ValueError(
            f"task.cfg.task.delay_until_ratio is {task.delay_until_ratio}: Forge's success-prediction "
            "penalty then switches on from the batch's success rate, which no recorded frame holds"
        )
    rows = held_pos.shape[0]
    stand_in = SimpleNamespace(
        held_pos=held_pos, held_quat=held_quat, fixed_pos=fixed_pos, fixed_quat=fixed_quat,
        actions=actions, prev_actions=prev_actions, cfg_task=task, num_envs=rows, device=held_pos.device,
        # success with check_rot (nut threading) reads the fingertip's yaw
        fingertip_midpoint_quat=fingertip_quat,
    )
    stand_in._get_curr_successes = lambda success_threshold, check_rot=False: FactoryEnv._get_curr_successes(
        stand_in, success_threshold, check_rot
    )
    check_rot = task.name == "nut_thread"
    successes = stand_in._get_curr_successes(task.success_threshold, check_rot=check_rot)
    terms, scales = FactoryEnv._get_factory_rew_dict(stand_in, successes)
    terms["action_penalty_ee"] = torch.norm(actions, p=2, dim=-1)

    # Forge's action penalty: the preclipped target against the fingertip (Forge's _apply_action)
    bounds = torch.tensor(ctrl.pos_action_bounds, device=held_pos.device)
    delta_pos = hole_estimate + actions[:, 0:3] * bounds - fingertip_pos
    yaw_action = np.deg2rad(-180.0) + np.deg2rad(270.0) * (actions[:, 5] + 1.0) / 2.0
    zeros = torch.zeros_like(yaw_action)
    bolt_quat = torch_utils.quat_from_euler_xyz(roll=zeros, pitch=zeros, yaw=yaw_action)
    flip = torch_utils.quat_from_euler_xyz(
        roll=torch.full_like(zeros, np.pi), pitch=zeros, yaw=zeros
    )
    target_quat = torch_utils.quat_mul(flip, bolt_quat)
    curr_yaw = factory_utils.wrap_yaw(torch_utils.get_euler_xyz(fingertip_quat)[2])
    desired_yaw = factory_utils.wrap_yaw(torch_utils.get_euler_xyz(target_quat)[2])
    delta_yaw = desired_yaw - curr_yaw
    terms["action_penalty_asset"] = (
        torch.norm(delta_pos, p=2, dim=-1) / ctrl.pos_action_threshold[0]
        + torch.abs(delta_yaw) / ctrl.rot_action_threshold[0]
    )
    scales["action_penalty_asset"] = -task.action_penalty_asset_scale
    terms["contact_penalty"] = torch.nn.functional.relu(force.norm(dim=-1) - force_threshold.reshape(-1))
    scales["contact_penalty"] = -task.contact_penalty_scale

    total = torch.zeros(rows, device=held_pos.device)
    for name, value in terms.items():
        total = total + value * scales[name]
    return {"total": total, **terms}


# ---------------------------------------------------------------------- entry point
def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", required=True, help="the residual experiment the transitions are for")
    parser.add_argument("--dataset", default=None, help="demo dataset (default: the base policy's bc.dataset)")
    parser.add_argument("--out", default=None, help="output file (default: the config's sac.demo_transitions)")
    parser.add_argument("--check_in_sim", type=int, default=0, metavar="N",
                        help="also replay the first N demos in the training env and compare rewards")
    return parser


def main(argv=None, setup=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(parser)

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args, overrides = parser.parse_known_args(argv)
    args.headless = True
    app_launcher = AppLauncher(args)
    try:
        code = _make(args, overrides, setup)
        sys.stdout.flush()
        app_launcher.app.close()
        return code
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


def _make(args, overrides, setup) -> int:
    if setup is not None:
        setup()

    import torch
    from omegaconf import OmegaConf

    from robonuke_rl_core.config import load_config
    from robonuke_rl_core.evaluation import resolved_config_path

    from my_project.residual import base_run_dir, base_source, load_base_policy

    cfg = load_config(args.config, list(overrides) + ["task.cfg.scene.num_envs=1"])
    if not cfg.residual.enabled or cfg.trainer.learner != "sac":
        raise ValueError(f"{args.config}: demo transitions are for a SAC residual (residual.enabled, trainer.learner: sac)")
    out = Path(args.out) if args.out else (Path(cfg.sac.demo_transitions) if cfg.sac.demo_transitions else None)
    if out is None:
        raise ValueError(f"{args.config} sets no sac.demo_transitions; pass --out")
    base_dir = base_run_dir(cfg)  # the local run, or the wandb run downloaded once
    if args.dataset:
        dataset = Path(args.dataset).expanduser()
    else:
        base_cfg = OmegaConf.load(resolved_config_path(base_dir))
        dataset = Path(base_cfg.bc.dataset).expanduser()
    lines = (dataset / "meta" / "episodes.jsonl").read_text().splitlines()
    episodes = [json.loads(line) for line in lines if line]
    if not episodes:
        raise ValueError(f"{dataset} has no episodes")
    from my_project.families import FORGE, family_of

    family = family_of(cfg.task_name)
    if family == FORGE:
        _check_sessions(dataset, episodes, cfg)
    elif args.check_in_sim:
        raise ValueError("--check_in_sim replays Forge recordings; an imported FurnitureBench demo has no env snapshot to replay from")

    # the check's env is built first, under the recording session's seed: a demo replays
    # exactly only in an env made like the recorder's (measured: 0.0 mm under the session seed,
    # 4-25 mm drift under another seed or when built after the conversion)
    check_env = _build_check_env(args, overrides, episodes[: args.check_in_sim]) if args.check_in_sim else None
    # the env's resets draw from torch's global RNG: keep its state as the env left it, so the
    # conversion below (seeded, sampling the diffusion base) cannot change what they draw
    rng = (torch.random.get_rng_state(), torch.cuda.get_rng_state_all()) if check_env is not None else None

    torch.manual_seed(int(cfg.experiment.seed))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy = load_base_policy(base_dir, cfg.residual.base_checkpoint, device)
    from my_project.families import is_abs

    absolute = is_abs(cfg.task_name)
    frames = [_read_episode(dataset, e, policy.obs_dim, device, family, absolute) for e in episodes]
    limit = _episode_limit(cfg)
    for episode, f in zip(episodes, frames):
        if f["transitions"] > limit:
            raise ValueError(
                f"episode {episode['episode_index']} has {f['action'].shape[0]} frames; the experiment's "
                f"episodes end after {limit} steps: lengthen task.cfg.episode_length_s or drop the demo"
            )

    transitions, stats = convert(cfg, policy, episodes, frames, device)
    meta = {
        "fingerprint": demo_fingerprint(cfg),
        "dataset": str(dataset),
        "base": f"{base_source(cfg)}/{cfg.residual.base_checkpoint}",
        "base_step": int(getattr(policy, "checkpoint_step", -1)),
        **{k: v for k, v in stats.items() if k not in ("rewards_by_episode", "terms_by_episode")},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"transitions": {k: v.cpu() for k, v in transitions.items()}, "meta": meta}, out)
    print(
        f"[demos] {meta['transitions']} transitions from {meta['demo_steps']} demo steps of "
        f"{len(episodes)} demos -> {out}\n"
        f"[demos] mean return per demo {meta['mean_return']:.2f}; success reached in "
        f"{meta['success_rate']:.0%} of demos\n[demos] residual action beyond the policy's [-1, 1] "
        f"on {meta['beyond_scale_fraction']:.1%} of demo steps ({meta['demo_actions_beyond_scale']}d); "
        f"per dim: " + ", ".join(f"{v:.1%}" for v in meta["beyond_scale_fraction_per_dim"])
        + "; largest |action| per dim: " + ", ".join(f"{v:.2f}" for v in meta["max_abs_action"]),
        flush=True,
    )
    if check_env is not None:
        torch.random.set_rng_state(rng[0])
        torch.cuda.set_rng_state_all(rng[1])
        _check_in_sim(check_env, dataset, episodes[: args.check_in_sim], frames, stats)
    return 0


def _build_check_env(args, overrides, episodes):
    """The experiment's training env (one env) under the demos' recording-session seed."""
    import gymnasium as gym
    from robonuke_rl_core.config import load_config
    from robonuke_rl_core.envs.build import build_env, prepare_task, wrap_for_skrl

    seeds = {int(e["seed"]) for e in episodes}
    if len(seeds) != 1:
        raise ValueError(f"--check_in_sim: the demos checked come from sessions with seeds {sorted(seeds)}; check one session at a time")
    # a demo replays exactly only in recording order from its session's start (restore does not
    # reproduce PhysX's internal state; each reset must recreate the recorded one)
    sessions = [e["session"] for e in episodes]
    first = min(i for i, e in enumerate(episodes) if e["session"] == sessions[0])
    if first != 0 or len(set(sessions)) != 1:
        raise ValueError("--check_in_sim replays the first demos of one session, in order")
    cfg = load_config(args.config, list(overrides) + ["task.cfg.scene.num_envs=1", f"experiment.seed={seeds.pop()}"])
    prepare_task(cfg, cfg.task_name, cfg.task_cfg)
    return wrap_for_skrl(build_env(cfg, gym.make(cfg.task_name, cfg=cfg.task_cfg), cfg.task_name), cfg.experiment.seed)


def _episode_limit(cfg) -> int:
    """Steps in one of the experiment's episodes (Forge ends one before ``max_episode_length``;
    a manager-based env at it)."""
    import math

    from my_project.families import FORGE, family_of

    task = cfg.task_cfg
    steps = math.ceil(task.episode_length_s / (task.sim.dt * task.decimation))
    return steps - 1 if family_of(cfg.task_name) == FORGE else steps


def _read_columns(dataset: Path, episode: dict, columns: Dict[str, Any], device) -> Dict[str, Any]:
    import pyarrow.parquet as pq
    import torch

    table = pq.read_table(dataset / episode["file"])
    out = {}
    for name, width in columns.items():
        if name not in table.column_names:
            raise KeyError(f"{dataset / episode['file']} has no column {name!r}: recorded before it existed?")
        column = table.column(name).combine_chunks()
        values = (column.flatten() if hasattr(column, "flatten") else column).to_numpy()  # vector or scalar
        values = torch.from_numpy(values.reshape(table.num_rows, -1).copy()).to(device, torch.float32)
        if width is not None and values.shape[1] != width:
            raise ValueError(f"{episode['file']}: {name} is {values.shape[1]} wide, expected {width}")
        out[FRAME_KEYS.get(name, name.split(".", 1)[-1])] = values
    return out


def _read_episode(
    dataset: Path, episode: dict, obs_dim: int, device, family: str = "forge", absolute: bool = False
) -> Dict[str, Any]:
    """One demo's frames: ``state`` (one more than the transitions), ``critic``, ``action``,
    ``transitions`` (how many ``t -> t+1`` the demo yields) and the family's reward material."""
    import torch

    from my_project.families import FURNITURE

    if family == FURNITURE:
        out = _read_columns(dataset, episode, FURNITURE_ABS_COLUMNS if absolute else FURNITURE_COLUMNS, device)
        if out["state"].shape[1] != obs_dim:
            raise ValueError(f"{episode['file']}: observation.state is {out['state'].shape[1]} wide; the base policy takes {obs_dim}")
        if out["next_state"].shape[1] != obs_dim:
            raise ValueError(f"{episode['file']}: observation.next_state is {out['next_state'].shape[1]} wide, expected {obs_dim}")
        # every action has its next state: T transitions over T + 1 states
        out["transitions"] = out["action"].shape[0]
        out["state"] = torch.cat((out["state"], out["next_state"][-1:]), dim=0)
        out["critic"] = out["state"]  # the env has no critic group
        out["success"] = bool(episode.get("success", False))
        return out
    out = _read_columns(dataset, episode, FORGE_COLUMNS, device)
    if out["state"].shape[1] != obs_dim:
        raise ValueError(f"{episode['file']}: observation.state is {out['state'].shape[1]} wide; the base policy takes {obs_dim}")
    if torch.isnan(out["applied"]).any():
        raise ValueError(f"{episode['file']}: action.applied has NaN rows (an episode that hit the time limit)")
    state = torch.load(dataset / episode["state_file"], map_location=device, weights_only=False)
    out["prev_applied"] = state["attrs"]["prev_actions"].to(device, torch.float32).reshape(1, -1)
    out["transitions"] = out["action"].shape[0] - 1  # the last frame's next state was never recorded
    return out


def _state_slices(env_cfg) -> Dict[str, slice]:
    from isaaclab_tasks.direct.factory.factory_env_cfg import STATE_DIM_CFG

    slices, start = {}, 0
    for name in env_cfg.state_order:
        slices[name] = slice(start, start + STATE_DIM_CFG[name])
        start += STATE_DIM_CFG[name]
    return slices


def convert(cfg: Any, policy, episodes: List[dict], frames: List[dict], device):
    """The transitions and their stats; see the module docstring."""
    import torch

    from my_project.models.diffusion_policy import ChunkedBase
    from my_project.residual import residual_observation, residual_scale
    from my_project.wrappers import HistoryBuffer, history_index, hybrid_demo_block

    from my_project.families import FORGE, family_of

    env_cfg = cfg.task_cfg
    forge = family_of(cfg.task_name) == FORGE
    if forge:
        slices = _state_slices(env_cfg)
        for name in ("fixed_pos", "fixed_quat"):
            if name not in slices:
                raise ValueError(f"task.cfg.state_order must hold {name}: the reward needs the hole's base pose, from the recorded critic state")
    scale = residual_scale(policy, cfg.residual.action_scale, cfg.residual.action_scale_fraction).to(device)
    n, longest = len(frames), max(f["state"].shape[0] for f in frames)

    # the base on every demo at once, each on its own chunk clock from its first frame; a
    # finished demo keeps repeating its last frame (its outputs are not used)
    base = ChunkedBase(policy, n)
    history = (
        HistoryBuffer(history_index(env_cfg, cfg.history.terms, cfg.task_name), cfg.history.length, n, device)
        if cfg.history.enabled
        else None
    )
    agent_obs, agent_state, base_actions = [], [], []
    for t in range(longest):
        obs = torch.stack([f["state"][min(t, f["state"].shape[0] - 1)] for f in frames])
        critic = torch.stack([f["critic"][min(t, f["critic"].shape[0] - 1)] for f in frames])
        fresh = torch.full((n,), t == 0, dtype=torch.bool, device=device)
        a_base = base.actions(obs, fresh)
        policy_part, base_n = residual_observation(policy.normalizer, obs, a_base)
        state_part = torch.cat((critic, base_n), dim=-1)
        if history is not None:
            past = history.step(policy_part, fresh)
            policy_part = torch.cat((policy_part, past), dim=-1)
            state_part = torch.cat((state_part, past), dim=-1)
        agent_obs.append(policy_part)
        agent_state.append(state_part)
        base_actions.append(a_base)

    columns: Dict[str, list] = {k: [] for k in (
        "observations", "actions", "rewards", "next_observations", "terminated", "truncated", "states", "next_states")}
    beyond_dims = torch.zeros(scale.numel(), device=device)
    beyond_rows, kept_rows, per_episode = 0, 0, []
    drop = cfg.residual.demo_actions_beyond_scale == "drop"
    for e, f in enumerate(frames):
        steps = f["transitions"]  # transitions t -> t+1
        now, nxt = slice(0, steps), slice(1, steps + 1)
        residual = (f["action"][now] - torch.stack([base_actions[t][e] for t in range(steps)])) / scale
        beyond = residual.abs() > 1.0
        beyond_dims += beyond.float().sum(dim=0)
        beyond_rows += int(beyond.any(dim=-1).sum())
        keep = ~beyond.any(dim=-1) if drop else torch.ones(steps, dtype=torch.bool, device=device)
        kept_rows += int(keep.sum())
        terminated = torch.zeros(steps, 1, dtype=torch.bool, device=device)
        if forge:
            critic_next = f["critic"][nxt]
            reward = forge_rewards(
                env_cfg,
                held_pos=f["peg_pos"][nxt],
                held_quat=f["peg_quat"][nxt],
                fixed_pos=critic_next[:, slices["fixed_pos"]],
                fixed_quat=critic_next[:, slices["fixed_quat"]],
                actions=f["applied"][now],
                prev_actions=torch.cat((f["prev_applied"], f["applied"][: steps - 1]), dim=0),
                hole_estimate=f["noisy_hole_pos"][now],
                fingertip_pos=f["fingertip_pos"][nxt],
                fingertip_quat=f["fingertip_quat"][nxt],
                force=f["wrench"][nxt, 0:3],
                force_threshold=f["force_threshold"][now],
            )
        else:  # the recorded (sparse) reward; the env ends a successful episode on assembly
            # Isaac Lab's reward manager pays weight x step_dt: the recorded 1 is 1 x step_dt online
            total = f["reward"][now, 0] * float(env_cfg.sim.dt * env_cfg.decimation)
            reward = {"total": total, "curr_success": total > 0}
            terminated[-1, 0] = f["success"]
        columns["observations"].append(torch.stack([agent_obs[t][e] for t in range(steps)])[keep])
        columns["next_observations"].append(torch.stack([agent_obs[t + 1][e] for t in range(steps)])[keep])
        columns["states"].append(torch.stack([agent_state[t][e] for t in range(steps)])[keep])
        columns["next_states"].append(torch.stack([agent_state[t + 1][e] for t in range(steps)])[keep])
        if cfg.hybrid.enabled:
            # the demos ran under position control: every axis selects position, no force target
            residual = torch.cat((residual, hybrid_demo_block(cfg.hybrid.axes_for(cfg), steps, device)), dim=-1)
        columns["actions"].append(residual[keep])
        columns["rewards"].append(reward["total"].reshape(-1, 1)[keep])
        columns["terminated"].append(terminated[keep])
        columns["truncated"].append(torch.zeros(steps, 1, dtype=torch.bool, device=device)[keep])
        per_episode.append({
            "episode_index": episodes[e]["episode_index"],
            "steps": steps,
            "return": float(reward["total"].sum()),
            "success": bool(reward["curr_success"].any()),
            "rewards": reward["total"].tolist(),
            "terms": {k: v.tolist() for k, v in reward.items() if k != "total"},
        })

    transitions = {k: torch.cat(v) for k, v in columns.items()}
    total = sum(p["steps"] for p in per_episode)
    stats = {
        "transitions": kept_rows,
        "demo_steps": total,
        "demo_actions_beyond_scale": cfg.residual.demo_actions_beyond_scale,
        "episodes": [{k: v for k, v in p.items() if k != "rewards"} for p in per_episode],
        "mean_return": sum(p["return"] for p in per_episode) / n,
        "success_rate": sum(p["success"] for p in per_episode) / n,
        "beyond_scale_fraction": beyond_rows / total,
        "beyond_scale_fraction_per_dim": (beyond_dims / total).tolist(),
        "max_abs_action": (
            transitions["actions"][:, : scale.numel()].abs().amax(dim=0).tolist() if kept_rows else [0.0] * scale.numel()
        ),
        "residual_scale": scale.tolist(),
        # each demo's per-step offline reward, for --check_in_sim
        "rewards_by_episode": [p["rewards"] for p in per_episode],
        "terms_by_episode": [p["terms"] for p in per_episode],
    }
    return transitions, stats


def _check_in_sim(env, dataset: Path, episodes: List[dict], frames: List[dict], stats) -> None:
    """Replay demos through the training env and compare its reward with the offline one."""
    import torch

    from my_project.demos.state import restore
    from my_project.residual import ResidualPolicyWrapper

    residual = _find(env, ResidualPolicyWrapper)
    worst, compared = 0.0, 0
    u = env.unwrapped
    for episode, f, offline, terms in zip(episodes, frames, stats["rewards_by_episode"], stats["terms_by_episode"]):
        state = torch.load(dataset / episode["state_file"], map_location="cpu", weights_only=False)
        # a plain reset of the gym stack, as the recorder and replay_demos.py do: skrl's
        # force_env_reset draws differently, and a restored start replays exactly only after the
        # same reset the recording had (measured: 0.0 mm vs 4-25 mm)
        env._env.reset()
        restore(env, state)
        actions = f["action"].to(residual.scale.device)
        if residual.extra_dims:
            from my_project.wrappers import hybrid_demo_block

            actions = torch.cat((actions, hybrid_demo_block(residual.extra_dims // 2, actions.shape[0], actions.device)), dim=-1)
        diffs, term_diffs, drift = [], {}, 0.0
        for t, expected in enumerate(offline):
            # the demo's own action, bit for bit, to the env under the residual wrapper: the
            # round trip base + scale * ((a - base) / scale) is off by float rounding, which
            # contact amplifies into a different trajectory
            _, reward, terminated, truncated, infos = residual.env.step(actions[t : t + 1])
            diffs.append(abs(float(reward) - expected))
            logged = infos.get("metrics_to_log", {}) if isinstance(infos, dict) else {}
            for name, values in terms.items():
                key = f"reward/{name}"
                if key in logged:
                    gap = abs(float(logged[key].reshape(-1)[0]) - values[t])
                    term_diffs[name] = max(term_diffs.get(name, 0.0), gap)
            # the physics against the recording: the peg after step t is frame t+1's
            drift = max(drift, float((u.held_pos[0].cpu() - f["peg_pos"][t + 1].cpu()).abs().max()))
            if bool(terminated) or bool(truncated):
                break
        if drift > 1e-4:
            # the residual's base samples the diffusion policy every replan, drawing from the
            # same RNG the env's resets use, so after the first demo the resets no longer recreate
            # the recorded ones: such a replay says nothing about the reward reconstruction
            print(
                f"[check] episode {episode['episode_index']}: physics did not replay exactly (peg "
                f"{drift * 1000:.3f} mm off the recording): not a reward comparison, skipped",
                flush=True,
            )
            continue
        worst, compared = max(worst, max(diffs)), compared + 1
        print(
            f"[check]   peg drift from the recording {drift * 1000:.3f} mm; largest per-term gaps: "
            + ", ".join(f"{k} {v:.2e}" for k, v in sorted(term_diffs.items(), key=lambda kv: -kv[1])),
            flush=True,
        )
        print(
            f"[check] episode {episode['episode_index']}: {len(diffs)} steps, |sim - offline| reward "
            f"max {max(diffs):.2e}, mean {sum(diffs) / len(diffs):.2e}",
            flush=True,
        )
    if not compared:
        raise RuntimeError("--check_in_sim: no demo replayed exactly, so no reward was compared")
    print(f"[check] largest per-step reward difference over {compared} exactly replayed demo(s): {worst:.2e}", flush=True)
    env.close()


def _check_sessions(dataset: Path, episodes, cfg) -> None:
    """Every recording session's task config must match the experiment's, up to
    :data:`TASK_DIFF_ALLOWED` and :data:`TASK_NARROWER_ALLOWED`, and so must its action
    interface."""
    from omegaconf import OmegaConf

    want_root = OmegaConf.to_container(cfg.root, resolve=True)
    want = _leaves(want_root["task"]["cfg"])
    for session in sorted({e["session"] for e in episodes}):
        recorded = OmegaConf.to_container(OmegaConf.load(dataset / "meta" / "sessions" / f"{session}.yaml"), resolve=True)
        if recorded.get("task", {}).get("name") != cfg.task_name:
            raise ValueError(f"session {session} recorded {recorded.get('task', {}).get('name')!r}, the experiment runs {cfg.task_name!r}")
        got = _leaves(recorded["task"]["cfg"])
        diff = sorted(
            k for k in set(got) | set(want)
            if not _allowed(k) and got.get(k, "<unset>") != want.get(k, "<unset>") and not _narrower(k, got, want)
        )
        if diff:
            shown = "; ".join(f"{k}: recorded {got.get(k, '<unset>')!r}, experiment {want.get(k, '<unset>')!r}" for k in diff[:8])
            raise ValueError(f"session {session} recorded a different task config ({len(diff)} difference(s): {shown})")
        made, used = recorded.get("action") or {}, want_root["action"]
        defaults = _section_defaults()
        # a field added after the recording, at its default: what the recording was made with
        if any(
            made.get(k, "<unset>") != used.get(k, "<unset>")
            and not (k not in made and used.get(k) == defaults.get(f"action.{k}", "<no default>"))
            for k in set(made) | set(used)
        ):
            raise ValueError(f"session {session} recorded action {recorded.get('action')}, the experiment uses {want_root['action']}")


def _narrower(path: str, got: Dict[str, Any], want: Dict[str, Any]) -> bool:
    """A :data:`TASK_NARROWER_ALLOWED` range recorded no wider than the experiment's, per value."""
    if path not in TASK_NARROWER_ALLOWED or path not in got or path not in want:
        return False
    recorded, experiment = got[path], want[path]
    if not isinstance(recorded, list) or not isinstance(experiment, list) or len(recorded) != len(experiment):
        return False
    return all(abs(float(r)) <= abs(float(e)) for r, e in zip(recorded, experiment))


def _find(env: Any, cls: type):
    node, seen = env, set()
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        if isinstance(node, cls):
            return node
        node = getattr(node, "_env", None) or getattr(node, "env", None)
    raise RuntimeError(f"the env has no {cls.__name__}")
