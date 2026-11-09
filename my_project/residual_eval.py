"""Evaluate a residual run's checkpoints as they appear, against its base policy alone.

    python scripts/watch_residual_eval.py --run <residual run dir> --eval_config <eval config>

Runs beside training (or after it, ``--once``), in its own Isaac process. ``--run`` is one agent's
directory (``<group>/<group>_a<i>``, holds ``checkpoints/``) or the **group directory** of a run
that trained several agents in one process (``experiment.num_agents``): then every agent of the
run is evaluated side by side, each in its own slot of one N-agent eval learner, on its own
block of envs, exactly the split training used. The env is the run's own (its resolved config)
with ``--eval_config`` on top (required: the eval conditions, e.g. the env count, which must divide
by the agent count), built through the same project wrappers training uses. Then:

1. **Step 0, the base alone**, right at startup: a residual of exactly zero, so the frozen
   diffusion policy acts by itself (no checkpoint needed). The same policy for every agent, so
   it runs once and is recorded for each.
2. **Every numbered checkpoint** (``ckpt_<step>.ckpt``) not yet evaluated, oldest first, once
   every agent has written its file of that step, with the residual on, the SAC policy
   deterministic (its mean action) or, with ``--stochastic``, sampled as in training; new ones
   are picked up as training writes them. It stops after the checkpoint at or past the run's
   ``trainer.total_timesteps``, with ``--once`` when no checkpoint on disk is left, or on Ctrl+C.

The two action modes measure different policies: training only ever values sampled actions,
so the mean alone can be much worse (measured on these runs: 15–40 points lower).

An evaluation is ``ceil(eval.num_rollouts / E)`` rounds, ``E`` the envs per agent: in a round all
envs reset together under one seed (round ``k`` under ``seed + k``, so every checkpoint sees the
same starting conditions) and step for the episode's full length; each env contributes exactly
one episode, and the first ``num_rollouts`` episodes of each agent count. Per episode, from the
true (noise-free) sensor force:

* success: the env's success check held at any step; engaged: its engage check did;
* the mean and the maximum contact-force magnitude over the episode (N).

On a FurnitureBench run (``Isaac-Furniture-*``, a manager-based env that ends an episode on
assembly or a part out of bounds) each env's FIRST episode of the round counts: success is the
env's ``success`` termination firing in it, plus the share that ended out of bounds and the
mean episode length; there is no force or engagement. Several agents at once are supported on
FurnitureBench envs (the ablations' case); a Forge or standalone env evaluates one agent.

With ``residual.steps_per_chunk: k > 1`` one ``env.step`` is ``k`` sim steps: a round is
``ceil(steps / k)`` agent steps, the success / engagement / force reads happen after each agent
step (every ``k`` sim steps), and an episode's length counts sim steps (the wrapper's ``counted``).

Logged per agent to its own wandb run (``<agent>_eval``, the training run's project and group), at
``step`` = the checkpoint's step: ``eval/success_rate``, ``eval/engaged_rate``,
``eval/force_mean`` (mean over episodes of each episode's mean force), ``eval/force_max_mean``,
and histograms ``eval/force_max_hist`` and ``eval/force_mean_hist``; on a FurnitureBench env that tracks one_leg's
phases, also ``eval/phase_success/<phase>`` (each phase's success among the episodes that reached it, the env's
``phase/passed_*``; ``screw`` is ``passed_success``), ``eval/phase_reached/<phase>`` (how many episodes reached it)
and ``eval/residual_mm/<phase>`` / ``eval/residual_deg/<phase>`` (the mean correction over the agent steps that
started in that phase). ``--skip_base`` leaves out the base-alone round (step 0), for one-off tests on checkpoints whose reference is already
known. Each result is also
appended to ``<agent dir>/eval_watch/results.jsonl``, which is also what a restarted watcher
resumes from (per agent). Each round shows its progress as a tqdm bar on stderr.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence



def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--run",
        required=True,
        help="the residual run: one agent's directory (holds checkpoints/), or the group directory of a run that "
        "trained several agents, to evaluate them all side by side",
    )
    parser.add_argument("--eval_config", required=True, help="eval conditions layered on the run's config")
    parser.add_argument("--poll_seconds", type=float, default=30.0, help="how often to look for new checkpoints")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="sample the SAC policy's actions (as training does) instead of its mean; results go to "
        "eval_watch_stochastic/ and their own wandb run (<run>_eval_stochastic)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="evaluate the checkpoints already on disk, then exit (for a run that has stopped)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=None,
        help="evaluate the base and only these checkpoint steps (each must be on disk), then exit",
    )
    parser.add_argument(
        "--skip_base",
        action="store_true",
        help="skip the base-alone round (step 0): for one-off tests on checkpoints whose reference is already known",
    )
    parser.add_argument(
        "--out_name",
        default=None,
        help="results folder in the run dir and wandb run suffix (default eval_watch / _eval; "
        "eval_watch_stochastic / _eval_stochastic with --stochastic): keeps a separate evaluation apart",
    )
    return parser


def main(argv=None, setup=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(parser)

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args, overrides = parser.parse_known_args(argv)
    args.headless = True
    from robonuke_rl_core.config import simulator_of, start_simulator
    from robonuke_rl_core.evaluation import resolved_config_path

    # Isaac Sim starts only for a task it simulates (the run's task.simulator)
    run_config = argparse.Namespace(config=None, from_run=str(resolved_config_path(Path(args.run).expanduser())))
    start_simulator(args, simulator_of(run_config, overrides))
    try:
        return _watch(args, overrides, setup)
    except KeyboardInterrupt:
        print("[watch] stopped", flush=True)
        sys.stdout.flush()
        os._exit(0)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


#: a checkpoint is read only once its file has gone this long unmodified: training writes it in
#: place (torch.save, ~730 MB for the nut residual), so a fresh file can still be half written
SETTLED_SECONDS = 60.0


def _numbered_checkpoints(directory: Path) -> Dict[int, Path]:
    found = {}
    now = time.time()
    for path in directory.glob("ckpt_*.ckpt"):
        match = re.fullmatch(r"ckpt_(\d+)\.ckpt", path.name)
        if match and now - path.stat().st_mtime >= SETTLED_SECONDS:
            found[int(match.group(1))] = path
    return found


# ---------------------------------------------------------------------- the run's agents
def agent_dirs(run_dir: Path, run_names: Sequence[str]) -> List[Path]:
    """The agent directories ``--run`` names, in slot order.

    ``run_dir`` is an **agent** directory when its name is one of the run's ``derived.run_names``
    (``<group>_a<i>``): one slot, that agent. Otherwise it is the **group** directory and every
    agent of the run is evaluated: ``<run_dir>/<name>`` for each run name, all of which must exist.
    """
    run_dir = Path(run_dir)
    names = [str(name) for name in run_names]
    if not names:
        raise ValueError(f"{run_dir}: the run's resolved config names no agents (derived.run_names)")
    if run_dir.name in names:
        return [run_dir]
    agents = [run_dir / name for name in names]
    missing = [str(path) for path in agents if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            f"--run {run_dir} is neither an agent directory (one of {names}) nor a group directory holding "
            f"every agent: missing {missing}"
        )
    return agents


def checkpoint_dirs(agents: Sequence[Path], routes: Sequence[str]) -> List[Path]:
    """Where each agent's numbered checkpoints are: ``<agent>/checkpoints``, or for a routed learner (one residual
    per phase, each phase writing its own checkpoints of the same names) the first phase's folder,
    ``<agent>/<routes[0]>/checkpoints``; every phase's file of a name is loaded together."""
    return [Path(agent) / routes[0] / "checkpoints" if routes else Path(agent) / "checkpoints" for agent in agents]


def pending_steps(ckpt_dirs: Sequence[Path], done: Sequence[set]) -> List[int]:
    """The checkpoint steps to evaluate next, oldest first: every agent has a settled file of that step on disk
    (``SETTLED_SECONDS``; the agents of one training process write together, so this waits for the slowest
    file only), and at least one agent has not recorded it yet (``done``: per agent, the steps in its results)."""
    on_disk = [_numbered_checkpoints(Path(d)) for d in ckpt_dirs]
    common = set.intersection(*(set(found) for found in on_disk)) if on_disk else set()
    return sorted(step for step in common if any(step not in agent_done for agent_done in done))


def _find_residual(env: Any):
    from my_project.residual import find_residual

    return find_residual(env)


def _residual_or_none(env: Any):
    """The env's residual wrapper, or None: ``evaluate`` also runs a plain env (one env.step, one sim step)."""
    try:
        return _find_residual(env)
    except RuntimeError:
        return None


def _agent_steps(sim_steps: int, steps_per_chunk: int) -> int:
    """Agent steps that cover ``sim_steps`` sim steps when one agent step is ``steps_per_chunk`` of them."""
    return -(-int(sim_steps) // int(steps_per_chunk))


def _round_bar(total: int, label: str):
    """The round's progress on stderr: env steps out of the episode's full length."""
    from tqdm import tqdm

    return tqdm(total=total, desc=f"[watch] {label}", unit="step", dynamic_ncols=True, leave=False)


def _is_furniture(env: Any) -> bool:
    """A manager-based FurnitureBench env (ends episodes through its termination manager)."""
    return hasattr(getattr(env, "unwrapped", env), "termination_manager")


def evaluate(env: Any, policy: Any, seed: int, label: str = "eval", episodes: int | None = None) -> Dict[str, Any]:
    """One agent's evaluation: success, engagement, mean and max force per episode over one round on every env
    (FurnitureBench: :func:`evaluate_rounds`, ``ceil(episodes / num_envs)`` rounds; standalone:
    :func:`evaluate_episodes`). ``label`` names the round's progress bar."""
    import torch

    if _is_furniture(env):
        return evaluate_rounds(env, policy, seed, label, episodes, num_agents=1)[0]
    if getattr(type(env.unwrapped.cfg), "SIMULATOR", "isaac") == "standalone":
        return evaluate_episodes(env, policy, seed, label, episodes)

    from robonuke_rl_core.evaluation import env_states, force_env_reset

    from robonuke_rl_core.envs.build import reseed_env

    u = env.unwrapped
    n = int(u.num_envs)
    residual = _residual_or_none(env)
    # every generator the reset draws from (Isaac Sim's set_seed), so every evaluation really
    # gets the same starts; torch.manual_seed alone does not reach NumPy, Warp or Replicator
    reseed_env(env, seed)
    observations, states = force_env_reset(env)
    if hasattr(policy, "reset"):
        policy.reset()
    success = torch.zeros(n, dtype=torch.bool, device=u.device)
    engaged = torch.zeros(n, dtype=torch.bool, device=u.device)
    force_sum = torch.zeros(n, device=u.device)
    force_max = torch.zeros(n, device=u.device)
    # Forge ends an episode one step before the limit; one env.step is steps_per_chunk sim steps
    steps = _agent_steps(int(u.max_episode_length) - 1, residual.steps_per_chunk if residual is not None else 1)
    bar = _round_bar(steps, label)
    for _ in range(steps):
        with torch.no_grad():
            actions = policy(observations, states)
        observations, _, terminated, truncated, infos = env.step(actions)
        if hasattr(policy, "observe"):
            # a routed policy (sac.route_labels) picks each env's residual from the phase the env just published
            policy.observe(infos)
        states = env_states(env)
        force = u.force_sensor_smooth[:, 0:3].norm(dim=-1)
        force_sum += force
        force_max = torch.maximum(force_max, force)
        # the env's own rules (Factory/Forge _get_rewards): success on nut threading also needs
        # the wrist turned past cfg_task.ee_success_yaw; engagement never checks rotation
        success |= u._get_curr_successes(u.cfg_task.success_threshold, check_rot=u.cfg_task.name == "nut_thread")
        engaged |= u._get_curr_successes(u.cfg_task.engage_threshold, check_rot=False)
        bar.update(1)
    bar.close()
    force_mean = force_sum / steps
    return {
        "success_rate": float(success.float().mean()),
        "engaged_rate": float(engaged.float().mean()),
        "force_mean": float(force_mean.mean()),
        "force_max_mean": float(force_max.mean()),
        "force_max": force_max.cpu().tolist(),
        "force_means": force_mean.cpu().tolist(),
        "episodes": n,
    }


def evaluate_episodes(env: Any, policy: Any, seed: int, label: str = "eval", episodes: int | None = None) -> Dict[str, Any]:
    """Rounds on every env of a standalone task (``task.simulator: standalone``, e.g. robosuite Can) until ``episodes``
    (``eval.num_rollouts``; None: one round) have run, round ``k`` reset under ``seed + k``: each env's first episode
    of a round, its success the env's ``episode_metrics_to_log["episode/success"]`` on the step it ends, its length,
    and the residual's mean correction (``residual_mm/all``, ``residual_deg/all``). The furniture round's keys, so the
    watcher prints and logs it the same way (no part can leave its bounds here: ``out_of_bounds_rate`` 0). A task
    whose envs are heavy (GR1: ~0.4 GB each) evaluates 50 episodes on a few envs this way."""
    n = int(env.unwrapped.num_envs)
    rounds = max(1, -(-int(episodes) // n)) if episodes else 1
    results = [_episodes_round(env, policy, seed + k, label if rounds == 1 else f"{label} {k + 1}/{rounds}")
               for k in range(rounds)]
    # each round's per-episode outcomes, for the pooled rate; not a result of its own
    per_round = [r.pop("successes") for r in results]
    if rounds == 1:
        return results[0]
    lengths = [v for r in results for v in r["lengths"]][: int(episodes)]
    successes = [v for r in per_round for v in r][: int(episodes)]
    out = {"success_rate": sum(successes) / len(successes), "out_of_bounds_rate": 0.0,
           "length_mean": sum(lengths) / len(lengths), "lengths": lengths, "episodes": len(lengths)}
    for key in {k for r in results for k in r if k.startswith("residual_")}:
        values = [r[key] for r in results if key in r]
        out[key] = sum(values) / len(values)
    return out


def _episodes_round(env: Any, policy: Any, seed: int, label: str) -> Dict[str, Any]:
    """One round of :func:`evaluate_episodes`: every env's first episode, reset under ``seed``."""
    import torch

    from robonuke_rl_core.envs.build import reseed_env
    from robonuke_rl_core.evaluation import env_states, force_env_reset

    u = env.unwrapped
    n = int(u.num_envs)
    residual = _residual_or_none(env)
    reseed_env(env, seed)
    observations, states = force_env_reset(env)
    if hasattr(policy, "reset"):
        policy.reset()
    alive = torch.ones(n, dtype=torch.bool, device=u.device)
    success = torch.zeros(n, dtype=torch.bool, device=u.device)
    length = torch.zeros(n, dtype=torch.long, device=u.device)
    steps = _agent_steps(int(u.max_episode_length), residual.steps_per_chunk if residual is not None else 1)
    bar = _round_bar(steps, label)
    correction: Dict[str, List[float]] = {}
    for _ in range(steps):
        with torch.no_grad():
            actions = policy(observations, states)
        observations, _, terminated, truncated, infos = env.step(actions)
        states = env_states(env)
        ending = alive & (terminated.reshape(-1).bool() | truncated.reshape(-1).bool())
        if isinstance(infos, dict):
            metrics = infos.get("metrics_to_log") or {}
            for key, unit in (("residual/position_mm", "mm"), ("residual/rot_deg", "deg")):
                value = metrics.get(key)
                if value is not None and bool(alive.any()):
                    correction.setdefault(f"residual_{unit}/all", []).extend(
                        torch.as_tensor(value, device=u.device).reshape(-1)[alive].tolist())
            ended_success = (infos.get("episode_metrics_to_log") or {}).get("episode/success")
            if ended_success is not None:
                success |= ending & (torch.as_tensor(ended_success, device=u.device).reshape(-1) > 0.5)
        length += alive.long() * (residual.counted.to(length.device) if residual is not None else 1)
        alive &= ~ending
        bar.set_postfix(running=int(alive.sum()), success=int(success.sum()), refresh=False)
        bar.update(1)
        if not alive.any():
            break
    bar.close()
    result = {
        "success_rate": float(success.float().mean()),
        "out_of_bounds_rate": 0.0,
        "length_mean": float(length.float().mean()),
        "lengths": length.cpu().tolist(),
        "successes": success.float().cpu().tolist(),
        "episodes": n,
    }
    for key, values in sorted(correction.items()):
        result[key] = sum(values) / len(values)
    return result


# ---------------------------------------------------------------------- FurnitureBench rounds
def furniture_round(env: Any, policy: Any, seed: int, label: str = "eval") -> Dict[str, Any]:
    """One round on every env of a manager-based FurnitureBench env, reset under ``seed``: each env's first
    episode (the env resets an env on its own when it ends). Per env: ``success`` / ``out`` (bool) and ``length``
    (sim steps); per phase outcome (the env's ``phase/passed_*``, NaN for an episode that never reached the phase)
    the ``(env, value)`` pairs of the episodes that ended; per residual correction (``residual_mm|deg/<phase>``,
    the phase each agent step started in) the ``(env, value)`` pairs over the agent steps. Keeping the env index
    is what lets :func:`split_furniture` attribute every number to the agent whose env produced it. The progress
    bar shows the envs still in their first episode and the successes so far; the round stops early when every
    env's episode has ended."""
    import torch

    from robonuke_rl_core.envs.build import reseed_env
    from robonuke_rl_core.evaluation import env_states, force_env_reset

    from my_project.residual import PHASE_NAMES

    u = env.unwrapped
    n = int(u.num_envs)
    residual = _residual_or_none(env)
    reseed_env(env, seed)
    observations, states = force_env_reset(env)
    if hasattr(policy, "reset"):
        policy.reset()
    manager = u.termination_manager
    alive = torch.ones(n, dtype=torch.bool, device=u.device)
    success = torch.zeros(n, dtype=torch.bool, device=u.device)
    out = torch.zeros(n, dtype=torch.bool, device=u.device)
    length = torch.zeros(n, dtype=torch.long, device=u.device)
    # one env.step is the residual's steps_per_chunk sim steps
    steps = _agent_steps(int(u.max_episode_length), residual.steps_per_chunk if residual is not None else 1)
    bar = _round_bar(steps, label)
    passed: Dict[str, List[Any]] = {}  # key -> [(env indices, values)] per step
    correction: Dict[str, List[Any]] = {}
    phase = torch.zeros(n, dtype=torch.long, device=u.device)
    for _ in range(steps):
        with torch.no_grad():
            actions = policy(observations, states)
        observations, _, terminated, truncated, infos = env.step(actions)
        if hasattr(policy, "observe"):
            # a routed policy (sac.route_labels) picks each env's residual from the phase the env just published
            policy.observe(infos)
        states = env_states(env)
        if isinstance(infos, dict):
            metrics = infos.get("metrics_to_log") or {}
            for key, unit in (("residual/position_mm", "mm"), ("residual/rot_deg", "deg")):
                value = metrics.get(key)
                if value is None:
                    continue
                value = torch.as_tensor(value, device=u.device).reshape(-1)
                for index, name in enumerate(PHASE_NAMES):
                    rows = alive & (phase == index)
                    if bool(rows.any()):
                        where = rows.nonzero().reshape(-1)
                        correction.setdefault(f"residual_{unit}/{name}", []).append((where.cpu(), value[where].cpu()))
            ending = alive & (terminated.reshape(-1).bool() | truncated.reshape(-1).bool())
            if bool(ending.any()):
                where = ending.nonzero().reshape(-1)
                for key, value in (infos.get("episode_metrics_to_log") or {}).items():
                    if key.startswith("phase/passed_"):
                        values = torch.as_tensor(value, device=u.device).reshape(-1)
                        passed.setdefault(key, []).append((where.cpu(), values[where].cpu()))
            label_ = (infos.get("eval_state") or {}).get("phase/id")
            if label_ is not None:
                phase = torch.as_tensor(label_, device=u.device).reshape(-1).round().long()
        # the episode's sim steps in this agent step (the residual's counted sub-steps; 1 without one)
        length += alive.long() * (residual.counted.to(length.device) if residual is not None else 1)
        # a term counts only on the step it ends the episode: Isaac Lab's term buffer keeps the
        # last ended episode's terms through a reset, so at a round's start it still holds the
        # previous round's successes
        ended = alive & terminated.reshape(-1).bool()
        success |= ended & manager.get_term("success").reshape(-1).bool()
        if "part_out_of_limits" in manager.active_terms:
            out |= ended & manager.get_term("part_out_of_limits").reshape(-1).bool()
        alive &= ~(terminated.reshape(-1).bool() | truncated.reshape(-1).bool())
        bar.set_postfix(running=int(alive.sum()), success=int(success.sum()), refresh=False)
        bar.update(1)
        if not alive.any():
            break
    bar.close()

    def pairs(parts):
        idx = torch.cat([i for i, _ in parts]) if parts else torch.zeros(0, dtype=torch.long)
        val = torch.cat([v for _, v in parts]) if parts else torch.zeros(0)
        return idx, val.float()

    return {
        "success": success.cpu(),
        "out": out.cpu(),
        "length": length.cpu(),
        "passed": {key: pairs(parts) for key, parts in passed.items()},
        "correction": {key: pairs(parts) for key, parts in correction.items()},
    }


def split_furniture(rounds: Sequence[Dict[str, Any]], envs: range, num_rollouts: Optional[int]) -> Dict[str, Any]:
    """One agent's result from :func:`furniture_round` data: its envs ``envs`` (a contiguous block) over the rounds,
    in round order, trimmed to the first ``num_rollouts`` episodes (None: all). The keys the watcher logs:
    ``success_rate``, ``out_of_bounds_rate``, ``length_mean``, ``lengths``, ``episodes``, and when the env tracks
    phases ``phase_success/<phase>`` (each phase's success among the episodes that reached it; ``screw`` is
    ``phase/passed_success``), ``phase_reached/<phase>`` and ``residual_mm|deg/<phase>`` (the mean correction over
    the agent steps that started in that phase)."""
    import torch

    per_round = len(envs)
    wanted = int(num_rollouts) if num_rollouts else per_round * len(rounds)
    success, out, length = [], [], []
    passed: Dict[str, List[torch.Tensor]] = {}
    correction: Dict[str, List[torch.Tensor]] = {}
    taken = 0
    for data in rounds:
        take = min(per_round, wanted - taken)
        if take <= 0:
            break
        lo, hi = envs.start, envs.start + take
        success.append(data["success"][lo:hi])
        out.append(data["out"][lo:hi])
        length.append(data["length"][lo:hi])
        for key, (idx, val) in data["passed"].items():
            keep = (idx >= lo) & (idx < hi)
            passed.setdefault(key, []).append(val[keep])
        for key, (idx, val) in data["correction"].items():
            keep = (idx >= lo) & (idx < hi)
            correction.setdefault(key, []).append(val[keep])
        taken += take
    success_t, out_t, length_t = torch.cat(success), torch.cat(out), torch.cat(length)
    result: Dict[str, Any] = {
        "success_rate": float(success_t.float().mean()),
        "out_of_bounds_rate": float(out_t.float().mean()),
        "length_mean": float(length_t.float().mean()),
        "lengths": length_t.tolist(),
        "episodes": int(length_t.numel()),
    }
    # phase_success/<phase>: each phase's success among the episodes that reached it (screw: phase/passed_success)
    for key, parts in sorted(passed.items()):
        values = torch.cat(parts)
        reached = values[~torch.isnan(values)]
        name = "screw" if key == "phase/passed_success" else key[len("phase/passed_"):]
        if reached.numel():
            result[f"phase_success/{name}"] = float(reached.mean())
            result[f"phase_reached/{name}"] = int(reached.numel())
    # residual_mm/<phase>, residual_deg/<phase>: the mean correction over the agent steps that started in that phase
    for key, parts in sorted(correction.items()):
        values = torch.cat(parts)
        if values.numel():
            result[key] = float(values.mean())
    return result


def evaluate_rounds(env: Any, policy: Any, seed: int, label: str, num_rollouts: Optional[int], num_agents: int = 1) -> List[Dict[str, Any]]:
    """Every agent's evaluation on a FurnitureBench env, side by side: agent ``i`` owns envs ``[i*E, (i+1)*E)``
    (``E = num_envs / num_agents``, the learner's own split), ``ceil(num_rollouts / E)`` rounds (one when
    ``num_rollouts`` is None or fits one round), round ``k`` reset under ``seed + k``. One result dict per agent."""
    n = int(env.unwrapped.num_envs)
    num_agents = int(num_agents)
    if num_agents < 1 or n % num_agents:
        raise ValueError(f"{num_agents} agents cannot share {n} envs: each agent owns a contiguous block")
    per_agent = n // num_agents
    rounds = max(1, -(-int(num_rollouts) // per_agent)) if num_rollouts else 1
    data = [furniture_round(env, policy, seed + k, label if rounds == 1 else f"{label} {k + 1}/{rounds}")
            for k in range(rounds)]
    return [split_furniture(data, range(i * per_agent, (i + 1) * per_agent), num_rollouts) for i in range(num_agents)]


def evaluate_furniture(env: Any, policy: Any, seed: int, label: str = "eval") -> Dict[str, Any]:
    """One round on every env of a FurnitureBench env for one agent: :func:`furniture_round` summarised by
    :func:`split_furniture` (kept for callers of the one-round form)."""
    return split_furniture([furniture_round(env, policy, seed, label)], range(int(env.unwrapped.num_envs)), None)


def evaluate_agents(env: Any, policy: Any, seed: int, label: str, num_rollouts: Optional[int], num_agents: int) -> List[Dict[str, Any]]:
    """One result per agent slot: :func:`evaluate_rounds` on a FurnitureBench env; the single-agent paths of
    :func:`evaluate` otherwise (several agents at once are not implemented for a Forge or standalone env)."""
    if _is_furniture(env):
        return evaluate_rounds(env, policy, seed, label, num_rollouts, num_agents)
    if int(num_agents) != 1:
        raise NotImplementedError(
            f"{num_agents} agents side by side: only FurnitureBench envs split their results per agent; evaluate "
            "a Forge or standalone run one agent directory at a time"
        )
    return [evaluate(env, policy, seed, label, num_rollouts)]


#: list-valued results logged as histograms, under these names
HISTOGRAMS = {"force_max": "force_max_hist", "force_means": "force_mean_hist", "lengths": "length_hist"}


def _summary(result: Dict[str, Any]) -> str:
    if "force_mean" in result:
        return (f"success {result['success_rate']:.3f}, engaged {result['engaged_rate']:.3f}, force mean "
                f"{result['force_mean']:.2f} N, max mean {result['force_max_mean']:.2f} N")
    phases = "".join(f", {name} {result[f'phase_success/{name}']:.2f}" for name in ("corner", "grasp", "insert", "screw")
                     if f"phase_success/{name}" in result)
    return (f"success {result['success_rate']:.3f}, out of bounds {result['out_of_bounds_rate']:.3f}, "
            f"length mean {result['length_mean']:.0f}{phases}")


def new_run_id() -> str:
    """A fresh wandb run id: 8 lowercase alphanumerics, wandb's own form (``wandb.sdk.lib.runid.generate_id``
    where it exists; ``wandb.util.generate_id`` is gone since wandb 0.30)."""
    try:
        from wandb.sdk.lib import runid

        return str(runid.generate_id())
    except (ImportError, AttributeError):
        import secrets
        import string

        alphabet = string.ascii_lowercase + string.digits
        return "".join(secrets.choice(alphabet) for _ in range(8))


class _AgentSink:
    """One agent's outputs: ``<agent dir>/<out_name>/results.jsonl`` (what a restarted watcher resumes from) and its
    wandb eval run (``<agent>_eval``, the training run's project and group), whose id lives beside the results so a
    restarted watcher continues the same run.

    ``results.jsonl`` is the record; the wandb run is a view of it. ``wandb_logged_steps`` beside them names the run
    (first line) and lists the steps it has received, and ``start`` re-sends every recorded step the run lacks -- so
    a fresh run (the id file removed) is filled from the results, and a step evaluated but not delivered is not lost.

    Several sinks live in one process, so every ``wandb.init`` passes ``reinit="create_new"``: wandb's default for a
    second ``init`` is to hand back the first run, which would pour every agent into one run.
    """

    def __init__(self, agent_dir: Path, out_name: str, run_name: str) -> None:
        self.agent_dir = Path(agent_dir)
        self.out_dir = self.agent_dir / out_name
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.results_path = self.out_dir / "results.jsonl"
        self.id_path = self.out_dir / "wandb_run_id"
        self.logged_path = self.out_dir / "wandb_logged_steps"
        self.run_name = run_name
        self.done = {row["step"] for row in self.rows()}
        # the steps delivered to wandb, and the run they went to (the file's first line): a different run lacks
        # them all
        self.logged_run, self.logged = None, set()
        if self.logged_path.exists():
            head, *steps = self.logged_path.read_text().split()
            self.logged_run, self.logged = head, {int(step) for step in steps}
        elif self.id_path.exists():
            # a run resumed from a watcher that kept no such list: it delivered what it recorded
            self.logged_run, self.logged = self.id_path.read_text().strip(), set(self.done)
        self.run = None

    def rows(self) -> List[Dict[str, Any]]:
        """The recorded results, oldest first."""
        if not self.results_path.exists():
            return []
        return [json.loads(line) for line in self.results_path.read_text().splitlines() if line]

    def start(self, cfg: Any, args: Any, num_envs: int, seed: int) -> None:
        import wandb

        def init(run_id: str):
            return wandb.init(
                entity=cfg.wandb.entity,
                project=cfg.wandb.project,
                group=cfg.wandb.group,
                name=self.run_name,
                id=run_id,
                resume="allow",
                reinit="create_new",  # one run per agent in this process, never the first run handed back
                mode=cfg.wandb.mode,
                config={
                    "run": str(self.agent_dir),
                    "eval_config": args.eval_config,
                    "num_envs": int(num_envs),
                    "seed": seed,
                    "deterministic": not args.stochastic,
                },
            )

        run_id = self.id_path.read_text().strip() if self.id_path.exists() else None
        try:
            self.run = init(run_id or new_run_id())
        except wandb.errors.CommError as error:
            if run_id is None or "deleted" not in str(error):
                raise
            # the previous eval run was deleted in wandb: start a new one and evaluate from step 0
            # again, so it holds the whole history
            print(f"[watch] {self.agent_dir.name}: wandb run {run_id} was deleted; starting a new eval run", flush=True)
            if self.results_path.exists():
                self.results_path.rename(self.out_dir / f"results_{run_id}.jsonl")
            self.done = set()
            self.logged = set()
            self.run = init(new_run_id())
        self.id_path.write_text(self.run.id)
        if self.run.id != self.logged_run:
            self.logged_run, self.logged = self.run.id, set()
        self.logged_path.write_text("".join(f"{line}\n" for line in [self.run.id, *sorted(self.logged)]))
        print(f"[watch] {self.agent_dir.name}: wandb eval run {self.run.id} ({self.run.name}), results to "
              f"{self.results_path}", flush=True)
        missing = sorted((row for row in self.rows() if row["step"] not in self.logged), key=lambda row: row["step"])
        for row in missing:
            self._log(row["step"], {k: v for k, v in row.items() if k not in ("step", "label")})
        if missing:
            print(f"[watch] {self.agent_dir.name}: {len(missing)} recorded result(s) the run lacked re-sent, steps "
                  f"{[row['step'] for row in missing]}", flush=True)

    def _log(self, step: int, result: Dict[str, Any]) -> None:
        import wandb

        logged = {f"eval/{k}": v for k, v in result.items() if k != "episodes" and k not in HISTOGRAMS}
        logged.update({f"eval/{HISTOGRAMS[k]}": wandb.Histogram(v) for k, v in result.items() if k in HISTOGRAMS})
        self.run.log(
            logged,
            step=step,
            commit=True,  # publish now; otherwise wandb holds a step until a later one arrives
        )
        with self.logged_path.open("a") as f:
            f.write(f"{step}\n")
        self.logged.add(step)

    def record(self, step: int, label: str, result: Dict[str, Any]) -> None:
        with self.results_path.open("a") as f:
            f.write(json.dumps({"step": step, "label": label, **result}) + "\n")
        self.done.add(step)
        self._log(step, result)
        print(f"[watch] {self.agent_dir.name} {label} (step {step}): {_summary(result)} over {result['episodes']} "
              "episodes", flush=True)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()  # flush and close the eval run on any exit, Ctrl+C included


def check_run_ids(sinks: Sequence[_AgentSink]) -> None:
    """Two agents must not resume the same wandb run. Stale ``wandb_run_id`` files from a watcher that opened one run
    for every agent are refused here, before any run is opened, naming the files to remove."""
    claimed: Dict[str, Path] = {}
    for sink in sinks:
        if not sink.id_path.exists():
            continue
        run_id = sink.id_path.read_text().strip()
        if run_id in claimed:
            raise RuntimeError(
                f"{sink.id_path} and {claimed[run_id]} name the same wandb run {run_id}; every agent needs its own eval "
                "run. Remove the stale wandb_run_id files (keep results.jsonl: a fresh run is filled from it) and "
                "relaunch"
            )
        claimed[run_id] = sink.id_path


def _watch(args, overrides, setup) -> int:
    if setup is not None:
        setup()

    import gymnasium as gym
    from omegaconf import OmegaConf

    from robonuke_rl_core.config import load_from_run
    from robonuke_rl_core.envs.build import build_env, prepare_task, wrap_for_skrl
    from robonuke_rl_core.evaluation import resolved_config_path

    run_dir = Path(args.run).expanduser()
    config_path = resolved_config_path(run_dir)
    # the run's agents, read before the config loads: the eval learner gets one slot per agent
    run_names = list(OmegaConf.load(config_path).derived.run_names)
    agents = agent_dirs(run_dir, run_names)
    cfg = load_from_run(
        config_path, list(overrides) + [f"experiment.num_agents={len(agents)}"], extra_files=[args.eval_config]
    )
    if not cfg.residual.enabled:
        raise ValueError(f"{run_dir} is not a residual run (residual.enabled is false)")
    from my_project.cfg import Q_LEARNERS, q_section

    routes = list(getattr(q_section(cfg), "route_labels", None) or []) if cfg.trainer.learner in Q_LEARNERS else []
    ckpt_dirs = checkpoint_dirs(agents, routes)
    prepare_task(cfg, cfg.task_name, cfg.task_cfg)
    env = wrap_for_skrl(build_env(cfg, gym.make(cfg.task_name, cfg=cfg.task_cfg), cfg.task_name), cfg.experiment.seed)
    _find_residual(env)  # fails loudly if the run's env has no residual
    seed = int(cfg.experiment.seed)
    if len(agents) > 1 and not _is_furniture(env):
        raise NotImplementedError(
            f"{len(agents)} agents side by side: only FurnitureBench envs split their results per agent; pass one "
            "agent directory as --run for this env"
        )

    out_name = args.out_name or ("eval_watch_stochastic" if args.stochastic else "eval_watch")
    suffix = f"_{args.out_name}" if args.out_name else ("_eval_stochastic" if args.stochastic else "_eval")
    sinks = [_AgentSink(agent, out_name, f"{agent.name}{suffix}") for agent in agents]
    check_run_ids(sinks)
    for sink in sinks:
        sink.start(cfg, args, int(env.num_envs), seed)
    print(f"[watch] {run_dir}: {len(agents)} agent(s) x {int(env.num_envs) // len(agents)} envs, "
          f"{int(cfg.eval.num_rollouts)} episodes per agent, seed {seed}, actions "
          f"{'sampled' if args.stochastic else 'deterministic (mean)'}", flush=True)
    try:
        return _evaluate_all(env, cfg, ckpt_dirs, sinks, seed, args)
    finally:
        for sink in sinks:
            sink.finish()


def _evaluate_all(env, cfg, ckpt_dirs: Sequence[Path], sinks: Sequence[_AgentSink], seed: int, args) -> int:
    import torch

    from robonuke_rl_core.evaluation import build_eval_policy

    total = int(cfg.trainer.total_timesteps)
    num_agents = len(sinks)
    num_rollouts = int(cfg.eval.num_rollouts)

    def record_all(step: int, label: str, results: Sequence[Dict[str, Any]]) -> None:
        for sink, result in zip(sinks, results):
            if step not in sink.done:
                sink.record(step, label, result)

    if any(0 not in sink.done for sink in sinks) and not getattr(args, "skip_base", False):
        # the base alone: a residual of exactly zero, no checkpoint needed; the same policy for every agent
        width = int(env.action_space.shape[-1])
        zero = lambda observations, states: torch.zeros((observations.shape[0], width), device=observations.device)
        record_all(0, "base only", evaluate_agents(env, zero, seed, "base only", num_rollouts, num_agents))

    policy = learner = None
    waiting = False
    if args.steps is not None:
        on_disk = set.intersection(*(set(_numbered_checkpoints(d)) for d in ckpt_dirs))
        missing = [s for s in args.steps if s not in on_disk]
        if missing:
            raise FileNotFoundError(
                f"--steps {missing}: no settled checkpoint of every agent in {[str(d) for d in ckpt_dirs]} "
                f"(have {sorted(on_disk)})"
            )
    while True:
        pending = pending_steps(ckpt_dirs, [sink.done for sink in sinks])
        if args.steps is not None:
            pending = [s for s in pending if s in args.steps]
            if not pending:
                print("[watch] --steps: every requested checkpoint evaluated", flush=True)
                return 0
        if not pending and args.once:
            print("[watch] --once: every checkpoint on disk evaluated", flush=True)
            return 0
        if not pending:
            if not waiting:
                print(f"[watch] waiting for new checkpoints in {', '.join(str(d) for d in ckpt_dirs)}", flush=True)
                waiting = True
            time.sleep(args.poll_seconds)
            continue
        waiting = False
        for step in pending:
            name = f"ckpt_{step}.ckpt"
            paths = [Path(d) / name for d in ckpt_dirs]
            if policy is None:
                policy, learner, _ = build_eval_policy(
                    cfg, env, paths[0], deterministic=not args.stochastic, num_agents=num_agents
                )
            for slot, path in enumerate(paths):
                learner.load_agent(path, slot=slot, with_optimizer=False)
            learner.enable_training_mode(False)
            record_all(step, name, evaluate_agents(env, policy, seed, name, num_rollouts, num_agents))
            if step >= total - 1:
                print("[watch] reached the run's last checkpoint", flush=True)
                return 0
        time.sleep(args.poll_seconds)


# ---------------------------------------------------------------------- sampled-eval publishing
PUBLISHED_KEYS = ("success_rate", "engaged_rate", "force_mean", "force_max_mean")


def publish_stochastic(run_dir: str | Path) -> int:
    """Add ``<run>/eval_watch_stochastic/results.jsonl`` to the run's deterministic eval wandb
    run (``<run>/eval_watch/wandb_run_id``) as ``eval_stochastic/*`` against its own x-axis
    ``eval_stochastic/step``: wandb refuses points at a step below one the run already
    logged, and the deterministic eval has logged past every checkpoint. Re-runnable: the
    steps already published are kept in the run's summary and skipped. Returns how many
    results were added."""
    import wandb
    from omegaconf import OmegaConf

    run_dir = Path(run_dir).expanduser()
    results_path = run_dir / "eval_watch_stochastic" / "results.jsonl"
    id_path = run_dir / "eval_watch" / "wandb_run_id"
    if not results_path.is_file():
        raise FileNotFoundError(f"no sampled-action results at {results_path}")
    if not id_path.is_file():
        raise FileNotFoundError(f"no deterministic eval run id at {id_path}")
    results = [json.loads(line) for line in results_path.read_text().splitlines() if line]
    resolved = OmegaConf.load(run_dir.parent / "resolved_config.yaml")

    run = wandb.init(
        entity=resolved.wandb.entity,
        project=resolved.wandb.project,
        id=id_path.read_text().strip(),
        resume="must",
    )
    try:
        run.define_metric("eval_stochastic/step")
        run.define_metric("eval_stochastic/*", step_metric="eval_stochastic/step")
        done = set(run.summary.get("eval_stochastic/published_steps", []) or [])
        fresh = sorted((r for r in results if r["step"] not in done), key=lambda r: r["step"])
        for r in fresh:
            run.log({"eval_stochastic/step": r["step"], **{f"eval_stochastic/{k}": r[k] for k in PUBLISHED_KEYS}})
        done |= {r["step"] for r in fresh}
        run.summary["eval_stochastic/published_steps"] = sorted(done)
        print(f"[publish] {run.name} ({run.id}): {len(fresh)} new result(s), steps {[r['step'] for r in fresh]}", flush=True)
    finally:
        run.finish()
    return len(fresh)


# ---------------------------------------------------------------------- scripted residual probes
#: hand-made residual policies, in units of the residual bound ([-1, 1] per dim, [x, y, z, yaw]):
#: does a crude bias or dither reproduce what the learned residual gains?
PROBES = {
    "z_offset": "constant z = -0.77, nothing else",
    "z_offset_xy_kicks": "z = -0.77 plus x/y of +-0.7, random sign each step",
    "z_offset_xy_noise": "z = -0.77 plus x/y ~ N(0, 0.25) clipped to [-1, 1], each step",
}


def _probe_policy(name: str):
    import torch

    def policy(observations, states):
        n, device = observations.shape[0], observations.device
        r = torch.zeros((n, 4), device=device)
        r[:, 2] = -0.77
        if name == "z_offset_xy_kicks":
            r[:, 0:2] = 0.7 * (torch.randint(0, 2, (n, 2), device=device) * 2 - 1)
        elif name == "z_offset_xy_noise":
            r[:, 0:2] = (0.25 * torch.randn((n, 2), device=device)).clamp(-1.0, 1.0)
        return r

    return policy


def probe_main(argv=None, setup=None) -> int:
    """Evaluate the scripted residual probes on a residual run's eval env (the watcher's: same
    config layering, 100 envs, same seeded starts), print the results and append them to
    ``<run>/eval_probe/results.jsonl``."""
    parser = argparse.ArgumentParser(description=probe_main.__doc__)
    parser.add_argument("--run", required=True, help="a residual run's agent directory")
    parser.add_argument("--eval_config", required=True)
    parser.add_argument("--probes", default=",".join(PROBES), help=f"comma list of {list(PROBES)}")

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args, overrides = parser.parse_known_args(argv)
    args.headless = True
    AppLauncher(args)
    try:
        if setup is not None:
            setup()
        import gymnasium as gym

        from robonuke_rl_core.config import load_from_run
        from robonuke_rl_core.envs.build import build_env, prepare_task, wrap_for_skrl
        from robonuke_rl_core.evaluation import SINGLE_AGENT_OVERRIDE, resolved_config_path

        run_dir = Path(args.run).expanduser()
        cfg = load_from_run(
            resolved_config_path(run_dir), list(overrides) + [SINGLE_AGENT_OVERRIDE], extra_files=[args.eval_config]
        )
        prepare_task(cfg, cfg.task_name, cfg.task_cfg)
        env = wrap_for_skrl(build_env(cfg, gym.make(cfg.task_name, cfg=cfg.task_cfg), cfg.task_name), cfg.experiment.seed)
        _find_residual(env)
        seed = int(cfg.experiment.seed)
        out = run_dir / "eval_probe"
        out.mkdir(parents=True, exist_ok=True)
        for name in [p for p in args.probes.split(",") if p]:
            if name not in PROBES:
                raise ValueError(f"unknown probe {name!r}; known: {list(PROBES)}")
            result = evaluate(env, _probe_policy(name), seed)
            with (out / "results.jsonl").open("a") as f:
                f.write(json.dumps({"probe": name, "description": PROBES[name], **result}) + "\n")
            print(
                f"[probe] {name} ({PROBES[name]}): success {result['success_rate']:.2f}, engaged "
                f"{result['engaged_rate']:.2f}, force mean {result['force_mean']:.2f} N, max mean "
                f"{result['force_max_mean']:.2f} N over {result['episodes']} episodes",
                flush=True,
            )
        sys.stdout.flush()
        os._exit(0)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
