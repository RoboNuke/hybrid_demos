"""Evaluate a residual run's checkpoints as they appear, against its base policy alone.

    python scripts/watch_residual_eval.py --run <residual run dir> --eval_config <eval config>

Runs beside training, in its own Isaac process. The env is the run's own (its resolved config)
with ``--eval_config`` on top (required: the eval conditions, e.g. the env count), built through the same project wrappers training uses. Then:

1. **Step 0, the base alone**, right at startup: a residual of exactly zero, so the frozen
   diffusion policy acts by itself (no checkpoint needed).
2. **Every numbered checkpoint** (``ckpt_<step>.ckpt``) not yet evaluated, oldest first, with
   the residual on, the SAC policy deterministic (its mean action) or, with ``--stochastic``,
   sampled as in training; new ones are picked up as training writes them. It stops after the
   checkpoint at or past the run's ``trainer.total_timesteps``, with ``--once`` when no
   checkpoint on disk is left, or on Ctrl+C.

The two action modes measure different policies: training only ever values sampled actions,
so the mean alone can be much worse (measured on these runs: 15–40 points lower).

Every evaluation is one round: all envs reset together under the same seed (the same 100
starting conditions every time) and step for the episode's full length; each env contributes
exactly one episode. Per episode, from the true (noise-free) sensor force:

* success: the env's success check held at any step; engaged: its engage check did;
* the mean and the maximum contact-force magnitude over the episode (N).

On a FurnitureBench run (``Isaac-Furniture-*``, a manager-based env that ends an episode on
assembly or a part out of bounds) each env's FIRST episode of the round counts: success is the
env's ``success`` termination firing in it, plus the share that ended out of bounds and the
mean episode length; there is no force or engagement.

Logged to its own wandb run (``<run name>_eval``, the training run's project and group), at
``step`` = the checkpoint's step: ``eval/success_rate``, ``eval/engaged_rate``,
``eval/force_mean`` (mean over episodes of each episode's mean force), ``eval/force_max_mean``,
and histograms ``eval/force_max_hist`` and ``eval/force_mean_hist``. Each result is also
appended to ``<run dir>/eval_watch/results.jsonl``. Each round shows its progress as a tqdm bar
on stderr.
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
from typing import Any, Dict, List



def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--run", required=True, help="the residual run's agent directory (holds checkpoints/)")
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
    app_launcher = AppLauncher(args)
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


def _find_residual(env: Any):
    from my_project.residual import ResidualPolicyWrapper

    node, seen = env, set()
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        if isinstance(node, ResidualPolicyWrapper):
            return node
        node = getattr(node, "_env", None) or getattr(node, "env", None)
    raise RuntimeError("the env has no ResidualPolicyWrapper: is residual.enabled set in the run's config?")


def _round_bar(total: int, label: str):
    """The round's progress on stderr: env steps out of the episode's full length."""
    from tqdm import tqdm

    return tqdm(total=total, desc=f"[watch] {label}", unit="step", dynamic_ncols=True, leave=False)


def evaluate(env: Any, policy: Any, seed: int, label: str = "eval") -> Dict[str, Any]:
    """One round on every env: success, engagement, mean and max force per episode
    (FurnitureBench: :func:`evaluate_furniture`). ``label`` names the round's progress bar."""
    import torch

    if hasattr(env.unwrapped, "termination_manager"):
        return evaluate_furniture(env, policy, seed, label)

    from robonuke_rl_core.evaluation import env_states, force_env_reset

    from robonuke_rl_core.envs.build import reseed_env

    u = env.unwrapped
    n = int(u.num_envs)
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
    steps = int(u.max_episode_length) - 1  # Forge ends an episode one step before the limit
    bar = _round_bar(steps, label)
    for _ in range(steps):
        with torch.no_grad():
            actions = policy(observations, states)
        observations, _, terminated, truncated, _ = env.step(actions)
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


def evaluate_furniture(env: Any, policy: Any, seed: int, label: str = "eval") -> Dict[str, Any]:
    """One round on every env of a manager-based FurnitureBench env: each env's first episode
    (the env resets an env on its own when it ends): success, out of bounds, length. The progress
    bar shows the envs still in their first episode and the successes so far; it stops early when
    every env's episode has ended."""
    import torch

    from robonuke_rl_core.envs.build import reseed_env
    from robonuke_rl_core.evaluation import env_states, force_env_reset

    u = env.unwrapped
    n = int(u.num_envs)
    reseed_env(env, seed)
    observations, states = force_env_reset(env)
    if hasattr(policy, "reset"):
        policy.reset()
    manager = u.termination_manager
    alive = torch.ones(n, dtype=torch.bool, device=u.device)
    success = torch.zeros(n, dtype=torch.bool, device=u.device)
    out = torch.zeros(n, dtype=torch.bool, device=u.device)
    length = torch.zeros(n, dtype=torch.long, device=u.device)
    bar = _round_bar(int(u.max_episode_length), label)
    for _ in range(int(u.max_episode_length)):
        with torch.no_grad():
            actions = policy(observations, states)
        observations, _, terminated, truncated, _ = env.step(actions)
        states = env_states(env)
        length += alive.long()
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
    return {
        "success_rate": float(success.float().mean()),
        "out_of_bounds_rate": float(out.float().mean()),
        "length_mean": float(length.float().mean()),
        "lengths": length.cpu().tolist(),
        "episodes": n,
    }


#: list-valued results logged as histograms, under these names
HISTOGRAMS = {"force_max": "force_max_hist", "force_means": "force_mean_hist", "lengths": "length_hist"}


def _summary(result: Dict[str, Any]) -> str:
    if "force_mean" in result:
        return (f"success {result['success_rate']:.3f}, engaged {result['engaged_rate']:.3f}, force mean "
                f"{result['force_mean']:.2f} N, max mean {result['force_max_mean']:.2f} N")
    return (f"success {result['success_rate']:.3f}, out of bounds {result['out_of_bounds_rate']:.3f}, "
            f"length mean {result['length_mean']:.0f}")


def _watch(args, overrides, setup) -> int:
    if setup is not None:
        setup()

    import gymnasium as gym
    import torch
    
    from robonuke_rl_core.config import load_from_run
    from robonuke_rl_core.envs.build import build_env, prepare_task, wrap_for_skrl
    from robonuke_rl_core.evaluation import SINGLE_AGENT_OVERRIDE, build_eval_policy, resolved_config_path

    run_dir = Path(args.run).expanduser()
    ckpt_dir = run_dir / "checkpoints"
    cfg = load_from_run(
        resolved_config_path(run_dir), list(overrides) + [SINGLE_AGENT_OVERRIDE], extra_files=[args.eval_config]
    )
    if not cfg.residual.enabled:
        raise ValueError(f"{run_dir} is not a residual run (residual.enabled is false)")
    prepare_task(cfg, cfg.task_name, cfg.task_cfg)
    env = wrap_for_skrl(build_env(cfg, gym.make(cfg.task_name, cfg=cfg.task_cfg), cfg.task_name), cfg.experiment.seed)
    _find_residual(env)  # fails loudly if the run's env has no residual
    total = int(cfg.trainer.total_timesteps)
    seed = int(cfg.experiment.seed)

    out_name = args.out_name or ("eval_watch_stochastic" if args.stochastic else "eval_watch")
    out_dir = run_dir / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    done = set()
    if results_path.exists():
        done = {json.loads(line)["step"] for line in results_path.read_text().splitlines() if line}

    import wandb

    id_path = out_dir / "wandb_run_id"

    def start(run_id: str):
        return wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            group=cfg.wandb.group,
            name=(f"{run_dir.name}_{args.out_name}" if args.out_name
                  else f"{run_dir.name}_eval" + ("_stochastic" if args.stochastic else "")),
            id=run_id,
            resume="allow",
            mode=cfg.wandb.mode,
            config={
                "run": str(run_dir),
                "eval_config": args.eval_config,
                "num_envs": int(env.num_envs),
                "seed": seed,
                "deterministic": not args.stochastic,
            },
        )

    # the run's id lives beside the results, so a restarted watcher continues the same run
    run_id = id_path.read_text().strip() if id_path.exists() else None
    try:
        run = start(run_id or wandb.util.generate_id())
    except wandb.errors.CommError as error:
        if run_id is None or "deleted" not in str(error):
            raise
        # the previous eval run was deleted in wandb: start a new one and evaluate from step 0
        # again, so it holds the whole history
        print(f"[watch] wandb run {run_id} was deleted; starting a new eval run", flush=True)
        if results_path.exists():
            results_path.rename(out_dir / f"results_{run_id}.jsonl")
        done = set()
        run = start(wandb.util.generate_id())
    id_path.write_text(run.id)
    print(f"[watch] wandb eval run {run.id} ({run.name}); actions "
          f"{'sampled' if args.stochastic else 'deterministic (mean)'}", flush=True)

    def record(step: int, label: str, result: Dict[str, Any]) -> None:
        logged = {f"eval/{k}": v for k, v in result.items() if k != "episodes" and k not in HISTOGRAMS}
        logged.update({f"eval/{HISTOGRAMS[k]}": wandb.Histogram(v) for k, v in result.items() if k in HISTOGRAMS})
        run.log(
            logged,
            step=step,
            commit=True,  # publish now; otherwise wandb holds a step until a later one arrives
        )
        with results_path.open("a") as f:
            f.write(json.dumps({"step": step, "label": label, **result}) + "\n")
        print(f"[watch] {label} (step {step}): {_summary(result)} over {result['episodes']} episodes", flush=True)

    print(f"[watch] {run_dir}: {int(env.num_envs)} envs, seed {seed}, results to {results_path}", flush=True)
    try:
        return _evaluate_all(env, cfg, ckpt_dir, done, record, seed, args, run)
    finally:
        run.finish()  # flush and close the eval run on any exit, Ctrl+C included


def _evaluate_all(env, cfg, ckpt_dir, done, record, seed, args, run) -> int:
    import torch

    from robonuke_rl_core.evaluation import build_eval_policy

    total = int(cfg.trainer.total_timesteps)
    if 0 not in done:
        # the base alone: a residual of exactly zero, no checkpoint needed
        width = int(env.action_space.shape[-1])
        zero = lambda observations, states: torch.zeros((observations.shape[0], width), device=observations.device)
        record(0, "base only", evaluate(env, zero, seed, "base only"))
        done.add(0)

    policy = learner = None
    waiting = False
    if args.steps is not None:
        on_disk = _numbered_checkpoints(ckpt_dir)
        missing = [s for s in args.steps if s not in on_disk]
        if missing:
            raise FileNotFoundError(f"--steps {missing}: no settled checkpoint in {ckpt_dir} (have {sorted(on_disk)})")
    while True:
        pending = sorted((s, p) for s, p in _numbered_checkpoints(ckpt_dir).items() if s not in done)
        if args.steps is not None:
            pending = [(s, p) for s, p in pending if s in args.steps]
            if not pending:
                print("[watch] --steps: every requested checkpoint evaluated", flush=True)
                return 0
        if not pending and args.once:
            print("[watch] --once: every checkpoint on disk evaluated", flush=True)
            return 0
        if not pending:
            if not waiting:
                print(f"[watch] waiting for new checkpoints in {ckpt_dir}", flush=True)
                waiting = True
            time.sleep(args.poll_seconds)
            continue
        waiting = False
        if policy is None:
            policy, learner, _ = build_eval_policy(cfg, env, pending[0][1], deterministic=not args.stochastic)
        for step, path in pending:
            learner.load_agent(path, slot=0, with_optimizer=False)
            learner.enable_training_mode(False)
            record(step, path.name, evaluate(env, policy, seed, path.name))
            done.add(step)
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
