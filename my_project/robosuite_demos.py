"""A robosuite task's recorded demos (``scripts/import_robomimic.py``'s dataset) as one whole-task residual's demo
transitions (``ddpg.demo_transitions``), offline, no sim stepped (the sim app starts only to load the experiment's full
config, for the file's ``meta["fingerprint"]``).

    python scripts/make_robosuite_demos.py --config configs/experiments/resfit_can/residual_resfit_can.yaml

For the residual ``action_space: per_dim`` with ``critic_state: observation`` and ``critic_action: full`` (ResFiT's
form: the executed action is the base's plus ``scale * r``, the critics see the observation and the executed action):
every demo frame ``t`` from 0 up to its ``success_frame`` (the step that put the can in its bin) becomes one transition:

* the base action at ``t`` as the env wrapper's :class:`~my_project.models.diffusion_policy.ChunkedBase` produces it
  along the demo (a fresh episode at frame 0, a re-plan every ``action_horizon`` frames, from the demo's observations);
* ``r_t = (a_demo_t - a_base_t) / scale`` (``scale``: :func:`~my_project.residual.residual_scale_and_clip`), never
  clipped (the critics take it as it is);
* the agent's observation, the critics' observation and the side input as the wrapper builds them, raw
  (:func:`~my_project.residual.residual_observation`; the side input the base action);
* reward 1 and terminated at ``success_frame`` (the env's sparse reward and its end on success), 0 before.

ResFiT builds its offline buffer the same way (the demo's action, the base action relabelled by running the base along
the demo), with two differences kept out here: its base queue is never reset between demos, and it labels the last
5-6 frames of every demo done / rewarded (robomimic's ``done_mode 2``) instead of the first successful one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

__all__ = ["demo_transitions", "main"]


def demo_transitions(policy, frames, scale: torch.Tensor, rc: Any) -> Dict[str, torch.Tensor]:
    """The transitions of ``frames`` (one ``{"obs": (T, O), "action": (T, A), "success_frame": int}`` per demo, raw),
    all demos run through one :class:`ChunkedBase` in lockstep (a finished demo repeats its last frame, unread)."""
    from my_project.models.diffusion_policy import ChunkedBase
    from my_project.residual import residual_observation

    device = policy.device
    n = len(frames)
    lengths = [int(f["success_frame"]) + 1 for f in frames]
    base = ChunkedBase(policy, n)
    base_actions = [torch.zeros((m, policy.action_dim)) for m in lengths]
    for t in range(max(lengths)):
        obs = torch.stack([f["obs"][min(t, m - 1)] for f, m in zip(frames, lengths)]).to(device)
        fresh = torch.full((n,), t == 0, dtype=torch.bool, device=device)
        step = base.actions(obs, fresh).cpu()
        for i, m in enumerate(lengths):
            if t < m:
                base_actions[i][t] = step[i]
    rows: Dict[str, list] = {}
    for f, m, a_base in zip(frames, lengths, base_actions):
        # built on the base's device, stored on the CPU; raw (the learner's state_preprocessor normalizes)
        obs, demo, a_base = f["obs"][:m].to(device), f["action"][:m].to(device), a_base.to(device)
        agent, _ = residual_observation(obs, a_base, bool(rc.observe_base_action))
        critic = obs
        side = a_base
        nxt = torch.clamp(torch.arange(m) + 1, max=m - 1)  # the last row ends the episode: its next is never read
        done = torch.zeros((m, 1), dtype=torch.bool)
        done[-1] = True
        part = {
            "observations": agent, "next_observations": agent[nxt], "states": critic, "next_states": critic[nxt],
            "actions": (demo - a_base) / scale.to(device), "rewards": done.to(torch.float32), "terminated": done,
            "truncated": torch.zeros((m, 1), dtype=torch.bool), "success": done.to(torch.float32),
            "residual_base": side, "next_residual_base": side[nxt],
        }
        for name, value in part.items():
            rows.setdefault(name, []).append(value.cpu())
    return {name: torch.cat(values) for name, values in rows.items()}


def main(argv=None, setup=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="the residual experiment (its ddpg.demo_transitions is written)")
    parser.add_argument("--dataset", default=None, help="the demo dataset (default: the base run's bc.dataset)")
    parser.add_argument("--out", default=None, help="output file (default: the learner section's demo_transitions)")

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    args.headless = True
    from robonuke_rl_core.config import simulator_of, start_simulator

    # Isaac Sim starts only for a task it simulates (task.simulator); a robosuite task's is standalone
    app_launcher = start_simulator(args, simulator_of(argparse.Namespace(config=args.config, from_run=None)))
    try:
        if setup is not None:
            setup()
        code = _build(args)
        sys.stdout.flush()
        if app_launcher is not None:
            app_launcher.app.close()
        return code
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


def _build(args) -> int:
    import pandas as pd
    from omegaconf import OmegaConf

    from robonuke_rl_core.config import load_config
    from robonuke_rl_core.evaluation import resolved_config_path

    from my_project.cfg import q_section
    from my_project.families import ROBOSUITE, family_of
    from my_project.models.diffusion_policy import MinMaxNormalizer
    from my_project.residual import base_run_dir, base_source, load_base_policy, override_base, residual_scale_and_clip
    from my_project.residual_demos import CRITIC_STATE_FORMAT, FURNITURE_FORMAT, demo_fingerprint

    cfg = load_config(args.config, [])
    rc, section = cfg.residual, q_section(cfg)
    if family_of(cfg.task_name) != ROBOSUITE:
        raise ValueError(f"{args.config}: task {cfg.task_name!r} is not a robosuite task")
    if not (rc.enabled and rc.action_space == "per_dim" and rc.critic_state == "observation" and rc.critic_action == "full"
            and int(rc.steps_per_chunk) == 1 and not rc.observation_drop):
        raise ValueError("these demos are for residual.action_space: per_dim, critic_state: observation, critic_action: full, "
                         "steps_per_chunk 1, no observation_drop")
    if getattr(section, "route_labels", None):
        raise ValueError("these demos are for one residual over the whole task (no route_labels)")
    out = Path(args.out or section.demo_transitions)
    base_dir = base_run_dir(cfg)
    dataset = Path(args.dataset or OmegaConf.load(resolved_config_path(base_dir)).bc.dataset).expanduser()
    episodes = [json.loads(line) for line in (dataset / "meta" / "episodes.jsonl").read_text().splitlines() if line]

    torch.manual_seed(int(cfg.experiment.seed))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy = override_base(load_base_policy(base_dir, rc.base_checkpoint, device), rc, base_source(cfg))
    frames = []
    for episode in episodes:
        table = pd.read_parquet(dataset / episode["file"])
        frames.append({
            "obs": torch.tensor(np.stack(table["observation.state"].values), dtype=torch.float32),
            "action": torch.tensor(np.stack(table["action"].values), dtype=torch.float32),
            "success_frame": int(episode["success_frame"]),
        })
    # the residual's scale as the env wrapper builds it (residual_scale_and_clip); the inputs raw
    scale, _ = residual_scale_and_clip(rc, policy, base_dir)
    rows = demo_transitions(policy, frames, scale, rc)
    meta = {
        "fingerprint": demo_fingerprint(cfg), "furniture_format": FURNITURE_FORMAT, "critic_state_format": CRITIC_STATE_FORMAT,
        "agent_obs_narrow_range": MinMaxNormalizer.NARROW_RANGE, "form": "observation", "route": "*",
        "dataset": str(dataset), "base": f"{base_source(cfg)}/{rc.base_checkpoint}",
        "base_step": int(getattr(policy, "checkpoint_step", -1)), "episodes": len(frames), "scale": scale.tolist(),
        "base_warm_start": bool(policy.cfg.warm_start),
        "inputs": "raw", "action_half_range_floor": rc.action_half_range_floor,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"transitions": rows, "meta": meta}, out)
    reach = rows["actions"].abs().amax(dim=-1)
    print(f"[robosuite demos] {rows['actions'].shape[0]} transitions from {len(frames)} demos, reward sum "
          f"{float(rows['rewards'].sum()):.0f}, |r| per dim mean {rows['actions'].abs().mean(0).tolist()}, "
          f"|r| max per row median {float(reach.median()):.3f}, beyond 1: {float((reach > 1).float().mean()):.1%} -> {out}")
    return 0
