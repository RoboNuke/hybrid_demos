"""Per-phase demo transitions for a routed residual (``sac.route_labels``), offline, CPU only, no sim.

    python scripts/make_phase_demos.py --config configs/experiments/parts_pd/residual_parts_pd.yaml

For each recorded demo episode (the base run's ``bc.dataset``: LeRobot parquet, ``observation.state_abs`` and
``action.abs``), and each chunk of ``residual.steps_per_chunk`` frames from frame 0 up to the frame whose step assembles
the leg (:func:`assembling_frame`), with the actions the env executed (:func:`executed_abs_actions`):

1. the task phase of every frame, from its observation (Robonuke_Furniture_Bench
   ``assembly/phases.phases_from_states``: the env's phase tracker's checks, offline);
2. the frozen base's ``k`` actions at the chunk's first frame, as the env runs it: a plan (its own observation
   history, its DDIM steps) at frames ``0, H, 2H, ...`` (``H`` its executed ``action_horizon``), each frame taking the
   plan's next action (``k`` divides ``H``, so a chunk never straddles a re-plan);
3. the residual that turns that base chunk into the demo's next ``k`` actions (``action_frame: target``'s map,
   inverted: ``R_base^T (p_demo - p_base) / pos_scale``, ``log(R_base^T R_demo) / rot_scale``, the gripper difference
   over its scale); it may lie beyond the actor's [-1, 1] (the critics take it as it is; the actor clones it clipped);
4. the agent's observation (the raw observation, then the base chunk in the EE-relative form,
   ``residual.ee_relative_actions``), the critic state (the raw observation: ``residual.critic_state: observation``)
   and the critic action transform's side input (the normalized base chunk, then the EE pose), at the chunk's first
   frame and at the next chunk's.

A chunk goes to the residual of the phase its first frame is in (``sac.route_labels``; chunks of other phases are
dropped), rewarded by that residual's local outcome: 1 when the next chunk's first frame is in a later
``sac.std_label_names`` entry, or the demo succeeds within the chunk (``success``); the attempt then ends (terminated),
as it does at the demo's end. With ``sac.potential_terms`` for the route, the reward gains the learner's shaping
``gamma Phi(s') - Phi(s)`` (``Phi = -sum weight * value``, the channels measured offline from the frame's observation:
``assembly/phases.leg_measures_from_states``), ``s`` the chunk's first frame and ``s'`` the frame after its last;
each row's ``success`` is its local outcome alone. One file per route: ``sac.demo_transitions`` with ``{route}`` filled in.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import torch

__all__ = ["invert_target", "episode_chunks", "assembling_frame", "executed_abs_actions", "main"]


def assembling_frame(rewards) -> int:
    """The frame whose step assembles the leg. The scripted collector rewards (1) the first frame whose observation
    is already assembled -- the expert's no-op there holds the pose and opens the gripper, and lifting clear
    follows -- so the step that assembles is the frame before it: it is the demo's success, and nothing after it is
    used."""
    import numpy as np

    hits = np.nonzero(np.asarray(rewards) > 0)[0]
    if hits.size == 0:
        raise ValueError("the episode has no reward: no assembled frame")
    if int(hits[0]) < 1:
        raise ValueError("the episode is rewarded on its first frame: no step assembled the leg")
    return int(hits[0]) - 1


def executed_abs_actions(source: Path, states: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """The ``action.abs`` actions the env was given, ``(T, 10)``, from the episode's source pkl. A DART collection
    sent perturbed commands during its bursts (``actions_executed``) while the dataset keeps the expert's own
    (the BC labels); the next observations come from the executed ones, so a transition must carry those. Converted
    with the dataset import's own ``fb_abs_frames``; the pkl's labels and states must be the dataset's (``labels``,
    ``states``), so an episode that does not line up is refused. A pkl with no ``actions_executed`` gives its
    labels."""
    import numpy as np

    from furniture_lab.demos import fb_abs_frames

    from my_project.demos.furniture import load_pkl

    data = load_pkl(Path(source))
    obstacle = states.shape[1] == 63  # the ResiP layout (robot-base parts, the front obstacle)
    T = labels.shape[0]
    states_pkl, labels_pkl = fb_abs_frames(data, obstacle=obstacle)
    if labels_pkl.shape[0] != T or not np.allclose(labels_pkl, labels.numpy(), atol=1e-5) or not np.allclose(
        states_pkl[:T], states.numpy(), atol=1e-5
    ):
        raise ValueError(f"{source}: its actions / states are not the dataset episode's; rebuild the dataset or check episodes.jsonl")
    _, executed = fb_abs_frames(data, obstacle=obstacle, executed=True)
    return torch.tensor(np.asarray(executed), dtype=torch.float32)


def invert_target(base: torch.Tensor, demo: torch.Tensor, pos_scale: float, rot_scale: float, grip_scale: float,
                  axes: str = "target") -> torch.Tensor:
    """The residual ``r`` ``(..., 7)`` that :class:`~my_project.residual_target.TargetResidual` (with ``axes``,
    ``residual.target_axes``) turns ``base`` into ``demo`` with (both ``(..., 10)`` absolute targets): its exact
    inverse, unclipped."""
    from my_project.models.action_frame import rot6d_to_mat
    from my_project.residual_ee import log_so3

    R_b = rot6d_to_mat(base[..., 3:9])
    R_d = rot6d_to_mat(demo[..., 3:9])
    if axes == "base":
        r_pos = (demo[..., 0:3] - base[..., 0:3]) / pos_scale
        r_rot = log_so3(R_d @ R_b.transpose(-1, -2)) / rot_scale
        r_grip = (demo[..., 9:10] - base[..., 9:10]) / grip_scale
        return torch.cat((r_pos, r_rot, r_grip), dim=-1)
    r_pos = (R_b.transpose(-1, -2) @ (demo[..., 0:3] - base[..., 0:3]).unsqueeze(-1)).squeeze(-1) / pos_scale
    r_rot = log_so3(R_b.transpose(-1, -2) @ R_d) / rot_scale
    r_grip = (demo[..., 9:10] - base[..., 9:10]) / grip_scale
    return torch.cat((r_pos, r_rot, r_grip), dim=-1)


def episode_chunks(
    states: torch.Tensor,
    actions: torch.Tensor,
    phases: torch.Tensor,
    success_frame: int,
    base_chunks: torch.Tensor,
    *,
    k: int,
    names: List[str],
    routes: List[str],
    scales: Any,
    ee_rel: Any,
    normalizer: Any,
    encode: Any = None,
    axes: str = "target",
    potentials: Dict[str, torch.Tensor] = None,
    gamma: float = 1.0,
    residual_gripper: bool = True,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """One demo episode's chunk transitions per route (see the module docstring). ``states`` ``(T, 63)``, ``actions``
    ``(T, 10)``, ``phases`` ``(T,)`` (0..3), ``success_frame`` the frame whose step succeeds (its last used frame),
    ``base_chunks`` ``(S, k, 10)`` the base's chunk at each chunk start ``0, k, 2k, ...`` up to ``success_frame``;
    ``scales`` the target map's ``(pos m, rot rad, gripper)`` per unit; ``potentials`` the shaped routes' ``Phi`` at
    every frame ``(T,)`` (``sac.potential_terms``), ``gamma`` the learner's ``discount_factor``."""
    starts = list(range(0, success_frame + 1, k))
    if base_chunks.shape[0] != len(starts):
        raise ValueError(f"{len(starts)} chunk starts but {base_chunks.shape[0]} base chunks")
    success_label = names.index("success") if "success" in names else len(names)
    route_ids = {names.index(route): route for route in routes}
    last = success_frame

    def at(index: int):
        """Agent observation, critic state and side input (None without one) at chunk ``index`` (its first frame);
        ``encode(state (1, obs_dim), chunk (1, k, 10))`` gives them for another setup than the EE-relative one."""
        t = starts[index]
        chunk = base_chunks[index].unsqueeze(0)
        if encode is not None:
            return encode(states[t : t + 1], chunk)
        obs = torch.cat((states[t : t + 1], ee_rel(chunk, states[t : t + 1, 0:9])), dim=-1)[0]
        side = torch.cat((normalizer.normalize(chunk, "action").reshape(1, -1), states[t : t + 1, 0:9]), dim=-1)[0]
        return obs, states[t], side

    def shaped(route: str, t: int) -> float:
        """The learner's potential shaping for the chunk starting at frame ``t`` (0 for an unshaped route)."""
        phi = (potentials or {}).get(route)
        if phi is None:
            return 0.0
        return float(gamma * phi[min(t + k, phi.shape[0] - 1)] - phi[t])

    out: Dict[str, Dict[str, List[torch.Tensor]]] = {}
    for index, t in enumerate(starts):
        route = route_ids.get(int(phases[t]))
        if route is None:
            continue
        frames = [min(t + j, last) for j in range(k)]
        residual = invert_target(base_chunks[index], actions[frames], *scales, axes=axes)
        if not residual_gripper:
            residual = residual[..., :6]  # residual.residual_gripper: false -- the gripper is the base's
        succeeded_in_chunk = t <= success_frame < t + k
        if succeeded_in_chunk or index + 1 >= len(starts):
            next_index, next_label = index, success_label if succeeded_in_chunk else int(phases[last])
        else:
            next_index, next_label = index + 1, int(phases[starts[index + 1]])
        local = next_label > names.index(route) and next_label <= success_label
        ended = local or index + 1 >= len(starts) or succeeded_in_chunk
        obs, state, side = at(index)
        next_obs, next_state, next_side = at(next_index)
        row = {
            "observations": obs, "next_observations": next_obs, "states": state, "next_states": next_state,
            "actions": residual.reshape(-1), "rewards": torch.tensor([(1.0 if local else 0.0) + shaped(route, t)]),
            "terminated": torch.tensor([bool(ended)]), "truncated": torch.tensor([False]),
            # the local outcome alone (a shaped reward is no success flag)
            "success": torch.tensor([1.0 if local else 0.0]),
        }
        if side is not None:
            row.update({"residual_base": side, "next_residual_base": next_side})
        bucket = out.setdefault(route, {})
        for name, value in row.items():
            bucket.setdefault(name, []).append(value)
    return {route: {name: torch.stack(values) for name, values in rows.items()} for route, rows in out.items()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="the routed residual experiment (its sac.demo_transitions holds {route})")
    parser.add_argument("--dataset", default=None, help="the demo dataset (default: the base run's bc.dataset)")
    parser.add_argument("--out", default=None, help="output path pattern with {route} (default: sac.demo_transitions)")
    args = parser.parse_args(argv)

    import glob

    import numpy as np
    import pandas as pd
    from omegaconf import OmegaConf

    from robonuke_rl_core.config import load_file_chain
    from robonuke_rl_core.evaluation import resolved_config_path

    from furniture_lab.tasks.assembly.phases import leg_measures_from_states, phases_from_states
    from my_project.residual import base_run_dir, load_base_policy, override_base, residual_scale
    from my_project.residual_target import EERelative, TargetResidual

    from robonuke_rl_core.learners.cfg import SACCfg

    from my_project.cfg import ResidualCfg

    # the sections' dataclass defaults under the config's own values (the files set only non-defaults)
    defaults = OmegaConf.create({"residual": OmegaConf.to_container(OmegaConf.structured(ResidualCfg)),
                                 "sac": OmegaConf.to_container(OmegaConf.structured(SACCfg))})
    cfg = OmegaConf.merge(defaults, *[layer for _, layer in load_file_chain(Path(args.config))])
    rc, sac = cfg.residual, cfg.sac
    routes, names = list(sac.route_labels), list(sac.std_label_names)
    if not routes:
        raise ValueError(f"{args.config} sets no sac.route_labels: these demos are per routed phase")
    ee_form = rc.ee_relative_actions and rc.critic_state == "observation"
    # the residual's critic state; the critics take r (Q(s, a_base, r)) or the executed targets rebuilt from the
    # base action in their state (critic_action: full): the rows are the same either way
    absolute_form = not rc.ee_relative_actions and rc.critic_state == "residual" and rc.critic_action in ("residual", "full")
    if rc.action_frame != "target" or not (ee_form or absolute_form):
        raise ValueError(
            "the per-phase demos are built for action_frame: target with either ee_relative_actions and critic_state: "
            "observation (the observation's critic, the base chunk a side input) or no ee_relative_actions with "
            "critic_state: residual and critic_action: residual (Q(s, a_base, r)) or full (the executed targets)"
        )
    if ee_form and rc.observation_drop:
        raise NotImplementedError("residual.observation_drop with the EE-relative demo form is not built here")
    out = args.out or sac.demo_transitions
    if not out or "{route}" not in str(out):
        raise ValueError("give an output pattern with {route} (--out or sac.demo_transitions)")
    base_dir = base_run_dir(cfg)
    dataset = Path(args.dataset).expanduser() if args.dataset else Path(OmegaConf.load(resolved_config_path(base_dir)).bc.dataset).expanduser()
    policy = load_base_policy(base_dir, rc.base_checkpoint, "cpu")
    override_base(policy, SimpleNamespace(base_action_horizon=rc.base_action_horizon, base_inference_steps=rc.base_inference_steps,
                                          steps_per_chunk=rc.steps_per_chunk, base_warm_start=rc.get("base_warm_start")))
    if policy.cfg.warm_start:
        raise ValueError(
            "the per-phase demos plan the base cold (every chunk from pure noise, at the env's replan frames); the env's "
            "base warm-starts each plan from the last one's tail, so the demo residuals would be measured against a "
            "different base: set residual.base_warm_start: false"
        )
    k = int(rc.steps_per_chunk)
    norm = policy.normalizer
    scale = residual_scale(policy, rc.action_scale, rc.action_scale_fraction)
    target_scales = (float(rc.ee_pos_scale), math.radians(float(rc.ee_rot_scale_deg)), float(scale[9]))
    ee_rel = EERelative(rc.ee_relative_pos_unit, rc.ee_relative_rot_unit, norm)
    encode = None
    if absolute_form:
        from my_project.residual import critic_state, observation_keep, residual_observation

        keep = observation_keep(int(policy.obs_dim), [list(pair) for pair in rc.observation_drop])

        def encode(state, chunk):
            # the env wrapper's own definitions: the agent's observation (the observation, then the base chunk
            # normalized) and the residual's critic state (the observation normalized, then the base chunk)
            obs, _ = residual_observation(norm, state, chunk, bool(rc.normalize_observation), bool(rc.observe_base_action), keep)
            return obs[0], critic_state(norm, state, chunk, keep=keep)[0], None
    furniture = OmegaConf.select(cfg, "task.cfg.furniture") or {}
    thresholds = dict(
        mesh_frame=furniture.get("mesh_frame", "obj"),
        top_grip_width=furniture.get("phase_top_grip_width", 0.02), top_grip_dist=furniture.get("phase_top_grip_dist", 0.13),
        top_grip_height=furniture.get("phase_top_grip_height", 0.03), lift=furniture.get("phase_lift", 0.03),
        held_dist=furniture.get("phase_held_dist", 0.03), release_width=furniture.get("phase_release_width", 0.05),
        hole_xz=furniture.get("phase_hole_xz", 0.013), tilt_deg=furniture.get("phase_tilt_deg", 16.0),
        hold_steps=furniture.get("phase_hold_steps", 1),
    )

    terms: Dict[str, Dict[str, float]] = {}
    for term in sac.potential_terms or []:
        if term.route in routes:
            terms.setdefault(term.route, {})[str(term.channel)] = float(term.weight)
    for route, channels in terms.items():
        offline = [c for c in channels if not c.startswith("eval_state/") or c.split("/", 1)[1] not in
                   ("phase/tip_mm", "phase/tilt_deg", "phase/height_mm")]
        if offline:
            raise ValueError(f"sac.potential_terms[{route!r}] {offline}: the demos measure only eval_state/phase/tip_mm, "
                             "tilt_deg and height_mm offline")

    torch.manual_seed(int(cfg.experiment.seed))
    collected: Dict[str, Dict[str, List[torch.Tensor]]] = {}
    files = sorted(glob.glob(str(dataset / "data" / "**" / "*.parquet"), recursive=True))
    import json

    sources = {}
    with open(dataset / "meta" / "episodes.jsonl") as f:
        for line in f:
            episode = json.loads(line)
            sources[str((dataset / episode["file"]).resolve())] = episode.get("source")
    dart_rows = 0
    skipped = []
    for path in files:
        frame = pd.read_parquet(path)
        states = torch.tensor(np.stack(frame["observation.state_abs"].values), dtype=torch.float32)
        labels = torch.tensor(np.stack(frame["action.abs"].values), dtype=torch.float32)
        rewards = frame["reward"].values
        if not (rewards > 0).any():
            skipped.append((Path(path).name, "no success"))
            continue
        source = sources.get(str(Path(path).resolve()))
        if not source:
            raise ValueError(f"{path}: meta/episodes.jsonl names no source pkl for it; the executed actions are there")
        actions = executed_abs_actions(Path(source), states, labels)
        success_frame = assembling_frame(rewards)
        dart_rows += int((actions[: success_frame + 1] - labels[: success_frame + 1]).abs().amax(dim=-1).gt(1e-6).sum())
        phases = phases_from_states(states, **thresholds)
        if int(phases[: success_frame + 1].max()) < max(names.index(route) for route in routes if names.index(route) < 4):
            skipped.append((Path(path).name, f"phases reach only {names[int(phases.max())]}"))
            continue
        starts = list(range(0, success_frame + 1, k))
        horizon = int(policy.action_horizon)
        plan_starts = list(range(0, success_frame + 1, horizon))
        history = torch.stack([states[[max(t - j, 0) for j in reversed(range(policy.n_obs_steps))]] for t in plan_starts])
        with torch.no_grad():
            nobs = norm.normalize(history, "obs").clamp(-policy.cfg.obs_clip, policy.cfg.obs_clip)
            plans = norm.denormalize(policy._denoise(nobs, None), "action")
        base_chunks = torch.stack([plans[t // horizon, t % horizon : t % horizon + k] for t in starts])
        potentials = None
        if terms:
            measures = leg_measures_from_states(states, mesh_frame=thresholds["mesh_frame"])
            potentials = {route: -sum(float(w) * measures[c.split("/", 1)[1]] for c, w in channels.items())
                          for route, channels in terms.items()}
        per_route = episode_chunks(states, actions, phases, success_frame, base_chunks, k=k, names=names, routes=routes,
                                   scales=target_scales, ee_rel=ee_rel, normalizer=norm, encode=encode,
                                   axes=str(rc.target_axes), potentials=potentials, gamma=float(sac.discount_factor),
                                   residual_gripper=bool(rc.residual_gripper))
        for route, rows in per_route.items():
            for name, value in rows.items():
                collected.setdefault(route, {}).setdefault(name, []).append(value)
    for route in routes:
        rows = {name: torch.cat(values) for name, values in collected.get(route, {}).items()}
        if not rows:
            raise ValueError(f"no demo chunk starts in phase {route!r}")
        path = Path(str(out).format(route=route))
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"route": route, "std_label_names": names, "steps_per_chunk": k, "target_scales": target_scales,
                "target_axes": str(rc.target_axes), "form": "ee_relative" if ee_form else "absolute",
                "observation_drop": [list(pair) for pair in rc.observation_drop], "residual_gripper": bool(rc.residual_gripper),
                "critic_action": str(rc.critic_action),
                "base_warm_start": bool(policy.cfg.warm_start), "actions": "executed", "success": "assembling step",
                "potential_terms": terms.get(route, {}), "discount_factor": float(sac.discount_factor),
                "episodes": len(files) - len(skipped), "skipped": skipped, "dataset": str(dataset)}
        torch.save({"transitions": rows, "meta": meta}, path)
        reach = rows["actions"].abs().amax(dim=-1)
        print(f"[phase demos] {route}: {rows['actions'].shape[0]} chunk transitions, {int(rows['terminated'].sum())} attempt ends, "
              f"reward sum {float(rows['rewards'].sum()):.2f}, "
              f"|r| max per row median {float(reach.median()):.2f}, beyond 1: {float((reach > 1).float().mean()):.0%} -> {path}")
    print(f"[phase demos] {dart_rows} frames (all phases) carry a DART-perturbed executed action instead of the label")
    if skipped:
        print(f"[phase demos] skipped {skipped}")
    return 0
