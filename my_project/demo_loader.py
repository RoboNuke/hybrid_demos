"""The residual's demo transitions, built when the learner loads a raw demo file (``my_project/raw_demos.py``), with the
env's own code: the package's ``register_demo_loader`` (every training script's ``setup()``).

For each raw episode (frames ``0..success_frame``):

1. the task phase of every frame, labelled by the env tracker's own checks on the recorded observations
   (Robonuke_Furniture_Bench ``assembly/phases.phases_from_states``: ``phase_checks`` / ``phase_step``, with this run's
   ``task.cfg.furniture.phase_*`` thresholds);
2. ``action_frame: multiply``: the env wrapper's per-step inputs along the demo
   (:meth:`~my_project.residual_multiply.MultiplyResidual.step_inputs`: the observation, the base's motion, the current
   target, the critics' side input), the chain's current target being the demo's own target after a frame in a phase
   with a residual (the residual followed the demo there) and the base's after one without, the EE at every chunk start
   (frames ``0, H, 2H, ...``, as the env's base replans from the episode start); and each frame's ``r``, the demo's motion
   over the base's (:meth:`~my_project.residual_multiply.MultiplyResidual.invert`);
3. the route's rows: a frame in the route's phase, rewarded 1 when the next frame is in a later ``std_label_names``
   entry or the demo succeeds on that frame (``success``); the attempt then ends (terminated), as it does at the demo's
   end.

The file's fingerprint (:func:`~my_project.raw_demos.raw_fingerprint`: the base run and checkpoint, its planning
settings, the task) must be this run's, or it is refused.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch

__all__ = ["residual_demo_loader", "episode_rows"]


def _phase_thresholds(cfg: Any) -> Dict[str, Any]:
    f = cfg.task_cfg.furniture
    return dict(mesh_frame=f.mesh_frame, top_grip_width=f.phase_top_grip_width, top_grip_dist=f.phase_top_grip_dist,
                top_grip_height=f.phase_top_grip_height, lift=f.phase_lift, held_dist=f.phase_held_dist,
                release_width=f.phase_release_width, hole_xz=f.phase_hole_xz, tilt_deg=f.phase_tilt_deg,
                hold_steps=int(f.phase_hold_steps))


def episode_rows(multiply, episode: Dict[str, torch.Tensor], labels: torch.Tensor, names: List[str], routes: List[str],
                 route: str, horizon: int, keep) -> Dict[str, torch.Tensor]:
    """One raw episode's transitions for ``route`` (the module docstring), ``labels`` ``(T,)`` its frames' phases."""
    states, demo, base = episode["states"], episode["actions"], episode["base_actions"]
    T = states.shape[0]
    success_frame = int(episode["success_frame"])
    if success_frame != T - 1:
        raise ValueError(f"a raw episode holds frames 0..success_frame, got {T} frames and success_frame {success_frame}")
    routed = torch.tensor([names[int(p)] in routes for p in labels])
    chunk_start = torch.tensor([t % int(horizon) == 0 for t in range(T)])
    # the chain's state before each frame: the demo's target after a routed frame, the base's otherwise; frame 0 the EE
    demo_pose = torch.cat((demo[:, 0:9], base[:, 9:10]), dim=-1)
    after = torch.where(routed.reshape(-1, 1), demo_pose, base)
    prev_sent = torch.cat((torch.cat((states[:1, 0:9], base[:1, 9:10]), dim=-1), after[:-1]))
    prev_base = torch.cat((base[:1], base[:-1]))
    inputs = multiply.step_inputs(states, base, keep, routed, prev_sent=prev_sent, prev_base=prev_base, chunk_start=chunk_start)
    r = multiply.invert(inputs["d"], demo, inputs["ref"])
    index = names.index(route)
    success_label = names.index("success") if "success" in names else len(names)
    rows: Dict[str, List[torch.Tensor]] = {}
    for t in torch.nonzero(labels == index).reshape(-1).tolist():
        nxt_label = success_label if t == success_frame else int(labels[t + 1])
        local = index < nxt_label <= success_label
        ended = local or t == T - 1
        n = t if ended else t + 1
        row = {"observations": inputs["policy"][t], "next_observations": inputs["policy"][n],
               "states": inputs["critic"][t], "next_states": inputs["critic"][n], "actions": r[t],
               "rewards": torch.tensor([1.0 if local else 0.0]), "terminated": torch.tensor([bool(ended)]),
               "truncated": torch.tensor([False]), "success": torch.tensor([1.0 if local else 0.0]),
               "residual_base": inputs["side"][t], "next_residual_base": inputs["side"][n]}
        for name, value in row.items():
            rows.setdefault(name, []).append(value)
    return {name: torch.stack(values) for name, values in rows.items()}


def residual_demo_loader(cfg: Any, env: Any):
    """The ``register_demo_loader`` builder: with ``residual.enabled`` and ``action_frame: multiply``, a loader that
    turns a raw demo file into the route's transitions (the module docstring); otherwise None."""
    rc = getattr(cfg, "residual", None)
    if rc is None or not rc.enabled or rc.action_frame != "multiply":
        return None
    from furniture_lab.tasks.assembly.phases import phases_from_states

    from my_project.cfg import q_section
    from my_project.raw_demos import RAW_FORMAT, raw_fingerprint
    from my_project.residual import find_residual

    wrapper = find_residual(env)
    sec = q_section(cfg)
    names, routes = list(sec.std_label_names), list(sec.route_labels)
    thresholds = _phase_thresholds(cfg)
    policy = wrapper.base.policy
    keep = None if wrapper.keep is None else wrapper.keep.cpu()

    def loader(data: Any, route: Any) -> Dict[str, Any]:
        if not isinstance(data, dict) or data.get("format") != RAW_FORMAT:
            raise ValueError("residual.action_frame: multiply loads raw demo episodes: build the file with scripts/make_raw_demos.py")
        if route is None:
            raise NotImplementedError("the raw demo loader builds per-phase transitions: set the learner's route_labels")
        made = (data.get("meta") or {}).get("fingerprint") or {}
        now = raw_fingerprint(rc, policy, made.get("dataset", ""), cfg.task_name, getattr(cfg.wandb, "entity", None))
        off = {k: (made.get(k), v) for k, v in now.items() if k != "dataset" and made.get(k) != v}
        if off:
            raise ValueError(f"sac.demo_transitions was built for another setup (file -> this run): {off}; rebuild it with "
                             "scripts/make_raw_demos.py")
        if getattr(sec, "potential_terms", None):
            raise NotImplementedError("the raw demo loader does not shape rewards: no potential_terms")
        out: Dict[str, List[torch.Tensor]] = {}
        for episode in data["episodes"]:
            episode = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in episode.items()}
            labels = phases_from_states(episode["states"], **thresholds).cpu()
            rows = episode_rows(wrapper.multiply, episode, labels, names, routes, str(route), int(policy.action_horizon), keep)
            for name, value in rows.items():
                out.setdefault(name, []).append(value)
        if not out:
            raise ValueError(f"no demo frame is in phase {route!r}")
        transitions = {name: torch.cat(values) for name, values in out.items()}
        print(f"[demo loader] {route}: {transitions['actions'].shape[0]} transitions from {len(data['episodes'])} raw episodes, "
              f"{int(transitions['terminated'].sum())} attempt ends", flush=True)
        return {"transitions": transitions, "meta": {"route": route, "fingerprint": made}}

    return loader
