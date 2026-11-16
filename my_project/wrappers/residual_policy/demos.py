"""The residual's demo transitions, built when the learner loads a raw demo file (``my_project/raw_demos.py``), with the
run's own residual wrappers: the package's ``register_demo_loader`` (registered in ``my_project/wrappers/register.py``).
One loader call per learner: a routed residual's (``route``, one of the learner's ``route_labels``) loads only its
phase's steps; an unrouted learner (``route`` None) loads every step.

Every demo frame goes through exactly what the env runs, offline, all episodes in lockstep:

* **observations, critic observations, side inputs** -- the recorded observation and the stored base action, laid out
  as the base policy lays them out, then every conversion below the residual (drop columns, target tracking,
  EE-relative), each with its state carried from frame to frame as the env carries it (an episode's first frame
  fresh, the base's chunks at frames ``0, H, 2H, ...``), then the residual's side input;
* **actions** -- the executed action through every conversion's inverse, then the residual's (``(executed - base) /
  scale`` or ``executed / base``);
* **phase** -- each frame's phase tag as the env published it with that observation (the recorded ``infos[<std_by_label>]``),
  the channel the router reads online;
* **rewards / terminated** -- the recorded unweighted reward terms times this run's ``task.cfg.rewards.<term>.weight``
  and the run's non-time-out termination terms (a demo with no env data, robomimic's: reward 1 on its last frame, the
  success); for a routed residual, the step whose next observation is in another phase is terminal too. Every loaded
  episode's last step is terminal (the demo ends at the env's success).

The file's fingerprint (:func:`~my_project.raw_demos.raw_fingerprint`: the base run and checkpoint, its planning
settings, the task) must be this run's.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

__all__ = ["residual_demo_loader", "convert_episodes", "episode_transitions", "env_rewards", "env_terminations"]


def _terms(cfg_section: Any) -> Dict[str, Any]:
    """``{name: term cfg}`` of an env cfg's reward / termination section (terms set to None are off)."""
    return {name: term for name, term in vars(cfg_section).items() if term is not None and hasattr(term, "func")}


def _cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.cpu()
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    return value


def env_rewards(reward_terms: Dict[str, torch.Tensor], rewards_cfg: Any) -> torch.Tensor:
    """``(T,)``: sum over the run's reward terms of ``weight x`` the recorded unweighted value."""
    terms = _terms(rewards_cfg)
    missing = [name for name in terms if name not in reward_terms]
    if missing:
        raise ValueError(f"the demos record no reward term {missing}: collect them with the current collector")
    total = None
    for name, term in terms.items():
        value = float(term.weight) * reward_terms[name].to(torch.float32)
        total = value if total is None else total + value
    if total is None:
        raise ValueError("the run's env has no reward terms")
    return total


def env_terminations(termination_terms: Dict[str, torch.Tensor], terminations_cfg: Any) -> torch.Tensor:
    """``(T,)`` bool: any of the run's termination terms that is not a time-out, as recorded."""
    terms = {name: term for name, term in _terms(terminations_cfg).items() if not getattr(term, "time_out", False)}
    missing = [name for name in terms if name not in termination_terms]
    if missing:
        raise ValueError(f"the demos record no termination term {missing}: collect them with the current collector")
    out = None
    for name in terms:
        value = termination_terms[name].to(torch.bool)
        out = value if out is None else out | value
    if out is None:
        raise ValueError("the run's env has no termination that is not a time-out")
    return out


def convert_episodes(combine: Any, episodes: List[Dict[str, Any]], phases: List[Optional[torch.Tensor]],
                     device: Any) -> List[Dict[str, torch.Tensor]]:
    """Every frame's agent observation, critic observation, side input and r for every episode (the module docstring),
    all episodes in lockstep (one row each, as the env's envs). ``combine`` the run's residual wrapper (its chain gives
    the conversions); ``phases`` each episode's ``(T,)`` labels (None: no phase scales read). Returns per episode
    ``{policy, critic, side, r}``, each ``(T, ...)``, on the CPU."""
    chain = combine.chain()
    conversions = [node.conversion for node in chain[1:-1]]
    horizon = int(chain[0].policy.action_horizon)
    n = len(episodes)
    lengths = [int(e["states"].shape[0]) for e in episodes]
    states = [c.initial_state(n, device) for c in conversions]

    def at(name: str, t: int) -> torch.Tensor:
        return torch.stack([e[name][min(t, length - 1)] for e, length in zip(episodes, lengths)]).to(device)

    def phase_at(t: int) -> Optional[torch.Tensor]:
        if combine.uses_phase_rows():
            return torch.stack([p[min(t, length - 1)] for p, length in zip(phases, lengths)]).to(device).long()
        return None

    out: List[Dict[str, List[torch.Tensor]]] = [{"policy": [], "critic": [], "side": [], "r": []} for _ in episodes]
    for t in range(max(lengths)):
        obs, base, executed = at("states", t), at("base_actions", t), at("actions", t)
        fresh = torch.full((n,), t == 0, dtype=torch.bool, device=device)
        new_chunk = torch.full((n,), t % horizon == 0, dtype=torch.bool, device=device)
        policy, critic, ctxs = obs, obs, []
        for k, conversion in enumerate(conversions):
            policy, critic, base, ctx, states[k] = conversion.observe(policy, critic, base, states[k], fresh=fresh,
                                                                      new_chunk=new_chunk)
            ctxs.append(ctx)
        side = combine.side_of(base, phase_at(t))
        for k, conversion in enumerate(conversions):
            executed, states[k] = conversion.invert(executed, ctxs[k], states[k])
        r = combine.invert(executed, side)
        agent = torch.cat((policy, base), dim=-1)
        for i, length in enumerate(lengths):
            if t < length:
                for name, value in (("policy", agent), ("critic", critic), ("side", side), ("r", r)):
                    out[i][name].append(value[i])
    return [{name: torch.stack(values).cpu() for name, values in o.items()} for o in out]


def episode_transitions(step: Dict[str, torch.Tensor], phase: Optional[torch.Tensor], rewards: torch.Tensor,
                        terminated: torch.Tensor, index: Optional[int]) -> Dict[str, List[torch.Tensor]]:
    """One episode's transitions (``step`` from :func:`convert_episodes`; ``phase`` ``(T,)`` labels or None; the env's
    ``rewards`` and ``terminated`` ``(T,)``): every step, or with ``index`` (a routed residual's label) the steps
    starting in that phase, the step whose next observation is in another phase terminal. The last step is the env's
    success: terminal. A terminal step's next observation is its own (unused: nothing bootstraps from it)."""
    T = int(rewards.shape[0])
    terminated = terminated.clone()
    terminated[T - 1] = True
    if index is not None:
        switch = torch.zeros(T, dtype=torch.bool)
        switch[: T - 1] = phase[1:] != phase[:-1]
        terminated = terminated | switch
    keep = torch.ones(T, dtype=torch.bool) if index is None else phase == index
    rows: Dict[str, List[torch.Tensor]] = {}
    for t in torch.nonzero(keep).reshape(-1).tolist():
        n = t if bool(terminated[t]) else t + 1
        row = {"observations": step["policy"][t], "next_observations": step["policy"][n],
               "states": step["critic"][t], "next_states": step["critic"][n], "actions": step["r"][t],
               "rewards": rewards[t].reshape(1), "terminated": terminated[t].reshape(1),
               "truncated": torch.tensor([False]), "residual_base": step["side"][t],
               "next_residual_base": step["side"][n]}
        for name, value in row.items():
            rows.setdefault(name, []).append(value)
    return rows


def residual_demo_loader(cfg: Any, env: Any):
    """The ``register_demo_loader`` builder: with ``base_policy.enabled``, a loader that turns a raw demo file into the
    learner's transitions (the module docstring); otherwise None."""
    if not getattr(getattr(cfg, "base_policy", None), "enabled", False):
        return None
    from my_project.cfg import q_section
    from my_project.raw_demos import RAW_FORMAT, raw_fingerprint
    from my_project.wrappers.residual_policy.combine import find_combine

    combine = find_combine(env)
    if combine.base_policy.own_critic:
        raise NotImplementedError("the demo loader builds the critics' observation from the recorded observation: an env "
                                  "critic group of its own (Forge's) is not recorded in the raw demos")
    sec = q_section(cfg)
    names = list(sec.std_label_names or [])
    label_path = sec.std_by_label
    policy = combine.base_policy.policy

    def loader(data: Any, route: Optional[str]) -> Dict[str, Any]:
        if not isinstance(data, dict) or data.get("format") != RAW_FORMAT:
            raise ValueError(f"the demo loader reads raw demo episodes ({RAW_FORMAT}): build the file with "
                             "scripts/make_raw_demos.py")
        made = (data.get("meta") or {}).get("fingerprint") or {}
        now = raw_fingerprint(cfg.base_policy, policy, made.get("dataset", ""), cfg.task_name, getattr(cfg.wandb, "entity", None))
        off = {k: (made.get(k), v) for k, v in now.items() if k != "dataset" and made.get(k) != v}
        if off:
            raise ValueError(f"the demo file was built for another base or task (file -> this run): {off}; rebuild it with "
                             "scripts/make_raw_demos.py")
        if getattr(sec, "potential_terms", None):
            raise NotImplementedError("the demo loader does not shape rewards: no potential_terms")
        index = None if route is None else names.index(route)
        needs_phase = route is not None or combine.uses_phase_rows()
        episodes, phases, rewards, terminated = [], [], [], []
        for episode in data["episodes"]:
            episode = _cpu(episode)  # the learner loads the file onto its device
            T = int(episode["states"].shape[0])
            phase = None
            if needs_phase:
                infos = episode.get("infos") or {}
                if not label_path or label_path not in infos:
                    raise ValueError(f"the demos record no phase channel {label_path!r} (the learner's std_by_label), which "
                                     "a routed residual or phase scales read")
                phase = infos[label_path].reshape(-1).round().long()
            if "reward_terms" in episode:
                reward = env_rewards(episode["reward_terms"], cfg.task_cfg.rewards)
                done = env_terminations(episode["termination_terms"], cfg.task_cfg.terminations)
            else:  # robomimic / DexMimicGen demos: the env's sparse success reward on the last frame, which ends it
                reward, done = torch.zeros(T), torch.zeros(T, dtype=torch.bool)
                reward[T - 1], done[T - 1] = 1.0, True
            episodes.append(episode)
            phases.append(phase)
            rewards.append(reward)
            terminated.append(done)
        steps = convert_episodes(combine, episodes, phases, combine.device)
        rows: Dict[str, List[torch.Tensor]] = {}
        for step, phase, reward, done in zip(steps, phases, rewards, terminated):
            for name, values in episode_transitions(step, phase, reward, done, index).items():
                rows.setdefault(name, []).extend(values)
        if not rows:
            raise ValueError(f"no demo step starts in phase {route!r}")
        transitions = {name: torch.stack(values) for name, values in rows.items()}
        print(f"[demos] {transitions['rewards'].shape[0]} transitions from {len(episodes)} episodes"
              + (f" (phase {route})" if route is not None else "")
              + f", reward sum {float(transitions['rewards'].sum()):.1f}", flush=True)
        return {"transitions": transitions, "meta": {"route": route, "fingerprint": made}}

    return loader
