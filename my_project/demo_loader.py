"""The residual's demo transitions, built when the learner loads a raw demo file (``my_project/raw_demos.py``), with the
env's own code: the package's ``register_demo_loader`` (every training script's ``setup()``). One loader call per
learner: a routed residual's (``route``, one of the learner's ``route_labels``) loads only its phase's steps; an unrouted
learner (``route`` None) loads every step.

The loader copies the env's data and computes nothing of its own:

* **phase** -- each frame's phase tag as the env published it with that observation (the episode's recorded
  ``infos[<std_by_label>]``, e.g. ``eval_state/phase/id``), the same channel the router reads online;
* **observations, critic states, side inputs, actions** -- the run's action space (:mod:`my_project.action_spaces`,
  the env wrapper's own instance): :meth:`~my_project.action_spaces.ActionSpace.observe` on the raw observations and
  the stored base actions, and :meth:`~my_project.action_spaces.ActionSpace.invert` on the executed actions, the
  space's state carried from frame to frame as the env carries it (an episode's first frame fresh, the base's chunks at
  frames ``0, H, 2H, ...``);
* **rewards** -- the recorded per-step reward terms (unweighted) times this run's ``task.cfg.rewards.<term>.weight``:
  the env reward manager's own sum (the package divides its ``step_dt`` back out);
* **terminated** -- the run's termination terms (those not ``time_out``) as recorded at each step, and, for a routed
  residual, the step whose next observation is in another phase (as the router marks it online): the only change made
  to the env's data. The demo ends at the env's success, so every loaded episode's last step is terminal.

The file's fingerprint (:func:`~my_project.raw_demos.raw_fingerprint`: the base run and checkpoint, its planning
settings, the task) must be this run's, or it is refused.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

__all__ = ["residual_demo_loader", "episode_steps", "episode_transitions", "env_rewards", "env_terminations"]


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
    """``(T,)`` bool: any of the run's termination terms that is not a time-out, as recorded (the demos are shorter
    than the run's episode, so a time-out never ends one)."""
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


def episode_steps(space: Any, episodes: List[Dict[str, Any]], horizon: int, routed_of, device) -> List[Dict[str, torch.Tensor]]:
    """Every frame's step inputs and ``r`` for every episode, all episodes in lockstep (one row each, as the env's
    envs): ``space.observe`` then ``space.invert`` of the executed action, the space's state carried on. ``routed_of``
    maps an episode's ``(T,)`` phase labels to its ``(T,)`` bool rows (the multiply space's routed flag). Returns per
    episode ``{policy, critic, side, r}``, each ``(T, ...)``."""
    n = len(episodes)
    lengths = [int(e["states"].shape[0]) for e in episodes]
    longest = max(lengths)
    state = space.initial_state(n, device)

    def at(name: str, t: int) -> torch.Tensor:
        return torch.stack([e[name][min(t, length - 1)] for e, length in zip(episodes, lengths)]).to(device)

    routed_rows = [routed_of(e["phase"]) for e in episodes]
    out: List[Dict[str, List[torch.Tensor]]] = [{"policy": [], "critic": [], "side": [], "r": []} for _ in episodes]
    for t in range(longest):
        obs, base, executed = at("states", t), at("base_actions", t), at("actions", t)
        fresh = torch.full((n,), t == 0, dtype=torch.bool, device=device)
        new_chunk = torch.full((n,), t % int(horizon) == 0, dtype=torch.bool, device=device)
        routed = torch.stack([rows[min(t, length - 1)] for rows, length in zip(routed_rows, lengths)]).to(device)
        step, state = space.observe(obs, base, state, fresh=fresh, new_chunk=new_chunk, routed=routed)
        r, state = space.invert(executed, step, state)
        if step["critic"] is None:
            raise NotImplementedError("the demo loader builds the critics' state from the observation: an env critic "
                                      "state of its own (Forge's) is not recorded in the raw demos")
        for i, length in enumerate(lengths):
            if t < length:
                for name, value in (("policy", step["policy"]), ("critic", step["critic"]), ("side", step["side"]), ("r", r)):
                    out[i][name].append(value[i])
    return [{name: torch.stack(values).cpu() for name, values in o.items()} for o in out]


def episode_transitions(step: Dict[str, torch.Tensor], phase: torch.Tensor, rewards: torch.Tensor,
                        terminated: torch.Tensor, index: Optional[int]) -> Dict[str, List[torch.Tensor]]:
    """One episode's transitions (``step`` from :func:`episode_steps`; ``phase`` ``(T,)`` labels; the env's
    ``rewards`` and ``terminated`` ``(T,)``): every step, or with ``index`` (a routed residual's label) the steps
    starting in that phase, the step whose next observation is in another phase terminal. The last step is the env's
    success: terminal. A terminal step's next observation is its own (unused: nothing bootstraps from it)."""
    T = int(phase.shape[0])
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
    """The ``register_demo_loader`` builder: with ``residual.enabled``, a loader that turns a raw demo file into the
    learner's transitions (the module docstring); otherwise None."""
    rc = getattr(cfg, "residual", None)
    if rc is None or not rc.enabled:
        return None
    from my_project.cfg import q_section
    from my_project.raw_demos import RAW_FORMAT, raw_fingerprint
    from my_project.residual import find_residual

    wrapper = find_residual(env)
    sec = q_section(cfg)
    names = list(sec.std_label_names)
    routes = list(sec.route_labels)
    policy = wrapper.base.policy
    if int(rc.steps_per_chunk) != 1 or wrapper.extra_dims:
        raise NotImplementedError("the demo loader makes one transition per demo step: residual.steps_per_chunk 1, no hybrid dims")
    if wrapper.env_critic:
        raise NotImplementedError("the demo loader builds the critics' state from the observation: an env critic state "
                                  "of its own (Forge's) is not recorded in the raw demos")
    if wrapper.phase_tables is not None:
        raise NotImplementedError("residual.phase_scales reads the env's tracker online; the demo loader does not rebuild "
                                  "its per-phase scale rows")
    label_path = sec.std_by_label
    if not label_path:
        raise ValueError("the demo loader reads each frame's phase tag from the recorded infos: set the learner's std_by_label")
    routed_set = routes or names  # no routing: one residual for every phase

    def routed_of(phase: torch.Tensor) -> torch.Tensor:
        return torch.tensor([0 <= int(p) < len(names) and names[int(p)] in routed_set for p in phase.tolist()], dtype=torch.bool)

    def loader(data: Any, route: Optional[str]) -> Dict[str, Any]:
        if not isinstance(data, dict) or data.get("format") != RAW_FORMAT:
            raise ValueError(f"the demo loader reads raw demo episodes ({RAW_FORMAT}): build the file with "
                             "scripts/make_raw_demos.py from demos collected with the current collector")
        made = (data.get("meta") or {}).get("fingerprint") or {}
        now = raw_fingerprint(rc, policy, made.get("dataset", ""), cfg.task_name, getattr(cfg.wandb, "entity", None))
        off = {k: (made.get(k), v) for k, v in now.items() if k != "dataset" and made.get(k) != v}
        if off:
            raise ValueError(f"sac.demo_transitions was built for another setup (file -> this run): {off}; rebuild it with "
                             "scripts/make_raw_demos.py")
        if getattr(sec, "potential_terms", None):
            raise NotImplementedError("the demo loader does not shape rewards: no potential_terms")
        episodes = []
        for episode in data["episodes"]:
            episode = _cpu(episode)  # the learner loads the file onto its device
            if label_path not in episode["infos"]:
                raise ValueError(f"the demos record no infos channel {label_path!r} (the learner's std_by_label)")
            episode["phase"] = episode["infos"][label_path].reshape(-1).round().long()
            episodes.append(episode)
        # on the wrapper's device: the space's tensors (its observation columns, scales) live there
        steps = episode_steps(wrapper.space, episodes, int(policy.action_horizon), routed_of, wrapper.scale.device)
        index = None if route is None else names.index(str(route))
        rows: Dict[str, List[torch.Tensor]] = {}
        for episode, step in zip(episodes, steps):
            rewards = env_rewards(episode["reward_terms"], cfg.task_cfg.rewards)
            terminated = env_terminations(episode["termination_terms"], cfg.task_cfg.terminations)
            for name, values in episode_transitions(step, episode["phase"], rewards, terminated, index).items():
                rows.setdefault(name, []).extend(values)
        if not rows:
            raise ValueError(f"no demo frame is in phase {route!r}")
        transitions = {name: torch.stack(values) for name, values in rows.items()}
        print(f"[demo loader] {route or 'all phases'}: {transitions['actions'].shape[0]} transitions from "
              f"{len(episodes)} raw episodes, {int(transitions['terminated'].sum())} terminal", flush=True)
        return {"transitions": transitions, "meta": {"route": route, "fingerprint": made}}

    return loader
