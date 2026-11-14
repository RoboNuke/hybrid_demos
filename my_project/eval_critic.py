"""The critics' health in the checkpoint watcher (``my_project/residual_eval.py``): after all of a checkpoint's eval
rollouts, every learner's (each routed phase's) critics are scored on

* the eval rows -- every agent step of every env's first episode in each round, built as the learner stores a
  transition in training: the env's reward, the step whose next observation is in another phase terminal (a routed
  learner's), the critics' side inputs from the env's infos, folded into n-step rows
  (``n_step_rows``, each env's run in its phase one episode-ordered segment);
* the run's demo rows -- loaded once through the run's own demo loader (``register_demo_loader``), as training loads
  them;
* both together,

with the package's ``critic_values`` (the critic update's own prediction and target, no update) and ``health_of``. The
result keys, per agent: ``critic_<phase>/<metric>`` (both), ``..._eval`` (the eval rows) and ``..._demo`` (the demo
rows), ``<metric>`` the explained variance, the TD error's mean and max and the mean Q.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence

import torch

__all__ = ["CriticEval"]


class CriticEval:
    """``parts``: ``{route or None: learner}`` (a routed eval policy's phases, or the one learner); ``names`` the
    learner's ``std_label_names``; ``label_path`` its ``std_by_label``; ``num_agents`` the eval's slots over
    ``num_envs`` envs."""

    def __init__(self, cfg: Any, env: Any, parts: Dict[Optional[str], Any], num_agents: int, num_envs: int) -> None:
        learner = next(iter(parts.values()))
        section = learner.cfg
        self.parts = parts
        self.names = list(section.std_label_names or [])
        self.label_path = section.std_by_label
        self.num_agents, self.num_envs = int(num_agents), int(num_envs)
        self.per_agent = self.num_envs // self.num_agents
        for part in parts.values():
            part.init_side_inputs()
        self.side_names = sorted(getattr(learner, "_side_inputs", {}) or {})
        if any(route is not None for route in parts) and not self.label_path:
            raise ValueError("the critics' eval health of a routed run reads the phase tag: set the learner's std_by_label")
        self.demos = self._load_demos(cfg, env)
        self.rounds: List[Dict[str, List[torch.Tensor]]] = []

    # ------------------------------------------------------------------ demos
    def _load_demos(self, cfg: Any, env: Any) -> Dict[Optional[str], Optional[Dict[str, torch.Tensor]]]:
        """Each part's demo transitions, through the run's demo loader (None: the run has no demo file)."""
        from robonuke_rl_core.learners.sac import demo_loader

        path = getattr(cfg[cfg.trainer.learner], "demo_transitions", None)
        out: Dict[Optional[str], Optional[Dict[str, torch.Tensor]]] = {}
        if not path:
            return {route: None for route in self.parts}
        loader = demo_loader(cfg, env)
        for route, learner in self.parts.items():
            learner_cfg = copy.copy(learner.cfg)
            learner_cfg.demo_transitions = str(path).format(route=route) if route is not None else str(path)
            saved_cfg, learner.cfg = learner.cfg, learner_cfg
            learner.demo_loader = loader
            try:
                out[route] = learner._load_demo_transitions()
            finally:
                learner.cfg = saved_cfg
        print(f"[critic eval] demo rows: { {r: (0 if d is None else int(d['rewards'].shape[0])) for r, d in out.items()} }",
              flush=True)
        return out

    # ------------------------------------------------------------------ the rollouts
    def start_round(self) -> None:
        self.rounds.append({})

    def step(self, *, observations, states, actions, rewards, next_observations, next_states, terminated, truncated,
             infos, alive, start_label) -> None:
        """One agent step of every env (values ``(num_envs, ...)``); ``alive`` the envs still in their first episode
        when the step started, ``start_label`` the phase of the observation it started from."""
        n = self.num_envs

        def col(value, width=None):
            value = torch.as_tensor(value, device=alive.device)
            return value.reshape(n, -1).to(torch.float32)

        row = {"observations": col(observations), "actions": col(actions), "rewards": col(rewards),
               "next_observations": col(next_observations), "terminated": col(terminated), "truncated": col(truncated),
               "alive": alive.reshape(n, 1).to(torch.float32)}
        if states is not None:
            row["states"], row["next_states"] = col(states), col(next_states)
        for name in self.side_names:
            for stored in (name, f"next_{name}"):
                value = infos.get(stored) if isinstance(infos, dict) else None
                if value is None:
                    raise RuntimeError(f"the critics' eval health reads the side input '{stored}' from the env's infos")
                row[stored] = col(value)
        start = start_label if start_label is not None else torch.zeros(n, device=alive.device)
        nxt = self._labels(infos)
        row["start_label"] = col(start)
        row["next_label"] = col(nxt if nxt is not None else start)
        for name, value in row.items():
            self.rounds[-1].setdefault(name, []).append(value)

    def _labels(self, infos: Any) -> Optional[torch.Tensor]:
        if not self.label_path or not isinstance(infos, dict):
            return None
        value = infos.get(self.label_path)
        if value is None and "/" in self.label_path:
            key, name = self.label_path.split("/", 1)
            value = (infos.get(key) or {}).get(name) if isinstance(infos.get(key), dict) else None
        return None if value is None else torch.as_tensor(value).reshape(-1).round()

    # ------------------------------------------------------------------ rows
    def _rows(self, data: Dict[str, List[torch.Tensor]], learner: Any, route: Optional[str], envs: range) -> Optional[Dict[str, torch.Tensor]]:
        """One round's transitions of ``envs`` for one learner, in env then time order, n-step folded."""
        from robonuke_rl_core.learners.n_step import n_step_rows

        steps = {name: torch.stack(values)[:, envs.start:envs.stop] for name, values in data.items()}  # (T, E, w)
        T, E = steps["rewards"].shape[:2]
        keep = steps["alive"][..., 0] > 0.5
        terminated = steps["terminated"][..., 0] > 0.5
        if route is not None:
            index = float(self.names.index(route))
            keep = keep & (steps["start_label"][..., 0] == index)
            terminated = terminated | (steps["next_label"][..., 0] != steps["start_label"][..., 0])  # the phase switch
        flat_keep = keep.transpose(0, 1).reshape(-1)  # env-major
        if not bool(flat_keep.any()):
            return None
        order = torch.nonzero(flat_keep).reshape(-1)
        rows = {}
        for name in learner._demo_names():
            value = steps[name].transpose(0, 1).reshape(E * T, -1)[order]
            rows[name] = value
        rows["terminated"] = terminated.transpose(0, 1).reshape(-1)[order].reshape(-1, 1)
        rows["truncated"] = (steps["truncated"][..., 0] > 0.5).transpose(0, 1).reshape(-1)[order].reshape(-1, 1)
        # every segment (an env's consecutive steps here) ends terminal or truncated: the rollout stopped there
        ends = torch.ones_like(order, dtype=torch.bool)
        ends[:-1] = (order[1:] != order[:-1] + 1) | (order[1:] // T != order[:-1] // T)  # a gap, or the next env
        rows["truncated"] = rows["truncated"] | (ends.reshape(-1, 1) & ~rows["terminated"])
        rows["terminated"] = rows["terminated"].bool()
        if learner._n_step() > 1:
            rows = n_step_rows(rows, learner._n_step(), float(learner.cfg.discount_factor))
        return rows

    def results(self) -> List[Dict[str, float]]:
        """Per agent, every part's health on the eval rows, the demo rows and both (the module docstring)."""
        out: List[Dict[str, float]] = [{} for _ in range(self.num_agents)]
        for route, learner in self.parts.items():
            label = route or "all"
            eval_sets: List[Optional[Dict[str, torch.Tensor]]] = []
            for agent in range(self.num_agents):
                envs = range(agent * self.per_agent, (agent + 1) * self.per_agent)
                parts = [r for r in (self._rows(data, learner, route, envs) for data in self.rounds) if r is not None]
                eval_sets.append({k: torch.cat([p[k] for p in parts]) for k in parts[0]} if parts else None)
            demo = self.demos.get(route)
            on_eval = learner.critic_values(eval_sets)
            on_demo = learner.critic_values([demo] * self.num_agents) if demo is not None else [None] * self.num_agents
            for agent in range(self.num_agents):
                sets = {"_eval": on_eval[agent], "_demo": on_demo[agent]}
                present = [v for v in sets.values() if v is not None]
                if present:
                    sets[""] = (torch.cat([q for q, _ in present]), torch.cat([y for _, y in present]))
                for suffix, values in sets.items():
                    if values is None:
                        continue
                    q, y = values
                    for key, value in learner.health_of(q.reshape(1, -1), y.reshape(1, -1), suffix).items():
                        out[agent][f"critic_{label}/{key[len('q/'):]}"] = float(value[0])
                    out[agent][f"critic_{label}/rows{suffix}"] = int(q.numel())
        self.rounds = []
        return out
