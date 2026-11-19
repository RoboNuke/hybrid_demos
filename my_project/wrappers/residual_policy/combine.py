"""What the two residuals share (:class:`CombineWrapper`): the outermost of the residual wrappers, it takes the agent's
``r``, combines it with the base action in the action space of the wrapper below (the subclass's rule: added,
:mod:`~my_project.wrappers.residual_policy.additive`, or multiplied, :mod:`~my_project.wrappers.residual_policy.multiply`)
and steps. Every step it also

* publishes the critics' side input -- the base action (and the per-env scale, with phase scales) the critics' action
  is rebuilt from -- as ``infos["residual_base"]`` / ``["next_residual_base"]`` (the learner's ``side_inputs``);
* splits every ``residual/<size>`` metric per task phase (``residual/<size>_<phase>``, NaN in the other phases) when the
  env publishes its phase (``infos["eval_state"]["phase/id"]``, ``furniture.track_phases``), by the phase of the
  observation the action was chosen for;
* counts each episode's agent steps (``episode/agent_steps``, and ``episode/count_offset`` against the env's own clock);
* checks, when it builds observations from the env's phase tracker (phase scales), that the label it used is the one
  the env publishes for the learner's routing (latched on the device, raised every ``LABEL_CHECK_EVERY`` steps).

The package hooks registered for it (``my_project/wrappers/register.py``): the critics' action, the noise scale
(``residual_noise``), the neutral action.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from my_project.cfg import PHASE_NAMES
from my_project.wrappers.chain import ChainWrapper

__all__ = ["CombineWrapper", "SIDE_INPUT", "LABEL_CHECK_EVERY", "RESIDUAL_SIZES", "find_combine"]

#: the per-transition side input (the learner's ``side_inputs``) the critics' action is rebuilt from; the env publishes
#: it and its ``next_`` value in ``infos`` every step
SIDE_INPUT = "residual_base"
#: how often (env steps) the latched phase-label check is read to the host and raised on (one synchronization)
LABEL_CHECK_EVERY = 100
#: the per-step correction sizes logged (``residual/<name>``), split per phase when the phases are tracked
RESIDUAL_SIZES = ("position_mm", "rot_deg", "yaw_deg", "multiplier")


def find_combine(env: Any) -> "CombineWrapper":
    """The residual (additive or multiplicative) in ``env``'s wrapper chain; refused when there is none."""
    from my_project.wrappers.residual_policy.base_policy import find_wrapper

    found = find_wrapper(env, CombineWrapper)
    if found is None:
        raise RuntimeError("the env has no residual wrapper: enable base_policy and additive_residual or multiply_residual")
    return found


class CombineWrapper(ChainWrapper):
    #: the r that sends the base's own action
    neutral: float = 0.0
    #: the residual's rule for ``residual_noise`` (``my_project/wrappers/residual_policy/noise_scale.py``)
    noise_kind: str = ""

    def __init__(self, env: Any, cfg: Any) -> None:
        super().__init__(env)
        from my_project.cfg import q_section

        section = q_section(cfg)
        #: the learner's phase labels (std_label_names) and routed phases
        self.phase_names = list(getattr(section, "std_label_names", None) or [])
        self.routes = list(getattr(section, "route_labels", None) or [])
        #: the phase of each env's current observation (from the env's published phase/id; None until it publishes)
        self.obs_phase: Optional[torch.Tensor] = None
        #: the label an observation was built with (the tracker's), checked against the published one
        self.obs_label: Optional[torch.Tensor] = None
        self._label_mismatch = None
        self._label_mismatch_pair = None
        self._label_checks = 0
        self.agent_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.side = None
        self.hooks.hook_observations(self._observations)

    # ------------------------------------------------------------------ the subclass's rule
    def side_of(self, base: torch.Tensor, phase: Optional[torch.Tensor]) -> torch.Tensor:
        """The critics' side input for a step: the base action (``phase``: each env's phase, for phase scales)."""
        return base

    def combine(self, r: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def invert(self, executed: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        """A demo's executed action (this wrapper's inner action space) -> its r."""
        raise NotImplementedError

    def noise_target(self, r: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        """A demo frame's correction in ``residual_noise``'s units (what ``scripts/fit_noise_std.py`` fits the std of)."""
        raise NotImplementedError

    def step_metrics(self, r: torch.Tensor, metrics) -> None:
        """Extra per-step metrics of the rule (``residual/multiplier``)."""

    def uses_phase_rows(self) -> bool:
        return False

    def action_bound(self):
        """The clamp on DDPG's deterministic action, ``observations -> (lo, hi)`` (``register_action_bound``), or None."""
        return None

    # ------------------------------------------------------------------ the env
    def _tracker_phase(self) -> torch.Tensor:
        """Each env's phase from the env's tracker, which already holds the phase of the observation being built."""
        tracker = getattr(self.env.unwrapped, "phases", None)
        if tracker is None:
            raise RuntimeError("the residual's per-phase scale reads the env's phase tracker: set task.cfg.furniture.track_phases: true")
        phase = tracker.phase.reshape(-1).long().clamp(0, len(self.phase_names) - 1).to(self.device)
        self.obs_label = phase.clone()
        return phase

    def _observations(self, observations):
        phase = self._tracker_phase() if self.uses_phase_rows() else None
        self.side = self.side_of(self.base_action, phase)
        return observations

    def step(self, action):
        r = action.to(self.device)
        side = self.side
        acted_in = self.obs_phase  # the phase of the observation this action was chosen for
        sent = self.combine(r, side)
        out = self.env.step(sent)
        infos = out[4]
        if isinstance(infos, dict):
            self.step_metrics(r, infos.setdefault("metrics_to_log", {}))
            infos[SIDE_INPUT], infos[f"next_{SIDE_INPUT}"] = side, self.side
        self._count_episode(out)
        self._phase_metrics(infos, acted_in)
        self._check_phase_label(infos)
        return out

    def reset(self, **kwargs):
        # a genuine reset starts every env's episode over: the agent-step count too
        self.agent_steps.zero_()
        if self.obs_phase is not None:
            self.obs_phase = torch.zeros_like(self.obs_phase)  # every episode starts in its first phase
        return self.env.reset(**kwargs)

    def _count_episode(self, out) -> None:
        """``episode/agent_steps`` (the agent steps of the ending episode, what the learner counts) and, with the env's own
        clock (``episode/isaac_steps``), ``episode/count_offset`` = Isaac's steps - agent steps (0)."""
        _, _, terminated, truncated, infos = out
        self.agent_steps += 1
        ended = (terminated.reshape(-1) | truncated.reshape(-1)).to(torch.bool).to(self.agent_steps.device)
        if isinstance(infos, dict):
            metrics = infos.setdefault("episode_metrics_to_log", {})
            steps = self.agent_steps.to(torch.float32)
            metrics["episode/agent_steps"] = steps.clone()
            isaac = metrics.get("episode/isaac_steps")
            if isaac is not None:
                metrics["episode/count_offset"] = torch.as_tensor(isaac, device=steps.device).reshape(-1) - steps
        self.agent_steps[ended] = 0

    def _phase_metrics(self, infos, acted_in) -> None:
        """With the env's phases published: this step's correction sizes per phase, ``residual/<size>_<phase>``, by the
        phase of the observation the action was chosen for -- each env's value in its phase's channel, NaN in the others,
        so every interval's mean is over that phase's steps alone. Then the new observation's phase, kept for the next
        step: an episode-end code (4-6) is followed by the next episode's first observation, the first phase."""
        state = infos.get("eval_state") if isinstance(infos, dict) else None
        phase = state.get("phase/id") if isinstance(state, dict) else None
        if phase is None:
            return
        if acted_in is not None:
            metrics = infos.setdefault("metrics_to_log", {})
            for name in [n for n in metrics if n.startswith("residual/") and n.split("/", 1)[1] in RESIDUAL_SIZES]:
                value = metrics[name]
                nan = torch.full_like(value, float("nan"))
                for k, label in enumerate(PHASE_NAMES):
                    metrics[f"{name}_{label}"] = torch.where(acted_in.to(value.device) == k, value, nan)
        now = torch.as_tensor(phase, device=self.device).reshape(-1).round().long()
        self.obs_phase = torch.where(now >= len(PHASE_NAMES), torch.zeros_like(now), now)

    def _check_phase_label(self, infos) -> None:
        """The phase an observation was built with (the tracker's) must be the one the env publishes with it for the
        learner's routing (an episode-end code there is followed by the next episode's first observation)."""
        label = self.obs_label
        state = infos.get("eval_state") if isinstance(infos, dict) else None
        published = state.get("phase/id") if isinstance(state, dict) else None
        if label is None or published is None:
            return
        now = torch.as_tensor(published, device=label.device).reshape(-1).round().long()
        expected = torch.where(now >= len(PHASE_NAMES), torch.zeros_like(now), now)
        # on the device (torch.equal would synchronize every step); a mismatch is latched and raised at the interval
        mismatch = expected != label
        if self._label_mismatch is None:
            self._label_mismatch, self._label_mismatch_pair = mismatch, (expected, label, now)
        else:
            self._label_mismatch = self._label_mismatch | mismatch
            self._label_mismatch_pair = tuple(torch.where(mismatch, new, old)
                                              for new, old in zip((expected, label, now), self._label_mismatch_pair))
        self._label_checks += 1
        if self._label_checks % LABEL_CHECK_EVERY == 0 and bool(self._label_mismatch.any()):
            expected, label, now = self._label_mismatch_pair
            bad = torch.nonzero(self._label_mismatch).reshape(-1)[:8].tolist()
            raise RuntimeError(f"the residual built envs {bad}'s observations with phase labels {label[bad].tolist()} but the "
                               f"env published {now[bad].tolist()} for the learner: the two must be one source (found "
                               f"within the last {LABEL_CHECK_EVERY} steps)")
