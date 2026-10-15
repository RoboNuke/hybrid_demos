"""The project's auxiliary losses. Imported (inside ``setup``) so ``@register_loss`` runs
before the config loads; a YAML ``losses.terms`` entry then turns a loss on. Follow the
package CLAUDE.md's "Add a loss" recipe: one raw value per agent, never averaged across
agents."""

import torch

from robonuke_rl_core.losses import AuxLoss, LossContext, register_loss


@register_loss
class DemoLogProbPenalty(AuxLoss):
    """Example only: penalize the policy's mean log-probability (an entropy-bonus shape).

    Demonstrates the contract -- read what the context carries, reshape to
    ``(num_agents, rows)`` and reduce per agent. Replace with your project's real loss.
    """

    name = "demo_log_prob_penalty"
    supported_targets = ("policy",)

    def compute(self, ctx: LossContext) -> torch.Tensor:
        if ctx.log_prob is None:
            raise ValueError("demo_log_prob_penalty needs the policy block's log_prob")
        return ctx.log_prob.view(ctx.learner.num_agents, -1).mean(dim=1)


@register_loss
class ForceSelectionLoss(AuxLoss):
    """Push the hybrid selection toward force control the harder the arm is pushing.

    Per selection axis ``i``, binary cross entropy between the policy's probability of **force**
    control (``policy_outputs["selection_prob"]``, the bit is 1 for force) and a soft target
    from the measured force on that axis::

        target_i = sigmoid(scale * (|f_i| - offset))

    ``offset`` (N) is the force at which the target is 0.5; ``scale`` (1/N) how sharply it turns
    from position (light contact or none) to force (pressing). ``f`` is the force the agent
    observed, read from the stored observation's last columns (``force_obs``, appended last):
    ``force_obs.mode`` says whether a torque follows it. ``axes`` are the force components, in
    selection order (default x, y, z: ``force_axes [1, 1, 1, 0, 0, 0]``).

    Graph-safe: tensor ops on the sampled batch and the policy outputs only.
    """

    name = "force_selection"
    supported_targets = ("policy",)
    graph_safe = True
    #: the probability is clamped to this before the log, so a saturated head is finite
    EPS = 1.0e-7

    def __init__(self, *, scale: float, offset: float, mode: str = "no_torque", axes=(0, 1, 2)) -> None:
        from my_project.cfg import ForceObsCfg

        if mode not in ForceObsCfg.MODES:
            raise ValueError(f"force_selection: mode must be one of {list(ForceObsCfg.MODES)} (force_obs.mode), got {mode!r}")
        if float(scale) <= 0:
            raise ValueError(f"force_selection: scale must be > 0 (1/N), got {scale}")
        axes = [int(a) for a in axes]
        if not axes or min(axes) < 0 or max(axes) > 2 or len(set(axes)) != len(axes):
            raise ValueError(f"force_selection: axes must be distinct force components in 0..2, got {axes}")
        self.scale, self.offset = float(scale), float(offset)
        self.width = ForceObsCfg.MODES[mode]
        self.axes = axes
        #: the axes as an index tensor per device, made on the first (eager) call: a Python list
        #: as an index is a host-to-device copy every call, which a CUDA graph cannot capture
        self._index = {}

    def target(self, observations: torch.Tensor) -> torch.Tensor:
        """``sigmoid(scale * (|f| - offset))`` per selection axis, ``(rows, len(axes))``."""
        start = observations.shape[-1] - self.width
        key = (observations.device, start)
        index = self._index.get(key)
        if index is None:
            index = self._index[key] = torch.tensor(self.axes, device=observations.device) + start
        force = observations.index_select(-1, index)
        return torch.sigmoid(self.scale * (force.abs() - self.offset))

    def compute(self, ctx: LossContext) -> torch.Tensor:
        probability = (ctx.policy_outputs or {}).get("selection_prob")
        if probability is None:
            raise RuntimeError(
                "loss 'force_selection' needs the actor's selection_prob: a hybrid action space "
                "(hybrid.enabled, model.actor.bernoulli_action_dims on its selection block)"
            )
        target = self.target(self.sampled(ctx, "observations")).to(probability.dtype)
        if probability.shape != target.shape:
            raise ValueError(
                f"force_selection: {tuple(probability.shape)} selection probabilities but axes {self.axes} "
                f"give {tuple(target.shape)} targets: axes must list one force component per selection dim"
            )
        probability = probability.clamp(self.EPS, 1.0 - self.EPS)
        per_row = -(target * torch.log(probability) + (1.0 - target) * torch.log(1.0 - probability))
        # per agent: the mean over its own rows and axes
        return per_row.view(ctx.learner.num_agents, -1).mean(dim=-1)
