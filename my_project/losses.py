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
