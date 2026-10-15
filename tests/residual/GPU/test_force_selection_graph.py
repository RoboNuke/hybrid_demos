"""``force_selection`` declares ``graph_safe``: it must capture in a CUDA graph and replay to the
eager value. Needs CUDA only (no Isaac Sim)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from my_project.losses import ForceSelectionLoss
from robonuke_rl_core.losses import LossContext

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("mode, axes", [("no_torque", (0, 1, 2)), ("with_torque", (2,))])
def test_it_captures_and_replays_to_the_eager_value(mode, axes):
    if not torch.cuda.is_available():
        pytest.fail("this GPU test needs CUDA")
    device = torch.device("cuda")
    loss = ForceSelectionLoss(scale=1.0, offset=3.0, mode=mode, axes=axes)
    obs = torch.randn(16, 20, device=device) * 5
    prob = torch.rand(16, len(axes), device=device).clamp(0.05, 0.95)
    ctx = LossContext(learner=SimpleNamespace(num_agents=2), target="policy", sampled={"observations": obs}, policy_outputs={"selection_prob": prob})

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        loss.compute(ctx)  # the eager warm-up, as StepGraphs runs it
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = loss.compute(ctx)
    for _ in range(3):
        obs.copy_(torch.randn_like(obs) * 5)
        prob.copy_(torch.rand_like(prob).clamp(0.05, 0.95))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.allclose(out, loss.compute(ctx), atol=1e-6)
