"""``force_obs`` and the ``force_selection`` loss, and ``hybrid`` on the furniture family. CPU only."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import my_project.losses  # noqa: F401  -- @register_loss
from my_project.cfg import ForceObsCfg, HybridCfg
from my_project.losses import ForceSelectionLoss
from robonuke_rl_core.losses import LOSSES, LossContext, LossesCfg, LossTermCfg, build_aux_losses

AGENTS, ROWS = 2, 4


def _obs(force, mode="no_torque", lead=5):
    """Observations: ``lead`` other columns, then the force (and a zero torque)."""
    force = torch.as_tensor(force, dtype=torch.float32)
    parts = [torch.randn(force.shape[0], lead), force]
    if mode == "with_torque":
        parts.append(torch.full((force.shape[0], 3), 99.0))  # a torque that must not be read as force
    return torch.cat(parts, dim=-1)


def _ctx(obs, prob):
    learner = SimpleNamespace(num_agents=AGENTS)
    return LossContext(learner=learner, target="policy", sampled={"observations": obs}, policy_outputs={"selection_prob": prob})


@pytest.mark.parametrize("mode", ["no_torque", "with_torque"])
def test_the_target_is_the_sigmoid_of_the_scaled_force_magnitude(mode):
    loss = ForceSelectionLoss(scale=2.0, offset=5.0, mode=mode)
    force = [[5.0, -5.0, 0.0], [-10.0, 6.0, 4.0]]
    target = loss.target(_obs(force, mode))
    want = torch.sigmoid(2.0 * (torch.tensor(force).abs() - 5.0))
    assert torch.allclose(target, want)
    assert torch.allclose(target[0, :2], torch.tensor([0.5, 0.5]))  # |f| = offset either sign


def test_the_loss_is_the_bce_per_agent():
    loss = ForceSelectionLoss(scale=1.0, offset=3.0)
    force = torch.randn(AGENTS * ROWS, 3) * 5
    prob = torch.rand(AGENTS * ROWS, 3).clamp(0.05, 0.95)
    raw = loss.compute(_ctx(_obs(force), prob))
    t = torch.sigmoid(force.abs() - 3.0)
    bce = -(t * prob.log() + (1 - t) * (1 - prob).log())
    assert raw.shape == (AGENTS,)
    assert torch.allclose(raw, bce.view(AGENTS, -1).mean(dim=-1), atol=1e-6)


def test_one_agents_rows_do_not_move_anothers_value():
    loss = ForceSelectionLoss(scale=1.0, offset=3.0)
    force = torch.randn(AGENTS * ROWS, 3)
    prob = torch.full((AGENTS * ROWS, 3), 0.3)
    plain = loss.compute(_ctx(_obs(force), prob))
    force[ROWS:] *= 100.0  # agent 1 only
    changed = loss.compute(_ctx(_obs(force), prob))
    assert torch.equal(plain[0], changed[0]) and not torch.equal(plain[1], changed[1])


def test_the_gradient_pushes_toward_force_when_pressing():
    loss = ForceSelectionLoss(scale=5.0, offset=2.0)
    prob = torch.full((AGENTS * ROWS, 3), 0.5, requires_grad=True)
    force = torch.zeros(AGENTS * ROWS, 3)
    force[:, 2] = 20.0  # pressing hard on z, nothing on x, y
    loss.compute(_ctx(_obs(force), prob)).sum().backward()
    assert bool((prob.grad[:, 2] < 0).all())  # descent raises p(force) on z
    assert bool((prob.grad[:, 0:2] > 0).all())  # and lowers it on the free axes


def test_it_is_graph_safe_and_builds_with_the_flag():
    assert LOSSES["force_selection"].graph_safe is True
    terms = [LossTermCfg(name="force_selection", target="policy", weight=1.0, graph_safe=True, kwargs={"scale": 1.0, "offset": 5.0})]
    assert build_aux_losses(LossesCfg(terms=terms), AGENTS).graph_safe is True


def test_a_policy_without_selection_says_so():
    loss = ForceSelectionLoss(scale=1.0, offset=1.0)
    ctx = LossContext(learner=SimpleNamespace(num_agents=AGENTS), target="policy", sampled={"observations": _obs(torch.zeros(8, 3))}, policy_outputs={})
    with pytest.raises(RuntimeError, match="selection_prob"):
        loss.compute(ctx)


@pytest.mark.parametrize("kwargs", [{"scale": 0.0, "offset": 1.0}, {"scale": 1.0, "offset": 1.0, "mode": "x"}, {"scale": 1.0, "offset": 1.0, "axes": [0, 3]}])
def test_bad_kwargs_raise(kwargs):
    with pytest.raises(ValueError):
        ForceSelectionLoss(**kwargs)


# ---------------------------------------------------------------------- force_obs
def _losses(*modes):
    return SimpleNamespace(terms=[SimpleNamespace(name="force_selection", kwargs={"mode": m}) for m in modes])


def test_force_obs_widths_and_checks():
    assert ForceObsCfg(enabled=True).width == 3
    assert ForceObsCfg(enabled=True, mode="with_torque").width == 6
    assert ForceObsCfg(enabled=False).width == 0
    with pytest.raises(ValueError, match="mode"):
        ForceObsCfg(mode="torque_only").validate(None)
    with pytest.raises(ValueError, match="force_obs.enabled"):
        ForceObsCfg(enabled=False).validate(SimpleNamespace(losses=_losses("no_torque")))
    with pytest.raises(ValueError, match="mode"):
        ForceObsCfg(enabled=True, mode="no_torque").validate(SimpleNamespace(losses=_losses("with_torque")))
    ForceObsCfg(enabled=True, mode="with_torque").validate(SimpleNamespace(losses=_losses("with_torque")))


# ---------------------------------------------------------------------- hybrid on furniture
FURNITURE_HYBRID = "Isaac-Furniture-OneLeg-FBAbs-ResiP-Hybrid-v0"


def _fcfg(task=FURNITURE_HYBRID, owns=True, bernoulli=(10, 11, 12)):
    task_cfg = SimpleNamespace(hybrid=SimpleNamespace(force_axes=[1, 1, 1, 0, 0, 0])) if owns else SimpleNamespace()
    return SimpleNamespace(task_name=task, task_cfg=task_cfg, model=SimpleNamespace(actor=SimpleNamespace(bernoulli_action_dims=list(bernoulli))))


def test_furniture_hybrid_layout():
    h, cfg = HybridCfg(enabled=True), _fcfg()
    h.validate(cfg)
    assert h.selection_dims_for(cfg) == [10, 11, 12]
    assert h.extra_dims_for(cfg) == 6
    assert HybridCfg(enabled=False).extra_dims_for(_fcfg(owns=False)) == 0


def test_furniture_hybrid_refuses():
    with pytest.raises(ValueError, match="hybrid.enabled: true"):
        HybridCfg(enabled=False).validate(_fcfg())  # the hybrid task id with the section off
    with pytest.raises(ValueError, match="Hybrid-v0"):
        HybridCfg(enabled=True).validate(_fcfg(task="Isaac-Furniture-OneLeg-FBAbs-ResiP-v0", owns=False))
    with pytest.raises(ValueError, match="task.cfg.hybrid"):
        HybridCfg(enabled=True, force_target_bounds=[10.0] * 3 + [5.0] * 3).validate(_fcfg())
    with pytest.raises(ValueError, match=r"\[10, 11, 12\]"):
        HybridCfg(enabled=True).validate(_fcfg(bernoulli=(4, 5, 6)))
    HybridCfg(enabled=False).validate(_fcfg(task="Isaac-Furniture-OneLeg-FBAbs-ResiP-v0", owns=False))
