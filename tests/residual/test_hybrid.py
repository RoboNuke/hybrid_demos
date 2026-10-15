"""``hybrid.*``'s config checks, the demos' position-control block and the demo fingerprint's
default rule for the new section. CPU only; the controller itself needs a live Forge env."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from my_project.cfg import HybridCfg
from my_project.residual_demos import _section_defaults
from my_project.wrappers import hybrid_demo_block


def _cfg(bernoulli=(4, 5, 6), world_pose=True, controller=False):
    return SimpleNamespace(
        task_name="Isaac-Forge-PegInsert-Direct-v0",
        action=SimpleNamespace(world_pose=world_pose),
        controller=SimpleNamespace(enabled=controller),
        model=SimpleNamespace(actor=SimpleNamespace(bernoulli_action_dims=list(bernoulli))),
    )


def test_layout():
    hybrid = HybridCfg(enabled=True)
    assert hybrid.selection_dims == [4, 5, 6]
    assert hybrid.extra_dims == 6
    assert HybridCfg(enabled=False).extra_dims == 0
    assert HybridCfg(enabled=True, force_axes=[0, 0, 1, 0, 0, 0]).selection_dims == [4]


def test_validate_accepts_the_selection_block():
    HybridCfg(enabled=True).validate(_cfg())
    HybridCfg(enabled=False).validate(_cfg(bernoulli=()))


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"bernoulli": (7, 8, 9)}, "selection block"),  # the package's layout, not ours
        ({"bernoulli": ()}, "selection block"),
        ({"world_pose": False}, "world_pose"),
        ({"controller": True}, "controller.enabled"),
    ],
)
def test_validate_refuses(kwargs, match):
    with pytest.raises(ValueError, match=match):
        HybridCfg(enabled=True).validate(_cfg(**kwargs))


def test_validate_runs_the_package_field_checks():
    with pytest.raises(ValueError):
        HybridCfg(enabled=True, force_target_bounds=[10.0, 10.0, 10.0, 0.0, 5.0, 5.0]).validate(None)


def test_demo_block_is_position_control():
    block = hybrid_demo_block(3, rows=2, device="cpu")
    assert torch.equal(block, torch.tensor([[-1.0, -1.0, -1.0, 0.0, 0.0, 0.0]] * 2))


def test_demo_block_reads_as_position_to_the_controller():
    """The package's own split of a world-pose-free action: the block selects S = 0."""
    from robonuke_rl_core.envs.interface import ActionInterface

    interface = ActionInterface(HybridCfg(enabled=True).controller_cfg())
    action = torch.cat((torch.zeros(1, 7), hybrid_demo_block(3, rows=1, device="cpu")), dim=-1)
    _, selection, force_target, _, _ = interface.split(action)
    assert torch.equal(selection, torch.zeros(1, 6))
    assert torch.equal(force_target, torch.zeros(1, 6))


def test_fingerprint_defaults_cover_the_new_section():
    """A demo file made before ``hybrid`` existed still matches a run at its defaults."""
    defaults = _section_defaults()
    assert defaults["hybrid.enabled"] is False
    assert defaults["hybrid.force_axes"] == [1, 1, 1, 0, 0, 0]
