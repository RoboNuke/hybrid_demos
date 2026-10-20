"""bc.normalization: per_unit -- one range per physical unit over obs and action together."""

from types import SimpleNamespace

import pytest
import torch

from my_project.families import action_units, obs_units
from my_project.models.diffusion_policy import MinMaxNormalizer

from tests.residual.test_families import furniture_layout  # noqa: F401  (the fixture)

RESIP = "Isaac-Furniture-OneLeg-FBAbs-ResiP-v0"


def resip_cfg():  # what fbabs_obs_layout reads: the policy group has an obstacle_xy term
    return SimpleNamespace(observations=SimpleNamespace(policy=SimpleNamespace(obstacle_xy=object())))


def test_the_resip_layout_names_every_dim(furniture_layout):
    u = obs_units(RESIP, resip_cfg())
    assert len(u) == 63
    assert u[0:3] == ["pos"] * 3 and u[3:9] == ["rot6d"] * 6 and u[9:12] == ["lin_vel"] * 3
    assert u[12:15] == ["ang_vel"] * 3 and u[15] == "finger_width"
    assert u[16:19] == ["pos"] * 3 and u[19:25] == ["rot6d"] * 6  # the top
    assert u[61:63] == ["pos"] * 2  # the obstacle's x, y
    assert action_units(RESIP) == ["pos"] * 3 + ["rot6d"] * 6 + ["gripper_cmd"]
    with pytest.raises(ValueError):
        obs_units("Isaac-Forge-PegInsert-Direct-v0", None)


def test_one_range_per_unit_shared_with_the_action():
    obs_u = ["pos", "pos", "rot6d", "lin_vel"]
    act_u = ["pos", "rot6d", "gripper_cmd"]
    obs = torch.tensor([[0.0, 0.5, 0.2, -1.0], [0.1, 0.5000001, 0.3, 2.0]])
    act = torch.tensor([[0.8, -0.9, 1.0], [0.2, 0.9, -1.0]])
    n = MinMaxNormalizer(4, 3)
    n.fit_units(obs, act, obs_u, act_u)
    # positions: one range over obs dims 0, 1 and action dim 0 -> [0, 0.8]
    assert torch.allclose(n.obs_min[:2], torch.tensor([0.0, 0.0])) and torch.allclose(n.obs_max[:2], torch.tensor([0.8, 0.8]))
    assert n.action_min[0] == 0.0 and n.action_max[0] == pytest.approx(0.8)
    # the near-constant dim 1 is NOT stretched: its 1e-7 spread stays ~1e-7 after scaling
    z = n.normalize(obs, "obs")
    assert abs(float(z[1, 1] - z[0, 1])) < 1e-6
    # rot6d spans obs and action ([-0.9, 0.9]); the gripper command its own
    assert n.obs_min[2] == pytest.approx(-0.9) and n.action_max[1] == pytest.approx(0.9)
    assert n.action_min[2] == -1.0 and n.action_max[2] == 1.0
    assert torch.allclose(n.denormalize(n.normalize(act, "action"), "action"), act, atol=1e-6)
    with pytest.raises(ValueError):
        n.fit_units(obs, act, obs_u[:3], act_u)
