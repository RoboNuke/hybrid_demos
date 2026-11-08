"""MinMaxNormalizer.NARROW_RANGE: a column constant up to float noise maps to ~0, never noise over [-1, 1]."""

import pytest
import torch

from my_project.models.diffusion_policy import MinMaxNormalizer
from my_project.residual import residual_observation


def test_a_column_constant_up_to_float_noise_maps_to_about_zero():
    # ranges: 1, 5e-7 (an obstacle x as the furniture demos record it), 0, 2e-4
    data = torch.tensor([[0.0, 0.6815, 5.0, 1.0], [1.0, 0.6815005, 5.0, 1.0002]])
    n = MinMaxNormalizer(4, 1)
    n.fit("obs", data)
    mid = float(data[:, 1].mean())
    # the 5e-7 column gets [mid - 1, mid + 1], as an exactly constant one ([5 - 1, 5 + 1]) always did
    assert float(n.obs_min[1]) == pytest.approx(mid - 1, abs=1e-6) and float(n.obs_max[1]) == pytest.approx(mid + 1, abs=1e-6)
    assert n.obs_min[2] == 4.0 and n.obs_max[2] == 6.0
    # a column at or above the threshold keeps its own range
    assert n.obs_min[0] == 0.0 and n.obs_max[0] == 1.0
    assert float(n.obs_max[3] - n.obs_min[3]) == pytest.approx(2e-4, rel=1e-3)
    # a value 1e-4 off the narrow column's data stays ~0 (it was ~400 under the exact-constant rule)
    assert abs(float(n.normalize(torch.tensor([[0.5, 0.6816, 5.0, 1.0001]]), "obs")[0, 1])) < 1e-3


def test_a_unit_constant_up_to_float_noise_is_widened_too():
    obs = torch.tensor([[0.3, 0.3000002], [0.3000001, 0.3]])
    act = torch.tensor([[0.3], [0.3000001]])
    n = MinMaxNormalizer(2, 1)
    n.fit_units(obs, act, ["pos", "pos"], ["pos"])
    assert float(n.obs_max[0] - n.obs_min[0]) == pytest.approx(2.0) and float(n.action_max[0] - n.action_min[0]) == pytest.approx(2.0)


def old_stats(obs_low, obs_high, act_low, act_high) -> MinMaxNormalizer:
    """A checkpoint's normalizer fit with the exact-constant rule: a noise-wide range kept as it was."""
    n = MinMaxNormalizer(len(obs_low), len(act_low))
    for buf, v in ((n.obs_min, obs_low), (n.obs_max, obs_high), (n.action_min, act_low), (n.action_max, act_high)):
        buf.copy_(torch.tensor(v))
    return n


def test_widen_reads_old_stats_as_the_rule_fits_them_and_the_bases_own_path_is_unchanged():
    data = torch.tensor([[0.0, 0.6815], [1.0, 0.6815005]])
    new = MinMaxNormalizer(2, 1)
    new.fit("obs", data)
    old = old_stats(data.min(0).values.tolist(), data.max(0).values.tolist(), [-1.0], [1.0])
    x = torch.tensor([[0.25, 0.6816]])
    assert torch.allclose(old.normalize(x, "obs", widen=True), new.normalize(x, "obs"))
    assert torch.equal(new.normalize(x, "obs", widen=True), new.normalize(x, "obs"))  # a no-op on stats fit since
    assert float(old.normalize(x, "obs")[0, 1]) > 100  # the base reads its stats as trained: noise stretched


def test_the_residuals_agent_observation_is_raw():
    # the residual no longer scales its inputs (the learner's state_preprocessor does): the observation and the base
    # action go out as they are
    obs, a_base = torch.tensor([[0.25, 0.6816]]), torch.tensor([[0.05, 0.2]])
    agent, base = residual_observation(obs, a_base)
    assert torch.equal(agent, torch.cat((obs, a_base), dim=-1)) and torch.equal(base, a_base)


