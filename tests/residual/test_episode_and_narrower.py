"""``episode.reset_on_success``'s config check and the demo converter's narrower-randomization
rule (``TASK_NARROWER_ALLOWED``). CPU only; the wrapper itself needs a live Forge env."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from my_project.cfg import EpisodeCfg
from my_project.residual_demos import _narrower


def _cfg(num_envs: int):
    return SimpleNamespace(task_cfg=SimpleNamespace(scene=SimpleNamespace(num_envs=num_envs)))


def test_reset_on_success_one_env():
    EpisodeCfg(reset_on_success=True).validate(_cfg(1))
    EpisodeCfg(reset_on_success=False).validate(_cfg(64))


def test_reset_on_success_refuses_many_envs():
    with pytest.raises(ValueError, match="num_envs: 1"):
        EpisodeCfg(reset_on_success=True).validate(_cfg(2))


KEY = "obs_rand.fixed_asset_pos"


@pytest.mark.parametrize(
    "recorded, experiment, ok",
    [
        ([0.0, 0.0, 0.0], [0.002, 0.002, 0.002], True),  # the nut demos: exact estimate
        ([0.002, 0.002, 0.002], [0.002, 0.002, 0.002], True),
        ([0.003, 0.0, 0.0], [0.002, 0.002, 0.002], False),  # recorded wider in one value
        ([0.0, 0.0], [0.002, 0.002, 0.002], False),  # different length
    ],
)
def test_narrower(recorded, experiment, ok):
    assert _narrower(KEY, {KEY: recorded}, {KEY: experiment}) is ok


def test_narrower_only_listed_fields():
    key = "obs_rand.ft_force"
    assert _narrower(key, {key: [0.0]}, {key: [1.0]}) is False
