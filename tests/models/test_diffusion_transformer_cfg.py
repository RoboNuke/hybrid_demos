"""DiffusionTransformerModelCfg: registers as an architecture, builds, and validates."""

from __future__ import annotations

import dataclasses

import pytest
from omegaconf import OmegaConf

from robonuke_rl_core.models.cfg import MODEL_ARCHITECTURES, register_architecture

from my_project.models.cfg import DiffusionTransformerActorCfg, DiffusionTransformerModelCfg
from my_project.models.diffusion_policy import DiffusionPolicy


@pytest.fixture()
def registered():
    register_architecture("diffusion_transformer", DiffusionTransformerModelCfg)
    yield
    del MODEL_ARCHITECTURES["diffusion_transformer"]


def test_it_registers_and_round_trips_through_omegaconf(registered):
    node = OmegaConf.structured(MODEL_ARCHITECTURES["diffusion_transformer"])
    OmegaConf.set_struct(node, True)
    node.merge_with(OmegaConf.from_dotlist(["actor.n_layer=4"]))
    cfg = OmegaConf.to_object(node)
    assert isinstance(cfg, DiffusionTransformerModelCfg) and cfg.actor.n_layer == 4
    with pytest.raises(Exception, match="actor_n"):  # a SimBa field is rejected
        node.merge_with(OmegaConf.from_dotlist(["actor.actor_n=1"]))


def test_the_defaults_validate_and_build_the_network():
    cfg = DiffusionTransformerModelCfg()
    cfg.validate(None)
    policy = DiffusionPolicy(cfg.actor, obs_dim=5, action_dim=3)
    assert policy.model.horizon == cfg.actor.horizon


@pytest.mark.parametrize(
    "overrides,field",
    [
        ({"p_drop_attn": 1.0}, "p_drop_attn"),
        ({"p_drop_emb": -0.1}, "p_drop_emb"),
        ({"n_emb": 30, "n_head": 4}, "n_head"),
        ({"n_emb": 9, "n_head": 1}, "even"),
        ({"horizon": 0}, "horizon"),
        ({"n_cond_layers": -1}, "n_cond_layers"),
        ({"time_as_cond": False}, "time_as_cond"),
    ],
)
def test_bad_values_raise_naming_the_field(overrides, field):
    cfg = DiffusionTransformerModelCfg(actor=DiffusionTransformerActorCfg(**overrides))
    with pytest.raises(ValueError, match=field):
        cfg.validate(None)
