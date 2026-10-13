"""The BC config through the package's real load_config, with a faked task.

Checks the "no redundant config" contract: horizons and the diffusion schedule live in
``model.actor``, run length in ``trainer``, and the `bc` section rejects what cannot work.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from typing import Optional

import pytest

from robonuke_rl_core import config as cfgmod
from robonuke_rl_core.config import load_config
from robonuke_rl_core.learners import cfg as learner_cfg
from robonuke_rl_core.models.cfg import MODEL_ARCHITECTURES, register_architecture

from my_project.bc.train import EarlyStopper, action_mse_metrics
from my_project.cfg import BC_LEARNER, BCCfg
from my_project.models.cfg import DiffusionTransformerModelCfg


@dataclass
class FakeScene:
    num_envs: int = 8


@dataclass
class FakeEnvCfg:
    seed: Optional[int] = None
    scene: FakeScene = field(default_factory=FakeScene)


@pytest.fixture(autouse=True)
def registered(monkeypatch):
    monkeypatch.setattr(cfgmod, "load_task_cfg", lambda name: FakeEnvCfg())
    monkeypatch.setattr(cfgmod, "task_cfg_to_dict", lambda c: dataclasses.asdict(copy.deepcopy(c)))

    def apply(env_cfg, data):
        for key, value in data.items():
            current = getattr(env_cfg, key)
            apply(current, value) if dataclasses.is_dataclass(current) else setattr(env_cfg, key, value)
        return env_cfg

    monkeypatch.setattr(cfgmod, "apply_task_cfg", apply)
    cfgmod.register_section("bc", BCCfg)
    register_architecture("diffusion_transformer", DiffusionTransformerModelCfg)
    learner_cfg.register_learner(BC_LEARNER)
    yield
    del cfgmod.SECTIONS["bc"]
    del MODEL_ARCHITECTURES["diffusion_transformer"]
    learner_cfg.PROJECT_LEARNERS.remove(BC_LEARNER)


BASE = """
task:
  name: Fake-Task-v0
experiment:
  seed: 1
wandb:
  entity: hur
  project: p
  group: g
trainer:
  learner: diffusion_bc
  total_timesteps: 5000
  checkpoint_interval: 2000
model:
  architecture: diffusion_transformer
bc:
  dataset: /demos
"""


def load(tmp_path, *overrides, extra: str = ""):
    path = tmp_path / "exp.yaml"
    path.write_text(BASE + extra)
    return load_config(path, list(overrides))


def test_the_defaults_load_and_match_the_reference_recipe(tmp_path):
    cfg = load(tmp_path)
    assert isinstance(cfg.bc, BCCfg) and isinstance(cfg.model, DiffusionTransformerModelCfg)
    a = cfg.model.actor
    assert (a.horizon, a.n_obs_steps, a.action_horizon) == (32, 1, 8)
    assert (a.num_diffusion_iters, a.inference_steps, a.beta_schedule) == (100, 16, "squaredcos_cap_v2")
    assert (a.warm_start, a.warm_start_timestep, a.obs_clip) == (True, 50, 3.0)
    assert (cfg.bc.num_demos, cfg.bc.test_split, cfg.bc.batch_size) == (50, 0.05, 256)
    assert cfg.bc.early_stop_patience == float("inf")


def test_no_field_is_configured_twice():
    model = {f.name for f in dataclasses.fields(DiffusionTransformerModelCfg().actor)}
    bc = {f.name for f in dataclasses.fields(BCCfg)}
    assert not model & bc


@pytest.mark.parametrize(
    "override,match",
    [
        ("experiment.num_agents=2", "num_agents"),
        ("trainer.total_timesteps=1500", "multiple of bc.steps_per_epoch"),
        ("trainer.checkpoint_interval=1500", "multiple of bc.steps_per_epoch"),
        ("bc.num_demos=0", "num_demos"),
        ("bc.test_split=0.0", "test_split"),
        ("bc.lr_schedule=sawtooth", "lr_schedule"),
        ("bc.early_stop_patience=0", "early_stop_patience"),
        ("model.actor.action_horizon=40", "action_horizon"),
        ("model.actor.obs_clip=0", "obs_clip"),
        ("model.actor.warm_start_timestep=100", "warm_start_timestep"),
        ("model.actor.inference_steps=200", "inference_steps"),
    ],
)
def test_bad_values_raise_naming_the_field(tmp_path, override, match):
    with pytest.raises(ValueError, match=match):
        load(tmp_path, override)


def test_bc_needs_a_dataset(tmp_path):
    with pytest.raises(ValueError, match="bc.dataset"):
        load(tmp_path, "bc.dataset=null")


def test_the_bc_rules_only_apply_to_a_bc_run(tmp_path):
    # a SAC run carries the bc section with its defaults and is not judged by it
    cfg = load(tmp_path, "trainer.learner=sac", "model.architecture=simba",
               "experiment.num_agents=2", "trainer.total_timesteps=1500")
    assert cfg.trainer.learner == "sac"


def test_bc_needs_the_diffusion_architecture(tmp_path):
    with pytest.raises(ValueError, match="diffusion_transformer"):
        load(tmp_path, "model.architecture=simba")


# ------------------------------------------------------------------ the logging helpers
def test_early_stopper_is_the_reference_one():
    stopper = EarlyStopper(patience=2, smooth_factor=0.5)
    assert not stopper.update(1.0) and (stopper.counter, stopper.best_loss, stopper.ema_loss) == (1, 1.0, 1.0)
    assert not stopper.update(0.0) and (stopper.counter, stopper.ema_loss) == (0, 0.5)
    assert not stopper.update(2.0)  # ema 1.25: counter 1
    assert stopper.update(2.0)  # ema 1.625: counter 2 -> stop
    assert not EarlyStopper(float("inf"), 0.9).update(1.0)


def test_action_mse_metrics_name_every_group():
    import torch

    predicted, target = torch.zeros(2, 4, 7), torch.ones(2, 4, 7)
    target[..., 6] = 3.0
    out = action_mse_metrics("val", predicted, target, {"pos": [0, 1, 2], "success": [6]})
    assert set(out) == {
        "action_sample/val_action_mse_error",
        "action_sample/val_action_mse_error_pos",
        "action_sample/val_action_mse_error_success",
    }
    assert out["action_sample/val_action_mse_error_pos"] == 1.0
    assert out["action_sample/val_action_mse_error_success"] == 9.0
