"""Where the residual's frozen base comes from: a local BC run directory (``base_policy.base_run``) or a wandb run
(``base_policy.base_project`` / ``base_run_name``), one of the two, and a wandb base in another project than the
residual run's own."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from my_project.cfg import BasePolicyCfg
from my_project.wrappers.residual_policy.base_policy import base_run_dir, base_source

TASK = "Isaac-Furniture-OneLeg-FBAbs-ResiP-v0"


def cfg_for(base: BasePolicyCfg, project: str = "hybrid_demo_furniture_residual"):
    return SimpleNamespace(base_policy=base, task_name=TASK, trainer=SimpleNamespace(learner="ddpg"),
                           wandb=SimpleNamespace(entity="hur", project=project))


def test_the_base_is_a_local_run_or_a_wandb_run_never_both_or_neither():
    BasePolicyCfg(enabled=True, base_run="runs/bc").validate(cfg_for(BasePolicyCfg()))
    BasePolicyCfg(enabled=True, base_project="bc_proj", base_run_name="bc_a0").validate(cfg_for(BasePolicyCfg()))
    for bad in (
        BasePolicyCfg(enabled=True),
        BasePolicyCfg(enabled=True, base_run="runs/bc", base_project="bc_proj", base_run_name="bc_a0"),
        BasePolicyCfg(enabled=True, base_project="bc_proj"),  # no run name
        BasePolicyCfg(enabled=True, base_run_name="bc_a0"),  # no project
    ):
        with pytest.raises(ValueError, match="base"):
            bad.validate(cfg_for(bad))


def test_a_wandb_base_lives_in_another_project():
    bc = BasePolicyCfg(enabled=True, base_project="hybrid_demo_furniture", base_run_name="bc_a0")
    with pytest.raises(ValueError, match="another project"):
        bc.validate(cfg_for(bc, project="hybrid_demo_furniture"))


def test_a_wandb_base_is_downloaded_with_its_checkpoint(monkeypatch, tmp_path):
    asked = []

    def fetch(spec, which="best", cache_root=None, **kwargs):
        asked.append((spec, which))
        return tmp_path

    monkeypatch.setattr("robonuke_rl_core.evaluation.fetch_wandb_run", fetch)
    bc = BasePolicyCfg(enabled=True, base_project="hybrid_demo_furniture", base_run_name="physic_aligned_evalHappy_a0")
    cfg = cfg_for(bc)
    assert base_source(cfg) == "hur/hybrid_demo_furniture/physic_aligned_evalHappy_a0"  # wandb.entity by default
    assert base_run_dir(cfg) == tmp_path
    assert asked == [("hur/hybrid_demo_furniture/physic_aligned_evalHappy_a0", "actor_chkpt_best_success_rate.pt")]
    bc.base_entity = "someone"
    assert base_source(cfg).startswith("someone/")


def test_a_local_base_is_its_directory_and_nothing_is_downloaded(monkeypatch):
    monkeypatch.setattr("robonuke_rl_core.evaluation.fetch_wandb_run", lambda *a, **k: pytest.fail("downloaded"))
    cfg = cfg_for(BasePolicyCfg(enabled=True, base_run="~/runs/bc"))
    assert base_run_dir(cfg) == Path("~/runs/bc").expanduser() and base_source(cfg) == "~/runs/bc"
