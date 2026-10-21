"""Where the residual's frozen base comes from: a local BC run directory (``residual.base_run``)
or a wandb run (``residual.base_project`` / ``base_run_name``), one of the two, and a wandb base in
another project than the residual run's own."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from my_project.cfg import ResidualCfg
from my_project.residual import base_run_dir, base_source

TASK = "Isaac-Furniture-OneLeg-FBAbs-ResiP-v0"


def cfg_for(residual: ResidualCfg, project: str = "hybrid_demo_furniture_residual"):
    return SimpleNamespace(
        residual=residual, task_name=TASK, trainer=SimpleNamespace(learner="sac"),
        action=SimpleNamespace(world_pose=False), wandb=SimpleNamespace(entity="hur", project=project),
    )


def test_the_base_is_a_local_run_or_a_wandb_run_never_both_or_neither():
    ResidualCfg(enabled=True, base_run="runs/bc").validate(cfg_for(ResidualCfg()))
    ResidualCfg(enabled=True, base_project="bc_proj", base_run_name="bc_a0").validate(cfg_for(ResidualCfg()))
    for bad in (
        ResidualCfg(enabled=True),
        ResidualCfg(enabled=True, base_run="runs/bc", base_project="bc_proj", base_run_name="bc_a0"),
        ResidualCfg(enabled=True, base_project="bc_proj"),  # no run name
        ResidualCfg(enabled=True, base_run_name="bc_a0"),  # no project
    ):
        with pytest.raises(ValueError, match="base"):
            bad.validate(cfg_for(bad))


def test_a_wandb_base_lives_in_another_project():
    rc = ResidualCfg(enabled=True, base_project="hybrid_demo_furniture", base_run_name="bc_a0")
    with pytest.raises(ValueError, match="another project"):
        rc.validate(cfg_for(rc, project="hybrid_demo_furniture"))


def test_a_wandb_base_is_downloaded_with_its_checkpoint(monkeypatch, tmp_path):
    asked = []

    def fetch(spec, which="best", cache_root=None, **kwargs):
        asked.append((spec, which))
        return tmp_path

    monkeypatch.setattr("robonuke_rl_core.evaluation.fetch_wandb_run", fetch)
    rc = ResidualCfg(enabled=True, base_project="hybrid_demo_furniture", base_run_name="physic_aligned_evalHappy_a0")
    cfg = cfg_for(rc)
    assert base_source(cfg) == "hur/hybrid_demo_furniture/physic_aligned_evalHappy_a0"  # wandb.entity by default
    assert base_run_dir(cfg) == tmp_path
    assert asked == [("hur/hybrid_demo_furniture/physic_aligned_evalHappy_a0", "actor_chkpt_best_success_rate.pt")]
    rc.base_entity = "someone"
    assert base_source(cfg).startswith("someone/")


def test_a_local_base_is_its_directory_and_nothing_is_downloaded(monkeypatch):
    monkeypatch.setattr("robonuke_rl_core.evaluation.fetch_wandb_run", lambda *a, **k: pytest.fail("downloaded"))
    cfg = cfg_for(ResidualCfg(enabled=True, base_run="~/runs/bc"))
    assert base_run_dir(cfg) == Path("~/runs/bc").expanduser() and base_source(cfg) == "~/runs/bc"
