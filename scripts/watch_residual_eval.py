"""Evaluate a residual run's checkpoints as they appear (and its base alone, as step 0):
delegates to :func:`my_project.residual_eval.main`.

The ``setup`` hook runs after AppLauncher (so Isaac imports work) and before the config
loads -- every project registration goes in there, never at module top."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make my_project importable


def setup() -> None:
    import my_project.tasks  # noqa: F401  -- gym.register (imports Isaac Lab)
    import furniture_lab.tasks  # noqa: F401  -- Isaac-Furniture-*-v0 (pip install -e ~/Robonuke_Furniture_Bench)
    import my_project.losses  # noqa: F401  -- @register_loss
    from my_project.obs_terms import install_obs_terms

    install_obs_terms()  # extra obs_order terms (hole_estimate), before any env is made

    from robonuke_rl_core.config import register_section

    from robonuke_rl_core.learners.cfg import register_learner

    from my_project.cfg import ActionCfg, BC_LEARNER, BCCfg, DemoCfg, EpisodeCfg, HybridCfg, ForceObsCfg, RecordCfg, HistoryCfg, ForgeV1Cfg, ResidualCfg

    register_section("demo", DemoCfg)
    register_section("bc", BCCfg)
    register_section("record", RecordCfg)  # scripts/record_demos.py
    register_section("action", ActionCfg)  # my_project/wrappers.py
    register_section("residual", ResidualCfg)  # my_project/residual.py
    register_section("history", HistoryCfg)  # my_project/wrappers.py
    register_section("episode", EpisodeCfg)  # my_project/wrappers.py
    register_section("hybrid", HybridCfg)  # my_project/wrappers.py
    register_section("force_obs", ForceObsCfg)  # my_project/wrappers.py
    register_section("forge_v1", ForgeV1Cfg)  # scripts/import_forge_v1.py

    from robonuke_rl_core.envs.build import register_env_wrapper

    from my_project.wrappers import project_env_wrappers

    register_env_wrapper(project_env_wrappers)  # every env the package builds

    from robonuke_rl_core.evaluation import register_eval_policy

    from my_project.bc.eval_policy import build_bc_eval_policy

    register_eval_policy(BC_LEARNER, build_bc_eval_policy)  # eval / debug of BC runs
    register_learner(BC_LEARNER)  # scripts/train_bc.py

    from my_project.models.cfg import register_all

    register_all()  # every diffusion architecture (my_project/models/cfg.py)


if __name__ == "__main__":
    from my_project.residual_eval import main

    raise SystemExit(main(setup=setup))
