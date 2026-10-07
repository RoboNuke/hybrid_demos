"""Project entry point: delegates to :func:`robonuke_rl_core.debug.main`.

The ``setup`` hook runs after AppLauncher (so Isaac imports work) and before the config
loads -- every project registration goes in there, never at module top."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make my_project importable


def setup() -> None:
    import my_project.tasks  # noqa: F401  -- gym.register (imports Isaac Lab)
    import my_project.losses  # noqa: F401  -- @register_loss

    from robonuke_rl_core.config import register_section

    from my_project.cfg import DemoCfg

    register_section("demo", DemoCfg)


if __name__ == "__main__":
    from robonuke_rl_core.debug import main

    raise SystemExit(main(setup=setup))
